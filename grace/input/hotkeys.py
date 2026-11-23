"""Global hotkeys (the PRIMARY toggles), read from settings.yaml ``hotkeys:``.

    manager = HotkeyManager(settings.hotkeys, handlers, KeyboardLibBackend()).start()

``handlers`` maps every name in ``HOTKEY_NAMES`` (plus ``PTT_RELEASE``) to a callable that gets
the monotonic time of the key press. Handlers run on the keyboard hook thread, so they must
return quickly: flip state, wake a worker, never do I/O or wait.

Why the latch: the ``keyboard`` library calls a hotkey callback on every key-down that matches,
and Windows auto-repeats a held key ~30 times a second, so holding Ctrl+Alt+M would toggle the
mic over and over. A hotkey therefore fires once and stays latched until one of its keys is
released. ``trigger_on_release`` can't be used for push-to-talk: the library reads its pressed-key
table after the released key was removed, so release hotkeys don't fire reliably. Releases come
from a raw key-up hook instead. If a key-up is lost (e.g. the key was released while an elevated
window had focus), the latch expires once no matching key-down arrived for ``STALE_S``; auto-
repeat keeps a really held key fresh.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from grace.config import HotkeySettings

log = logging.getLogger(__name__)

HOTKEY_NAMES: tuple[str, ...] = tuple(HotkeySettings.model_fields)
PUSH_TO_TALK = "push_to_talk"
PTT_RELEASE = "push_to_talk_release"
STALE_S = 1.0  # auto-repeat sends a key-down every ~33 ms; 1 s of silence = released

Handler = Callable[[float], None]


class KeyboardBackend(Protocol):
    def add_hotkey(self, combo: str, callback: Callable[[], None]) -> Any: ...

    def remove_hotkey(self, handle: Any) -> None: ...

    def on_key_up(self, callback: Callable[[int], None]) -> Any:
        """Call ``callback(scan_code)`` for every key release, from any application."""
        ...

    def unhook(self, handle: Any) -> None: ...

    def scan_codes(self, combo: str) -> frozenset[int]:
        """Scan codes of every key in ``combo`` (both left and right modifiers)."""
        ...


class HotkeyManager:
    def __init__(
        self,
        settings: HotkeySettings,
        handlers: Mapping[str, Handler],
        backend: KeyboardBackend,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        missing = [n for n in (*HOTKEY_NAMES, PTT_RELEASE) if n not in handlers]
        if missing:
            raise ValueError(f"missing hotkey handlers: {missing}")
        self.combos: dict[str, str] = {name: getattr(settings, name) for name in HOTKEY_NAMES}
        self._handlers = dict(handlers)
        self._backend = backend
        self._clock = clock
        self._lock = threading.Lock()
        self._codes: dict[str, frozenset[int]] = {}
        self._latched: dict[str, float] = {}  # name -> time of the last matching key-down
        self._handles: list[Any] = []
        self._key_up_handle: Any = None
        self.failed: dict[str, str] = {}  # name -> why it could not be registered

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> HotkeyManager:
        for name, combo in self.combos.items():
            try:
                self._codes[name] = self._backend.scan_codes(combo)
                self._handles.append(
                    self._backend.add_hotkey(combo, lambda name=name: self.pressed(name))
                )
            except Exception as exc:  # noqa: BLE001 - one bad combo must not stop the others
                self.failed[name] = str(exc)
                log.warning("hotkey %s (%s) not registered: %s", name, combo, exc)
        try:
            self._key_up_handle = self._backend.on_key_up(self.released)
        except Exception as exc:  # noqa: BLE001
            log.warning("key-up hook unavailable (push-to-talk release won't be seen): %s", exc)
        registered = len(self.combos) - len(self.failed)
        log.info("hotkeys: %d of %d registered", registered, len(self.combos))
        return self

    def stop(self) -> None:
        for handle in self._handles:
            try:
                self._backend.remove_hotkey(handle)
            except Exception:  # noqa: BLE001
                log.debug("remove_hotkey failed", exc_info=True)
        self._handles.clear()
        if self._key_up_handle is not None:
            try:
                self._backend.unhook(self._key_up_handle)
            except Exception:  # noqa: BLE001
                log.debug("unhook failed", exc_info=True)
            self._key_up_handle = None
        with self._lock:
            ptt_held = self._latched.pop(PUSH_TO_TALK, None) is not None
            self._latched.clear()
        if ptt_held:
            self._call(PTT_RELEASE, self._clock())

    # ------------------------------------------------------------------ events (hook thread)
    def pressed(self, name: str) -> bool:
        """A key-down matched ``name``. Returns True when it fired (not a repeat)."""
        now = self._clock()
        with self._lock:
            last = self._latched.get(name)
            self._latched[name] = now
            fire = last is None or now - last >= STALE_S
            stale_ptt = name == PUSH_TO_TALK and last is not None and fire
        if stale_ptt:  # the release of the previous hold was lost
            self._call(PTT_RELEASE, now)
        if fire:
            self._call(name, now)
        return fire

    def released(self, scan_code: int) -> None:
        now = self._clock()
        with self._lock:
            names = [n for n in self._latched if scan_code in self._codes.get(n, ())]
            for n in names:
                del self._latched[n]
        if PUSH_TO_TALK in names:
            self._call(PTT_RELEASE, now)

    def expire_stale(self) -> None:
        """Drop latches whose key-up was lost. Call periodically (the app does, every 250 ms)."""
        now = self._clock()
        with self._lock:
            stale = [n for n, t in self._latched.items() if now - t >= STALE_S]
            for n in stale:
                del self._latched[n]
        if PUSH_TO_TALK in stale:
            self._call(PTT_RELEASE, now)

    def _call(self, name: str, t: float) -> None:
        try:
            self._handlers[name](t)
        except Exception:  # noqa: BLE001 - never let an exception reach the keyboard hook
            log.exception("hotkey handler %s failed", name)


# --------------------------------------------------------------------------- backends
class KeyboardLibBackend:
    """The ``keyboard`` package (global low-level hook; Windows). Keys are not suppressed."""

    def __init__(self) -> None:
        import keyboard

        self._kb = keyboard

    def add_hotkey(self, combo: str, callback: Callable[[], None]) -> Any:
        return self._kb.add_hotkey(combo, callback, suppress=False)

    def remove_hotkey(self, handle: Any) -> None:
        self._kb.remove_hotkey(handle)

    def on_key_up(self, callback: Callable[[int], None]) -> Any:
        kb = self._kb

        def hook(event: Any) -> None:
            if event.event_type == kb.KEY_UP:
                callback(event.scan_code)

        return kb.hook(hook)

    def unhook(self, handle: Any) -> None:
        self._kb.unhook(handle)

    def scan_codes(self, combo: str) -> frozenset[int]:
        codes: set[int] = set()
        for part in combo.split("+"):
            codes.update(self._kb.key_to_scan_codes(part))
        return frozenset(codes)


class FakeKeyboard:
    """In-memory backend for tests: ``press("ctrl+alt+m")``, ``release("m")``."""

    def __init__(self, reject: tuple[str, ...] = ()) -> None:
        self.hotkeys: dict[str, Callable[[], None]] = {}
        self.key_up: Callable[[int], None] | None = None
        self._reject = reject
        self._ids = itertools.count(1)
        self._code_of: dict[str, int] = {}

    def code(self, key: str) -> int:
        return self._code_of.setdefault(key, next(self._ids))

    def add_hotkey(self, combo: str, callback: Callable[[], None]) -> Any:
        if combo in self._reject:
            raise ValueError(f"unknown key in {combo!r}")
        self.hotkeys[combo] = callback
        return combo

    def remove_hotkey(self, handle: Any) -> None:
        self.hotkeys.pop(handle, None)

    def on_key_up(self, callback: Callable[[int], None]) -> Any:
        self.key_up = callback
        return "key_up"

    def unhook(self, handle: Any) -> None:
        self.key_up = None

    def scan_codes(self, combo: str) -> frozenset[int]:
        return frozenset(self.code(part) for part in combo.split("+"))

    def press(self, combo: str) -> None:
        self.hotkeys[combo]()

    def release(self, key: str) -> None:
        if self.key_up is not None:
            self.key_up(self.code(key))
