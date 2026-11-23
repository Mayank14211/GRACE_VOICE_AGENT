"""Shared application state: what is on, what was said. One instance per process.

Every change goes through ``AppState.update`` (or ``add_turn``), which takes a lock, applies the
change and, if anything actually changed, publishes a ``STATE`` event with an immutable
``StateSnapshot``. Readers on other threads use ``snapshot()`` or the properties; nobody mutates
fields directly. ``cause_t`` (monotonic time of the key press that caused the change) travels
with the event so the overlay can measure hotkey-to-screen latency.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

from grace.core.bus import STATE, EventBus

Mode = Literal["command", "dictation"]
Speaker = Literal["user", "grace"]


@dataclass(frozen=True)
class Turn:
    speaker: Speaker
    text: str
    t: float  # monotonic


@dataclass(frozen=True)
class StateSnapshot:
    mic_on: bool = True
    gesture_on: bool = False
    ptt_active: bool = False  # push-to-talk key is held
    listening: bool = False  # a command is being captured right now
    mode: Mode = "command"
    gesture: str = "OFF"  # name for the status chip ("SCROLL", "PAUSED", "NO CAMERA", ...)
    provider: str = "offline"  # LLM provider in use, for the status chip
    history_visible: bool = False
    turns: tuple[Turn, ...] = ()


_FIELDS = frozenset(f.name for f in fields(StateSnapshot)) - {"turns"}


class AppState:
    def __init__(
        self,
        bus: EventBus | None = None,
        *,
        max_turns: int = 10,
        clock: Callable[[], float] = time.monotonic,
        **initial: Any,
    ) -> None:
        unknown = set(initial) - _FIELDS
        if unknown:
            raise TypeError(f"unknown state fields: {sorted(unknown)}")
        self._bus = bus
        self._clock = clock
        self._lock = threading.Lock()
        self._turns: deque[Turn] = deque(maxlen=max_turns)
        self._snap = StateSnapshot(**initial)

    # ------------------------------------------------------------------ read
    def snapshot(self) -> StateSnapshot:
        with self._lock:
            return self._snap

    @property
    def mic_on(self) -> bool:
        return self.snapshot().mic_on

    @property
    def gesture_on(self) -> bool:
        return self.snapshot().gesture_on

    @property
    def ptt_active(self) -> bool:
        return self.snapshot().ptt_active

    @property
    def mode(self) -> Mode:
        return self.snapshot().mode

    # ------------------------------------------------------------------ write
    def update(self, cause_t: float | None = None, **changes: Any) -> frozenset[str]:
        """Apply ``changes``; publish one STATE event if anything changed. Returns changed names."""
        unknown = set(changes) - _FIELDS
        if unknown:
            raise TypeError(f"unknown state fields: {sorted(unknown)}")
        with self._lock:
            changed = frozenset(k for k, v in changes.items() if getattr(self._snap, k) != v)
            if changed:
                self._snap = replace(self._snap, **{k: changes[k] for k in changed})
            snap = self._snap
        if changed:
            self._publish(snap, changed, cause_t)
        return changed

    def toggle(self, name: str, cause_t: float | None = None) -> bool:
        """Flip a boolean field atomically; returns its new value."""
        if name not in _FIELDS or not isinstance(getattr(StateSnapshot, name, None), bool):
            raise TypeError(f"{name!r} is not a boolean state field")
        with self._lock:
            value = not getattr(self._snap, name)
            self._snap = replace(self._snap, **{name: value})
            snap = self._snap
        self._publish(snap, frozenset({name}), cause_t)
        return value

    def add_turn(self, speaker: Speaker, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._turns.append(Turn(speaker, text, self._clock()))
            self._snap = replace(self._snap, turns=tuple(self._turns))
            snap = self._snap
        self._publish(snap, frozenset({"turns"}), None)

    def _publish(self, snap: StateSnapshot, changed: frozenset[str], cause_t: float | None) -> None:
        if self._bus is not None:
            self._bus.publish(STATE, snapshot=snap, changed=changed, cause_t=cause_t)
