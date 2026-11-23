"""Earcons: short tones that confirm a toggle without looking at the screen.

``BeepEarcons.play(name)`` returns at once; a single daemon thread plays the tones with
``winsound.Beep`` (Windows only, no extra dependency). If several toggles arrive faster than the
tones play, only the newest few are kept, so sound never lags behind the state.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from collections.abc import Callable
from typing import Protocol

log = logging.getLogger(__name__)

# name -> (frequency Hz, duration ms) tones played back to back
TONES: dict[str, tuple[tuple[int, int], ...]] = {
    "on": ((660, 50), (990, 70)),  # rising: something turned on
    "off": ((990, 50), (660, 70)),  # falling: something turned off
    "wake": ((740, 40), (988, 60)),  # wake word heard: say the command
    "ptt_start": ((880, 40),),
    "ptt_end": ((620, 40),),
    "panic": ((400, 90), (260, 160)),
    "confirm": ((880, 50), (880, 50)),
    "marked": ((520, 60), (520, 60), (520, 60)),
    "error": ((220, 180),),
}
MAX_PENDING = 2


class Earcons(Protocol):
    def play(self, name: str) -> None: ...

    def close(self) -> None: ...


class SilentEarcons:
    def play(self, name: str) -> None:
        pass

    def close(self) -> None:
        pass


class FakeEarcons:
    def __init__(self) -> None:
        self.played: list[str] = []

    def play(self, name: str) -> None:
        if name not in TONES:
            raise KeyError(name)
        self.played.append(name)

    def close(self) -> None:
        pass


def _winsound_beep() -> Callable[[int, int], None]:
    import winsound

    return winsound.Beep


class BeepEarcons:
    def __init__(self, beep: Callable[[int, int], None] | None = None) -> None:
        self._beep = beep or _winsound_beep()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="grace-earcons", daemon=True)
        self._thread.start()

    def play(self, name: str) -> None:
        if name not in TONES:
            raise KeyError(name)
        while self._queue.qsize() >= MAX_PENDING:  # drop stale tones, keep the newest
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(name)

    def _run(self) -> None:
        while True:
            name = self._queue.get()
            if name is None:
                return
            for freq, ms in TONES[name]:
                try:
                    self._beep(freq, ms)
                except Exception:  # noqa: BLE001 - no audio device: stay silent, keep running
                    log.debug("earcon %s failed", name, exc_info=True)
                    break

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(1.0)


def create_earcons(enabled: bool = True) -> Earcons:
    if not enabled or sys.platform != "win32":
        return SilentEarcons()
    try:
        return BeepEarcons()
    except Exception as exc:  # noqa: BLE001
        log.warning("earcons unavailable: %s", exc)
        return SilentEarcons()
