"""Gesture worker thread: tracker -> controller -> intents, plus the active gesture name.

The loop owns the controller: other threads only call ``set_enabled`` / ``stop``, and the loop
applies those between frames, so the state machine is never touched from two threads.
``on_intent`` must not block (pass ``ActionExecutor.submit``); ``on_gesture`` receives the name
for the overlay status chip whenever it changes. Turning gestures off releases a held drag and
closes the camera. There is no sleep in the frame path; waits only happen while disabled or
while the camera is unavailable, and they wake immediately on stop.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from grace.input.gestures.controller import GestureController
from grace.input.gestures.tracker import HandTracker
from grace.intent.types import Intent

log = logging.getLogger(__name__)

OFF = "OFF"
NO_CAMERA = "NO CAMERA"
MAX_EMPTY_READS = 30  # consecutive failed reads before the camera is reopened


@dataclass
class LoopStats:
    frames: int = 0
    hand_frames: int = 0
    intents: int = 0
    process_ms_total: float = 0.0

    @property
    def mean_process_ms(self) -> float:
        return self.process_ms_total / self.frames if self.frames else 0.0


class GestureLoop:
    def __init__(
        self,
        open_tracker: Callable[[], HandTracker],
        controller: GestureController,
        on_intent: Callable[[Intent], None],
        on_gesture: Callable[[str], None] | None = None,
        *,
        enabled: bool = True,
        retry_s: float = 2.0,
    ) -> None:
        self._open_tracker = open_tracker
        self.controller = controller
        self._on_intent = on_intent
        self._on_gesture = on_gesture
        self._want_enabled = enabled
        self._enabled_applied: bool | None = None
        self._retry_s = retry_s
        self._tracker: HandTracker | None = None
        self._empty_reads = 0
        self._gesture: str | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = LoopStats()

    # ------------------------------------------------------------------ control (any thread)
    @property
    def enabled(self) -> bool:
        return self._want_enabled

    def set_enabled(self, on: bool) -> None:
        self._want_enabled = on
        self._wake.set()

    def toggle(self) -> bool:
        self.set_enabled(not self._want_enabled)
        return self._want_enabled

    @property
    def gesture(self) -> str | None:
        return self._gesture

    def start(self) -> GestureLoop:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="grace-gestures", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("gesture loop did not stop within %.1f s", timeout)
            self._thread = None
        else:
            self._shutdown()

    # ------------------------------------------------------------------ loop thread
    def _publish(self, name: str) -> None:
        if name != self._gesture:
            self._gesture = name
            if self._on_gesture is not None:
                try:
                    self._on_gesture(name)
                except Exception:  # noqa: BLE001 - a broken UI callback must not stop gestures
                    log.exception("gesture name callback failed")

    def _emit(self, intents: list[Intent]) -> None:
        for intent in intents:
            self.stats.intents += 1
            self._on_intent(intent)

    def _close_tracker(self) -> None:
        if self._tracker is not None:
            try:
                self._tracker.close()
            except Exception:  # noqa: BLE001
                log.exception("closing the camera failed")
            self._tracker = None

    def _shutdown(self) -> None:
        self._emit(self.controller.reset())
        self._close_tracker()

    def step(self) -> bool:
        """One iteration. Returns False when the loop should wait (disabled / no camera)."""
        if self._want_enabled != self._enabled_applied:
            self._enabled_applied = self._want_enabled
            if not self._want_enabled:
                self._shutdown()
                log.info("gestures off")
            else:
                log.info("gestures on")
        if not self._want_enabled:
            self._publish(OFF)
            return False
        if self._tracker is None:
            try:
                self._tracker = self._open_tracker()
                self._empty_reads = 0
            except Exception as exc:  # noqa: BLE001 - CameraError, missing mediapipe, ...
                log.warning("gesture camera unavailable: %s", exc)
                self._publish(NO_CAMERA)
                return False
        frame = self._tracker.read()
        if frame is None:
            self._empty_reads += 1
            if self._empty_reads >= MAX_EMPTY_READS:
                log.warning("camera stopped delivering frames; reopening")
                self._emit(self.controller.reset())
                self._close_tracker()
                self._publish(NO_CAMERA)
                return False
            return True
        self._empty_reads = 0
        started = time.perf_counter()
        self._emit(self.controller.update(frame.hand, frame.t_ms))
        self.stats.frames += 1
        self.stats.hand_frames += frame.hand is not None
        self.stats.process_ms_total += (time.perf_counter() - started) * 1000
        self._publish(self.controller.gesture)
        return True

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    active = self.step()
                except Exception:  # noqa: BLE001 - keep the thread alive, retry after a pause
                    log.exception("gesture loop error")
                    self._emit(self.controller.reset())
                    self._close_tracker()
                    active = False
                if not active:
                    # Disabled: wait for set_enabled/stop. No camera: retry after retry_s.
                    self._wake.wait(None if not self._want_enabled else self._retry_s)
                    self._wake.clear()
        finally:
            self._shutdown()
