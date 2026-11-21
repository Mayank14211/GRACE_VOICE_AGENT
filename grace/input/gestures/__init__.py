"""Hand-gesture input: tracker (camera + MediaPipe) -> classifier (pure) -> controller (pure).

    loop = build_gesture_loop(settings, on_intent=executor.submit, on_gesture=print)
    loop.start(); loop.set_enabled(False); loop.stop()

Only ``tracker`` touches the camera, and only lazily; every OS effect is an Intent that the caller
sends to the action registry (through ``grace.actions.executor.ActionExecutor``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from grace.input.gestures.classifier import HandObservation, Pose, Reading, classify, read_hand
from grace.input.gestures.controller import (
    Behaviour,
    ControllerConfig,
    GestureController,
)
from grace.input.gestures.loop import GestureLoop
from grace.intent.types import Intent

if TYPE_CHECKING:
    from grace.config import GestureSettings
    from grace.input.gestures.tracker import HandTracker

__all__ = [
    "Behaviour",
    "ControllerConfig",
    "GestureController",
    "GestureLoop",
    "HandObservation",
    "Pose",
    "Reading",
    "build_gesture_loop",
    "classify",
    "open_mediapipe_tracker",
    "read_hand",
]


def open_mediapipe_tracker(gs: GestureSettings) -> HandTracker:
    from grace.input.gestures.tracker import MediaPipeTracker

    return MediaPipeTracker(
        gs.camera_index,
        gs.frame_width,
        gs.frame_height,
        model_complexity=gs.model_complexity,
        min_detection_confidence=gs.min_detection_confidence,
        min_tracking_confidence=gs.min_tracking_confidence,
    )


def build_gesture_loop(
    gs: GestureSettings,
    screen_size: tuple[int, int],
    on_intent: Callable[[Intent], None],
    on_gesture: Callable[[str], None] | None = None,
    *,
    open_tracker: Callable[[], HandTracker] | None = None,
    enabled: bool = True,
) -> GestureLoop:
    controller = GestureController(ControllerConfig.from_settings(gs, screen_size))
    return GestureLoop(
        open_tracker or (lambda: open_mediapipe_tracker(gs)),
        controller,
        on_intent,
        on_gesture,
        enabled=enabled,
    )
