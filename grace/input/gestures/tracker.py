"""Camera + MediaPipe Hands -> HandObservation per frame. Headless: no preview window, no win32.

The frame is flipped horizontally before detection. That makes the cursor mirror the hand (move
right -> cursor right) and is what MediaPipe expects for a correct handedness label ("it
determines handedness assuming the input image is mirrored", mediapipe/python/solutions/hands.py).

Uses the legacy ``mp.solutions.hands`` API, present in the pinned mediapipe 0.10.14 (the doctor
checks it). cv2 and mediapipe are imported lazily so the pure modules import without them.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from grace.input.gestures.classifier import HandObservation

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackerFrame:
    t_ms: float  # monotonic capture time
    hand: HandObservation | None  # None = camera frame without a hand


class CameraError(RuntimeError):
    """The camera could not be opened or stopped delivering frames."""


class HandTracker(Protocol):
    def read(self) -> TrackerFrame | None:
        """Next frame (blocks for at most one camera frame). None = no frame from the camera."""
        ...

    def close(self) -> None: ...


class MediaPipeTracker:
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        *,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        import cv2
        import mediapipe as mp

        self._cv2 = cv2
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(camera_index, backend)
        if not self._cap.isOpened():
            self._cap.release()
            raise CameraError(f"camera {camera_index} did not open (in use or missing?)")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.camera_index = camera_index

    def read(self) -> TrackerFrame | None:
        ok, frame = self._cap.read()
        t_ms = time.monotonic() * 1000
        if not ok or frame is None:
            return None
        cv2 = self._cv2
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # lets MediaPipe skip a copy
        h, w = rgb.shape[:2]
        result = self._hands.process(rgb)
        return TrackerFrame(t_ms, observation_from_result(result, aspect=w / h))

    def close(self) -> None:
        try:
            self._hands.close()
        finally:
            self._cap.release()


def observation_from_result(result: Any, aspect: float) -> HandObservation | None:
    """First hand of a ``Hands.process`` result, or None."""
    hands = getattr(result, "multi_hand_landmarks", None)
    labels = getattr(result, "multi_handedness", None)
    if not hands or not labels:
        return None
    cls = labels[0].classification[0]
    points = [(lm.x, lm.y, lm.z) for lm in hands[0].landmark]
    return HandObservation.from_lists(points, cls.label, cls.score, aspect)


class FakeTracker:
    """Replays observations (tests): (t_ms, hand) pairs or TrackerFrames; None = no frame."""

    def __init__(
        self, frames: Iterable[TrackerFrame | tuple[float, HandObservation | None] | None]
    ):
        self._frames: Iterator[Any] = iter(frames)
        self.closed = False
        self.reads = 0

    def read(self) -> TrackerFrame | None:
        self.reads += 1
        try:
            item = next(self._frames)
        except StopIteration:
            return None
        if item is None or isinstance(item, TrackerFrame):
            return item
        return TrackerFrame(float(item[0]), item[1])

    def close(self) -> None:
        self.closed = True
