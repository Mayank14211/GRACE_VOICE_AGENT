"""Gesture loop (fake tracker) and the MediaPipe result adapter. Real camera: hardware-marked."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from gesture_synth import make_hand

from grace.config import GestureSettings
from grace.input.gestures import build_gesture_loop
from grace.input.gestures.classifier import Pose
from grace.input.gestures.loop import MAX_EMPTY_READS, NO_CAMERA, OFF
from grace.input.gestures.tracker import (
    CameraError,
    FakeTracker,
    TrackerFrame,
    observation_from_result,
)
from grace.intent.types import Intent

SCREEN = (1920, 1080)
MS = 1000 / 30


def frames(*segments: tuple[Pose | None, int], start: float = 0.0) -> list[TrackerFrame]:
    out, t = [], start
    for pose, n in segments:
        for _ in range(n):
            out.append(TrackerFrame(t, make_hand(pose) if pose else None))
            t += MS
    return out


class Harness:
    def __init__(self, tracker_frames: list[TrackerFrame], **kw: object) -> None:
        self.intents: list[Intent] = []
        self.gestures: list[str] = []
        self.trackers: list[FakeTracker] = []
        self._frames = tracker_frames

        def open_tracker() -> FakeTracker:
            t = FakeTracker(self._frames)
            self.trackers.append(t)
            return t

        self.loop = build_gesture_loop(
            GestureSettings(),
            SCREEN,
            self.intents.append,
            self.gestures.append,
            open_tracker=kw.get("open_tracker", open_tracker),  # type: ignore[arg-type]
        )

    def run(self, n: int) -> None:
        for _ in range(n):
            self.loop.step()


def test_frames_become_intents_and_gesture_names() -> None:
    h = Harness(frames((Pose.INDEX_UP, 10), (Pose.PINCH_INDEX, 6), (Pose.INDEX_UP, 6)))
    h.run(22)
    assert sum(i.name == "input.click" for i in h.intents) == 1
    assert h.gestures[:3] == ["NONE", "MOVE", "CLICK"] and h.gestures[-1] == "MOVE"
    assert h.loop.stats.frames == 22 and h.loop.stats.hand_frames == 22


def test_disable_releases_drag_closes_camera_and_shows_off() -> None:
    h = Harness(frames((Pose.PINCH_INDEX, 4 + 20)))
    h.run(24)
    assert [i.name for i in h.intents].count("input.mouse_down") == 1
    h.loop.set_enabled(False)
    assert h.loop.step() is False
    assert h.intents[-1].name == "input.mouse_up"
    assert h.trackers[0].closed and h.gestures[-1] == OFF
    h.loop.set_enabled(True)
    h.loop.step()
    assert len(h.trackers) == 2  # camera reopened


def test_missing_camera_shows_no_camera_and_retries() -> None:
    calls = []

    def broken() -> FakeTracker:
        calls.append(1)
        raise CameraError("camera 0 did not open")

    h = Harness([], open_tracker=broken)
    assert h.loop.step() is False and h.loop.step() is False
    assert h.gestures == [NO_CAMERA] and len(calls) == 2


def test_camera_that_stops_delivering_is_reopened() -> None:
    h = Harness([])  # every read returns None
    h.run(MAX_EMPTY_READS)
    assert h.trackers[0].closed and h.gestures[-1] == NO_CAMERA
    h.run(1)
    assert len(h.trackers) == 2


@pytest.mark.timeout(10)
def test_thread_start_toggle_stop() -> None:
    seen = threading.Event()
    endless = iter(lambda: TrackerFrame(0.0, make_hand(Pose.OPEN_PALM)), None)
    gestures: list[str] = []

    def on_gesture(name: str) -> None:
        gestures.append(name)
        if name == "READY":
            seen.set()

    loop = build_gesture_loop(
        GestureSettings(),
        SCREEN,
        lambda i: None,
        on_gesture,
        open_tracker=lambda: FakeTracker(endless),
    )
    loop.start()
    assert seen.wait(3)
    assert loop.toggle() is False
    loop.stop()
    assert loop.gesture in ("READY", OFF)


# --------------------------------------------------------------------------- MediaPipe adapter
def _mp_result(hand) -> SimpleNamespace:
    lms = SimpleNamespace(landmark=[SimpleNamespace(x=x, y=y, z=z) for x, y, z in hand.landmarks])
    label = SimpleNamespace(classification=[SimpleNamespace(label=hand.handedness, score=0.93)])
    return SimpleNamespace(multi_hand_landmarks=[lms], multi_handedness=[label])


def test_observation_from_mediapipe_result() -> None:
    hand = make_hand(Pose.TWO_UP, "Left")
    obs = observation_from_result(_mp_result(hand), aspect=4 / 3)
    assert obs is not None and obs.handedness == "Left" and obs.score == 0.93
    assert obs.landmarks == hand.landmarks
    empty = SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
    assert observation_from_result(empty, 4 / 3) is None


@pytest.mark.hardware
def test_mediapipe_tracker_reads_frames() -> None:
    from grace.input.gestures.tracker import MediaPipeTracker

    tracker = MediaPipeTracker(0)
    try:
        got = [tracker.read() for _ in range(10)]
    finally:
        tracker.close()
    assert sum(f is not None for f in got) >= 8
