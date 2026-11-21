"""Pure classifier: fixtures (both hands), rotation, handedness-based thumb, pinch hysteresis."""

from __future__ import annotations

import math
import typing
from pathlib import Path

import pytest
from gesture_synth import FIXTURE_DIR, make_hand, variants

from grace.config import GestureBehaviour, GesturePose
from grace.input.gestures.classifier import (
    INDEX_TIP,
    MIDDLE_MCP,
    THUMB_TIP,
    WRIST,
    ClassifierConfig,
    HandObservation,
    Pose,
    classify,
    features,
    palm_facing_camera,
    read_hand,
)
from grace.input.gestures.controller import Behaviour
from grace.input.gestures.fixtures import dump_fixture, load_fixture

FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))
REAL_POSES = [p for p in Pose if p is not Pose.NONE]


# --------------------------------------------------------------------------- fixtures
def test_fixtures_cover_every_pose_with_both_hands() -> None:
    have = {(load_fixture(p).pose, load_fixture(p).hand) for p in FIXTURES}
    missing = [(p.value, h) for p in REAL_POSES for h in ("Right", "Left") if (p, h) not in have]
    assert not missing


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_frames_classify_as_their_pose(path: Path) -> None:
    fx = load_fixture(path)
    assert len(fx.hands) >= 20
    prev: Pose | None = None
    results = []
    for hand in fx.hands:
        assert hand.handedness == fx.hand
        prev = classify(hand, prev)
        results.append(prev)
    share = results.count(fx.pose) / len(results)
    # Synthetic stand-ins must be perfect; real recordings may contain a stray frame (debounce
    # needs 4 in a row anyway).
    need = 1.0 if fx.source == "synthetic" else 0.9
    assert share >= need, f"{path.name}: {share:.0%} {[r.value for r in results]}"


def test_fixture_roundtrip(tmp_path: Path) -> None:
    hand = make_hand(Pose.FIST, "Left")
    doc = {
        "version": 1,
        "pose": "fist",
        "hand": "Left",
        "source": "recorded",
        "aspect": hand.aspect,
        "mirrored": True,
        "frames": [
            {"handedness": "Left", "score": 0.9, "landmarks": [list(p) for p in hand.landmarks]}
        ],
    }
    path = tmp_path / "x.json"
    path.write_text(dump_fixture(doc), encoding="utf-8")
    fx = load_fixture(path)
    assert fx.pose is Pose.FIST and fx.hand == "Left" and fx.hands[0].landmarks == hand.landmarks


def test_fixture_from_unmirrored_frames_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text(
        '{"version": 1, "pose": "fist", "hand": "Left", "mirrored": false, "frames": []}'
    )
    with pytest.raises(ValueError, match="mirrored"):
        load_fixture(path)


# --------------------------------------------------------------------------- invariance
@pytest.mark.parametrize("pose", REAL_POSES, ids=str)
@pytest.mark.parametrize("hand", ["Right", "Left"])
@pytest.mark.parametrize("rotation", [-35, -15, 0, 15, 35])
def test_every_pose_both_hands_any_tilt(pose: Pose, hand: str, rotation: float) -> None:
    for v in range(variants(pose)):
        for size in (0.12, 0.2, 0.3):
            h = make_hand(pose, hand, rotation_deg=rotation, variant=v, size=size)
            assert classify(h) is pose, (pose, hand, rotation, v, size)


@pytest.mark.parametrize("aspect", [4 / 3, 16 / 9, 1.0])
def test_aspect_ratio_is_corrected(aspect: float) -> None:
    for pose in REAL_POSES:
        assert classify(make_hand(pose, "Right", aspect=aspect)) is pose


def test_no_hand_is_none() -> None:
    assert classify(None) is Pose.NONE
    assert read_hand(None) is None


# --------------------------------------------------------------------------- thumb + handedness
def _relabel(hand: HandObservation, label: str) -> HandObservation:
    return HandObservation(hand.landmarks, label, hand.score, hand.aspect)  # type: ignore[arg-type]


def test_palm_facing_uses_handedness() -> None:
    right = make_hand(Pose.OPEN_PALM, "Right")
    left = make_hand(Pose.OPEN_PALM, "Left")
    assert palm_facing_camera(right) and palm_facing_camera(left)
    # A mirror image labelled as the other hand is that hand seen from the back.
    assert not palm_facing_camera(_relabel(left, "Right"))
    assert not palm_facing_camera(_relabel(right, "Left"))


@pytest.mark.parametrize(
    "pose", [Pose.OPEN_PALM, Pose.INDEX_UP, Pose.TWO_UP, Pose.THUMB_INDEX_MIDDLE]
)
def test_thumb_correct_for_back_of_hand(pose: Pose) -> None:
    # Right hand, back to the camera = the left-hand picture with a "Right" label.
    back = _relabel(make_hand(pose, "Left", rotation_deg=10), "Right")
    assert features(back).extended == features(make_hand(pose, "Right")).extended


def _legacy_thumb_extended(h: HandObservation) -> bool:
    """Hand_Gesture.py's raw x-ordering test, kept here only to prove the regression."""
    tip, ip, wrist = h.landmarks[THUMB_TIP], h.landmarks[3], h.landmarks[WRIST]
    return tip[0] < ip[0] if tip[0] < wrist[0] else tip[0] > ip[0]


@pytest.mark.parametrize(
    ("hand", "rotation"), [("Right", -20), ("Right", -40), ("Left", 20), ("Left", 40)]
)
def test_folded_thumb_on_tilted_hand_is_not_extended(hand: str, rotation: float) -> None:
    # Tilted toward the thumb side, a thumb folded across the palm passes the wrist's x and the
    # legacy test called it extended (two fingers read as "volume" instead of "scroll").
    h = make_hand(Pose.TWO_UP, hand, rotation_deg=rotation)
    assert _legacy_thumb_extended(h) is True
    assert features(h).extended[0] is False
    assert classify(h) is Pose.TWO_UP


# --------------------------------------------------------------------------- pinch
def _with_pinch_ratio(ratio: float, hand: str = "Right") -> HandObservation:
    """A thumb-index pinch whose thumb tip is moved to exactly ``ratio`` x hand size away."""
    base = make_hand(Pose.PINCH_INDEX, hand)
    lm = list(base.landmarks)
    a = base.aspect
    size = math.hypot((lm[MIDDLE_MCP][0] - lm[WRIST][0]) * a, lm[MIDDLE_MCP][1] - lm[WRIST][1])
    ix, iy, _ = lm[INDEX_TIP]
    tx, ty, _ = lm[THUMB_TIP]
    dx, dy = (tx - ix) * a, ty - iy
    n = math.hypot(dx, dy)
    scale = ratio * size / n
    lm[THUMB_TIP] = (ix + dx * scale / a, iy + dy * scale, 0.0)
    return HandObservation(tuple(lm), base.handedness, base.score, a)


def test_pinch_is_distance_over_hand_size() -> None:
    for size in (0.12, 0.3):
        f = features(make_hand(Pose.PINCH_INDEX, "Right", size=size))
        assert f.pinch_index < 0.15
    assert features(_with_pinch_ratio(0.27)).pinch_index == pytest.approx(0.27, abs=1e-6)


def test_pinch_hysteresis() -> None:
    cfg = ClassifierConfig(pinch_on=0.22, pinch_off=0.32)
    between = _with_pinch_ratio(0.27)
    assert classify(between, prev=None, cfg=cfg) is not Pose.PINCH_INDEX  # not yet pinching
    assert classify(between, prev=Pose.PINCH_INDEX, cfg=cfg) is Pose.PINCH_INDEX  # stays pinched
    assert classify(_with_pinch_ratio(0.35), prev=Pose.PINCH_INDEX, cfg=cfg) is not Pose.PINCH_INDEX
    assert classify(_with_pinch_ratio(0.18), prev=None, cfg=cfg) is Pose.PINCH_INDEX


def test_fist_is_never_a_pinch() -> None:
    for hand in ("Right", "Left"):
        for rot in (-30, 0, 30):
            f = features(make_hand(Pose.FIST, hand, rotation_deg=rot))
            assert f.pinch_middle < 0.22  # tips do touch: only the reach guard tells them apart
            assert (
                classify(make_hand(Pose.FIST, hand, rotation_deg=rot), prev=Pose.PINCH_MIDDLE)
                is Pose.FIST
            )


def test_reading_points() -> None:
    h = make_hand(Pose.INDEX_UP, "Right", center=(0.4, 0.6))
    r = read_hand(h)
    assert r is not None and r.pose is Pose.INDEX_UP
    assert r.pointer == pytest.approx(h.landmarks[INDEX_TIP][:2])
    assert r.palm == pytest.approx((0.4, 0.6), abs=1e-6)
    assert r.pointer[1] < r.palm[1]  # fingertip above the palm (image y down)


# --------------------------------------------------------------------------- validation
def test_observation_validation() -> None:
    good = make_hand(Pose.FIST)
    with pytest.raises(ValueError, match="21"):
        HandObservation(good.landmarks[:20], "Right")
    with pytest.raises(ValueError, match="handedness"):
        HandObservation(good.landmarks, "Up")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ClassifierConfig(pinch_on=0.4, pinch_off=0.3)


def test_settings_names_match_enums() -> None:
    assert set(typing.get_args(GesturePose)) == {p.value for p in REAL_POSES}
    assert set(typing.get_args(GestureBehaviour)) == {b.value for b in Behaviour}
