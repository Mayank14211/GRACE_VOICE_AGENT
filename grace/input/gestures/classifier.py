"""Pure hand-pose classifier: 21 MediaPipe landmarks + handedness -> Pose. No hardware, no state.

Coordinates are MediaPipe's normalised image coordinates of a **mirrored** frame (the tracker flips
the camera image first, so the picture is what the user sees in a mirror and MediaPipe's
handedness label is the user's real hand). ``x`` is normalised by the frame width and ``y`` by
the height, so every distance here first rescales ``x`` by the frame aspect ratio.

Rules (all scale-free: lengths are divided by the hand size, wrist -> middle knuckle):

- A finger is extended when its tip is farther from the wrist than its PIP joint. This works at
  any hand rotation, unlike the old "tip above PIP" test.
- The thumb is extended when its tip sticks out sideways past the index knuckle, on the thumb
  side of the palm. The thumb side comes from the handedness label (a right hand seen palm-first
  in a mirror has its thumb on the left), flipped when the back of the hand faces the camera; the
  facing comes from handedness + the winding of wrist, index knuckle and pinky knuckle. No raw x
  ordering, so it works with the hand tilted.
- A pinch is thumb tip to fingertip distance / hand size below ``pinch_on``; once pinching it
  stays a pinch until the ratio rises above ``pinch_off`` (hysteresis). Both tips must also be
  away from the palm centre, otherwise a fist (thumb lying on the curled fingers) would read as a
  pinch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

Point = tuple[float, float]
Landmark = tuple[float, float, float]
Handedness = Literal["Left", "Right"]

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
NUM_LANDMARKS = 21

FINGERS = (  # (MCP, PIP, TIP) for index, middle, ring, pinky
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
)
PALM = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


class Pose(StrEnum):
    """Hand poses. Values are the keys used in settings.yaml ``gestures.map``."""

    OPEN_PALM = "open_palm"
    FIST = "fist"
    INDEX_UP = "index_up"
    TWO_UP = "two_up"  # index + middle
    THREE_UP = "three_up"  # index + middle + ring
    THUMB_INDEX_MIDDLE = "thumb_index_middle"
    PINCH_INDEX = "pinch_index"  # thumb tip on index tip
    PINCH_MIDDLE = "pinch_middle"  # thumb tip on middle tip
    NONE = "none"  # no hand, or a hand shape with no meaning


PINCHES = (Pose.PINCH_INDEX, Pose.PINCH_MIDDLE)

# (thumb, index, middle, ring, pinky) extended -> pose. None = "either".
_PATTERNS: tuple[tuple[tuple[bool | None, ...], Pose], ...] = (
    ((True, True, True, True, True), Pose.OPEN_PALM),
    ((False, True, True, True, True), Pose.OPEN_PALM),  # thumb tucked: still an open hand
    ((False, False, False, False, False), Pose.FIST),
    ((None, True, False, False, False), Pose.INDEX_UP),  # thumb either way: pointing
    ((False, True, True, False, False), Pose.TWO_UP),
    ((True, True, True, False, False), Pose.THUMB_INDEX_MIDDLE),
    ((False, True, True, True, False), Pose.THREE_UP),
)


@dataclass(frozen=True)
class HandObservation:
    """One detected hand in one frame."""

    landmarks: tuple[Landmark, ...]  # 21 x (x, y, z), normalised, mirrored frame
    handedness: Handedness  # the user's real hand (frame was mirrored before detection)
    score: float = 1.0
    aspect: float = 4 / 3  # frame width / height

    def __post_init__(self) -> None:
        if len(self.landmarks) != NUM_LANDMARKS:
            raise ValueError(f"expected {NUM_LANDMARKS} landmarks, got {len(self.landmarks)}")
        if self.handedness not in ("Left", "Right"):
            raise ValueError(f"handedness must be 'Left' or 'Right', got {self.handedness!r}")
        if self.aspect <= 0:
            raise ValueError("aspect must be positive")

    @classmethod
    def from_lists(
        cls,
        landmarks: Sequence[Sequence[float]],
        handedness: str,
        score: float = 1.0,
        aspect: float = 4 / 3,
    ) -> HandObservation:
        pts = tuple(
            (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0) for p in landmarks
        )
        return cls(pts, handedness, float(score), float(aspect))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ClassifierConfig:
    pinch_on: float = 0.22
    pinch_off: float = 0.32
    pinch_min_reach: float = 0.28  # both tips at least this far (x hand size) from the palm centre
    thumb_out: float = 0.25  # thumb tip this far (x hand size) past the index knuckle = extended

    def __post_init__(self) -> None:
        if not 0 < self.pinch_on < self.pinch_off:
            raise ValueError("need 0 < pinch_on < pinch_off")


@dataclass(frozen=True)
class Features:
    extended: tuple[bool, bool, bool, bool, bool]  # thumb, index, middle, ring, pinky
    pinch_index: float  # thumb-index tip distance / hand size
    pinch_middle: float
    pinch_index_reach: bool  # thumb and index tips clear of the palm centre
    pinch_middle_reach: bool
    palm_facing_camera: bool
    hand_size: float  # aspect-corrected, in frame-height units


@dataclass(frozen=True)
class Reading:
    """What the controller needs from one frame. Points are normalised mirrored-frame coords."""

    pose: Pose
    pointer: Point  # index fingertip: drives the cursor
    pinch_point: Point  # midway between thumb and index tips: drives drags
    palm: Point  # palm centre: entry point for scroll / volume / brightness
    handedness: Handedness = "Right"


# --------------------------------------------------------------------------- geometry
def _xy(hand: HandObservation) -> list[Point]:
    """Landmarks in frame-height units (x rescaled by the aspect ratio)."""
    return [(x * hand.aspect, y) for x, y, _ in hand.landmarks]


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mean(points: Sequence[Point]) -> Point:
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def palm_facing_camera(hand: HandObservation) -> bool:
    """True when the palm (not the back of the hand) faces the camera.

    In a mirrored image a right hand seen palm-first has its index knuckle to the left of its
    pinky knuckle, which makes (index - wrist) x (pinky - wrist) positive (image y points down).
    The back of the hand, or a left hand palm-first, flips the sign.
    """
    p = _xy(hand)
    winding = _cross(_sub(p[INDEX_MCP], p[WRIST]), _sub(p[PINKY_MCP], p[WRIST]))
    return winding > 0 if hand.handedness == "Right" else winding < 0


def thumb_side(hand: HandObservation) -> Point:
    """Unit vector across the palm pointing to the thumb side, from handedness and facing."""
    p = _xy(hand)
    up = _sub(p[MIDDLE_MCP], p[WRIST])
    norm = math.hypot(*up) or 1.0
    up = (up[0] / norm, up[1] / norm)
    right_of_up = (-up[1], up[0])  # image-right when the hand points up (y down)
    # Palm-first in a mirror: a right hand's thumb is on the image left, a left hand's on the right.
    side = (-right_of_up[0], -right_of_up[1]) if hand.handedness == "Right" else right_of_up
    if not palm_facing_camera(hand):
        side = (-side[0], -side[1])
    return side


def features(hand: HandObservation, cfg: ClassifierConfig | None = None) -> Features:
    cfg = cfg or ClassifierConfig()
    p = _xy(hand)
    size = _dist(p[WRIST], p[MIDDLE_MCP]) or 1e-6
    wrist = p[WRIST]

    fingers = [_dist(wrist, p[tip]) > _dist(wrist, p[pip]) for _mcp, pip, tip in FINGERS]
    thumb_out = _dot(_sub(p[THUMB_TIP], p[INDEX_MCP]), thumb_side(hand)) / size
    thumb = thumb_out > cfg.thumb_out

    centre = _mean([p[i] for i in PALM])
    thumb_clear = _dist(p[THUMB_TIP], centre) / size > cfg.pinch_min_reach

    def reach(tip: int) -> bool:
        return thumb_clear and _dist(p[tip], centre) / size > cfg.pinch_min_reach

    return Features(
        extended=(thumb, *fingers),  # type: ignore[arg-type]
        pinch_index=_dist(p[THUMB_TIP], p[INDEX_TIP]) / size,
        pinch_middle=_dist(p[THUMB_TIP], p[MIDDLE_TIP]) / size,
        pinch_index_reach=reach(INDEX_TIP),
        pinch_middle_reach=reach(MIDDLE_TIP),
        palm_facing_camera=palm_facing_camera(hand),
        hand_size=size,
    )


def _pattern_pose(extended: Sequence[bool]) -> Pose:
    for pattern, pose in _PATTERNS:
        if all(want is None or want == got for want, got in zip(pattern, extended, strict=True)):
            return pose
    return Pose.NONE


def pose_from_features(f: Features, prev: Pose | None, cfg: ClassifierConfig) -> Pose:
    candidates: list[tuple[float, Pose]] = []
    for pose, ratio, clear in (
        (Pose.PINCH_INDEX, f.pinch_index, f.pinch_index_reach),
        (Pose.PINCH_MIDDLE, f.pinch_middle, f.pinch_middle_reach),
    ):
        limit = cfg.pinch_off if prev == pose else cfg.pinch_on
        if clear and ratio < limit:
            candidates.append((ratio, pose))
    if candidates:
        # Keep an ongoing pinch rather than hopping to the other finger mid-pinch.
        held = [c for c in candidates if c[1] == prev]
        return (held or sorted(candidates))[0][1]
    return _pattern_pose(f.extended)


def classify(
    hand: HandObservation | None, prev: Pose | None = None, cfg: ClassifierConfig | None = None
) -> Pose:
    """Pose of one hand. ``prev`` is the previous frame's pose (pinch hysteresis only)."""
    if hand is None:
        return Pose.NONE
    cfg = cfg or ClassifierConfig()
    return pose_from_features(features(hand, cfg), prev, cfg)


def read_hand(
    hand: HandObservation | None, prev: Pose | None = None, cfg: ClassifierConfig | None = None
) -> Reading | None:
    """Pose plus the tracking points the controller uses. None when there is no hand."""
    if hand is None:
        return None
    lm = hand.landmarks
    thumb, index = lm[THUMB_TIP], lm[INDEX_TIP]
    return Reading(
        pose=classify(hand, prev, cfg),
        pointer=(index[0], index[1]),
        pinch_point=((thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2),
        palm=_mean([(lm[i][0], lm[i][1]) for i in PALM]),
        handedness=hand.handedness,
    )
