"""Synthetic MediaPipe-style hand landmarks for tests (no camera needed).

A 2-D skeleton of a right hand seen palm-first in a mirrored frame, in "hand units" (wrist at the
origin, middle knuckle at (0, 1), y up). Poses set each finger extended or curled; pinches move the
thumb tip onto a fingertip. ``make_hand`` rotates, scales, mirrors (left hand), converts to
normalised image coordinates (y down, x squeezed by the frame aspect) and adds jitter.

These are stand-ins until the team records real ones with ``scripts/record_gestures.py``; the
fixture files they produce are marked ``"source": "synthetic"``.

    uv run python tests/gesture_synth.py      # (re)write tests/fixtures/gestures/synthetic_*.json
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

if __name__ == "__main__":  # run as a script: make `grace` importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grace.input.gestures.classifier import HandObservation, Pose
from grace.input.gestures.fixtures import FIXTURE_VERSION, dump_fixture, frame_dict

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gestures"

Vec = tuple[float, float]

MCP = {"index": (-0.28, 0.95), "middle": (0.0, 1.0), "ring": (0.24, 0.94), "pinky": (0.44, 0.84)}
SEGMENTS = {
    "index": (0.40, 0.24, 0.20),
    "middle": (0.45, 0.28, 0.21),
    "ring": (0.42, 0.26, 0.20),
    "pinky": (0.32, 0.20, 0.18),
}
SPREAD_DEG = {"index": -8.0, "middle": 0.0, "ring": 7.0, "pinky": 16.0}
ORDER = ("index", "middle", "ring", "pinky")
WRIST: Vec = (0.0, 0.0)
THUMB_CMC: Vec = (-0.22, 0.22)
THUMB_MCP: Vec = (-0.42, 0.42)


def _dir(deg: float) -> Vec:
    r = math.radians(deg)
    return (math.sin(r), math.cos(r))


def _add(a: Vec, b: Vec, k: float = 1.0) -> Vec:
    return (a[0] + k * b[0], a[1] + k * b[1])


def _finger(name: str, extended: bool) -> list[Vec]:
    """PIP, DIP, TIP for one finger."""
    base, d = MCP[name], _dir(SPREAD_DEG[name])
    if extended:
        l1, l2, l3 = SEGMENTS[name]
        pip = _add(base, d, l1)
        dip = _add(pip, d, l2)
        return [pip, dip, _add(dip, d, l3)]
    # Curled toward the camera: in projection the tip folds back below the knuckle.
    return [_add(base, d, 0.20), _add(base, d, -0.02), _add(base, d, -0.18)]


def _thumb(state: str) -> list[Vec]:
    """IP, TIP for the thumb: 'out' (extended sideways), 'folded' (across the palm), 'fist'."""
    if state == "out":
        d = _dir(-50.0)
        ip = _add(THUMB_MCP, d, 0.32)
        return [ip, _add(ip, d, 0.28)]
    if state == "fist":
        return [(-0.28, 0.66), (-0.02, 0.80)]
    return [(-0.32, 0.56), (-0.08, 0.68)]


def _assemble(thumb: list[Vec], fingers: dict[str, list[Vec]]) -> list[Vec]:
    pts: list[Vec] = [WRIST, THUMB_CMC, THUMB_MCP, *thumb]
    for name in ORDER:
        pts += [MCP[name], *fingers[name]]
    return pts


# (thumb state, index, middle, ring, pinky)
_SHAPES: dict[Pose, list[tuple[str, bool, bool, bool, bool]]] = {
    Pose.OPEN_PALM: [("out", True, True, True, True), ("folded", True, True, True, True)],
    Pose.FIST: [("fist", False, False, False, False)],
    Pose.INDEX_UP: [("folded", True, False, False, False), ("out", True, False, False, False)],
    Pose.TWO_UP: [("folded", True, True, False, False)],
    Pose.THREE_UP: [("folded", True, True, True, False)],
    Pose.THUMB_INDEX_MIDDLE: [("out", True, True, False, False)],
}


def _pinch_index(variant: int) -> list[Vec]:
    others = variant == 0  # 0 = "OK sign" (other fingers up), 1 = others curled
    index = [(-0.40, 1.30), (-0.52, 1.20), (-0.60, 1.05)]
    fingers = {
        "index": index,
        "middle": _finger("middle", others),
        "ring": _finger("ring", others),
        "pinky": _finger("pinky", others),
    }
    return _assemble([(-0.58, 0.72), (-0.62, 1.00)], fingers)


def _pinch_middle(variant: int) -> list[Vec]:
    others = variant == 0
    middle = [(0.0, 1.35), (-0.08, 1.30), (-0.18, 1.12)]
    fingers = {
        "index": _finger("index", True),
        "middle": middle,
        "ring": _finger("ring", others),
        "pinky": _finger("pinky", others),
    }
    return _assemble([(-0.45, 0.78), (-0.20, 1.08)], fingers)


def hand_points(pose: Pose, variant: int = 0) -> list[Vec]:
    """21 points in hand units (right hand, palm to camera, mirrored view, y up)."""
    if pose is Pose.PINCH_INDEX:
        return _pinch_index(variant % 2)
    if pose is Pose.PINCH_MIDDLE:
        return _pinch_middle(variant % 2)
    shapes = _SHAPES[pose]
    thumb, *ext = shapes[variant % len(shapes)]
    return _assemble(_thumb(thumb), {n: _finger(n, e) for n, e in zip(ORDER, ext, strict=True)})


def variants(pose: Pose) -> int:
    return 2 if pose in (Pose.PINCH_INDEX, Pose.PINCH_MIDDLE) else len(_SHAPES[pose])


def make_hand(
    pose: Pose,
    handedness: str = "Right",
    *,
    center: Vec = (0.5, 0.55),
    size: float = 0.18,
    rotation_deg: float = 0.0,
    variant: int = 0,
    jitter: float = 0.0,
    aspect: float = 4 / 3,
    rng: random.Random | None = None,
) -> HandObservation:
    """A hand whose wrist->middle-knuckle length is ``size`` of the frame height.

    ``center`` is where the palm centre lands (normalised image coords). Rotation is
    counter-clockwise as the user sees it. A left hand is the mirror image of a right hand.
    """
    rng = rng or random.Random(0)
    pts = hand_points(pose, variant)
    if handedness == "Left":
        pts = [(-x, y) for x, y in pts]
    r = math.radians(rotation_deg)
    cos, sin = math.cos(r), math.sin(r)
    rotated = [(x * cos - y * sin, x * sin + y * cos) for x, y in pts]
    palm = [rotated[i] for i in (0, 5, 9, 13, 17)]
    cx = sum(p[0] for p in palm) / 5
    cy = sum(p[1] for p in palm) / 5
    out = []
    for x, y in rotated:
        # hand units -> frame-height units (y flipped) -> normalised (x divided by aspect)
        u = center[0] + (x - cx) * size / aspect
        v = center[1] - (y - cy) * size
        out.append((u + rng.gauss(0, jitter), v + rng.gauss(0, jitter), 0.0))
    return HandObservation(tuple(out), handedness, 0.99, aspect)  # type: ignore[arg-type]


def synthetic_frames(pose: Pose, handedness: str, n: int = 20, seed: int = 0) -> list[dict]:
    """``n`` frames of one pose with varied position, size, rotation, variant and jitter."""
    rng = random.Random(f"{pose.value}-{handedness}-{seed}")
    frames = []
    for i in range(n):
        hand = make_hand(
            pose,
            handedness,
            center=(rng.uniform(0.35, 0.65), rng.uniform(0.4, 0.65)),
            size=rng.uniform(0.14, 0.24),
            rotation_deg=rng.uniform(-20, 20),
            variant=i % variants(pose),
            jitter=0.002,
            rng=rng,
        )
        frames.append(frame_dict(hand))
    return frames


def write_fixtures(directory: Path = FIXTURE_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for pose in Pose:
        if pose is Pose.NONE:
            continue
        for handedness in ("Right", "Left"):
            path = directory / f"synthetic_{pose.value}_{handedness.lower()}.json"
            doc = {
                "version": FIXTURE_VERSION,
                "pose": pose.value,
                "hand": handedness,
                "source": "synthetic",
                "aspect": 4 / 3,
                "mirrored": True,
                "frames": synthetic_frames(pose, handedness),
            }
            path.write_text(dump_fixture(doc), encoding="utf-8")
            written.append(path)
    return written


if __name__ == "__main__":
    for p in write_fixtures():
        print(p.relative_to(Path.cwd()) if p.is_relative_to(Path.cwd()) else p)
