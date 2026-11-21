"""Landmark fixture files (tests/fixtures/gestures/*.json): written by scripts/record_gestures.py.

    {
      "version": 1, "pose": "pinch_index", "hand": "Right", "source": "recorded" | "synthetic",
      "aspect": 1.333, "mirrored": true, "recorded_at": "...", "camera": "...",
      "frames": [ {"handedness": "Right", "score": 0.98, "landmarks": [[x, y, z], ... 21]}, ... ]
    }

One frame per line keeps the files small and diffs readable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grace.input.gestures.classifier import HandObservation, Pose

FIXTURE_VERSION = 1


@dataclass(frozen=True)
class Fixture:
    path: Path
    pose: Pose
    hand: str
    source: str
    hands: tuple[HandObservation, ...]


def frame_dict(hand: HandObservation, digits: int = 5) -> dict[str, Any]:
    return {
        "handedness": hand.handedness,
        "score": round(hand.score, 3),
        "landmarks": [[round(c, digits) for c in p] for p in hand.landmarks],
    }


def dump_fixture(doc: Mapping[str, Any]) -> str:
    header = {k: v for k, v in doc.items() if k != "frames"}
    lines = ["{"]
    for key, value in header.items():
        lines.append(f"  {json.dumps(key)}: {json.dumps(value)},")
    frames = [json.dumps(f, separators=(",", ":")) for f in doc.get("frames", [])]
    lines.append('  "frames": [')
    lines += [f"    {f}," for f in frames[:-1]] + ([f"    {frames[-1]}"] if frames else [])
    lines += ["  ]", "}"]
    return "\n".join(lines) + "\n"


def load_fixture(path: Path) -> Fixture:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("version") != FIXTURE_VERSION:
        raise ValueError(f"{path}: unsupported fixture version {doc.get('version')!r}")
    if not doc.get("mirrored", True):
        raise ValueError(f"{path}: frames must come from the mirrored (flipped) camera image")
    aspect = float(doc.get("aspect", 4 / 3))
    hands = tuple(
        HandObservation.from_lists(f["landmarks"], f["handedness"], f.get("score", 1.0), aspect)
        for f in doc["frames"]
    )
    return Fixture(Path(path), Pose(doc["pose"]), doc["hand"], doc.get("source", "recorded"), hands)
