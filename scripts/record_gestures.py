"""Record hand-landmark fixtures: 20 frames per pose -> tests/fixtures/gestures/*.json.

    uv run python scripts/record_gestures.py --hand right            # every pose, right hand
    uv run python scripts/record_gestures.py --hand left --pose fist --pose pinch_index
    uv run python scripts/record_gestures.py --hand right --name mg  # tag files with initials

For each pose: read the instruction, press Enter, hold the pose during the countdown and keep it
until "saved". No preview window: the console shows what the classifier sees, so a disagreement is
visible straight away (re-record that pose, or keep it as a hard case and tune the thresholds).
The tests classify every file in tests/fixtures/gestures, so new recordings become test cases.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace.input.gestures.classifier import HandObservation, Pose, classify  # noqa: E402
from grace.input.gestures.fixtures import FIXTURE_VERSION, dump_fixture, frame_dict  # noqa: E402
from grace.input.gestures.tracker import HandTracker  # noqa: E402

DEFAULT_OUT = ROOT / "tests" / "fixtures" / "gestures"

INSTRUCTIONS: dict[Pose, str] = {
    Pose.OPEN_PALM: "open hand, all five fingers spread, palm to the camera",
    Pose.FIST: "closed fist, thumb folded over the fingers, palm to the camera",
    Pose.INDEX_UP: "point up with the index finger only, other fingers curled",
    Pose.TWO_UP: "index + middle up (a 'V' held together or apart), thumb folded",
    Pose.THREE_UP: "index + middle + ring up, thumb and pinky folded",
    Pose.THUMB_INDEX_MIDDLE: "thumb out sideways, index + middle up, ring + pinky curled",
    Pose.PINCH_INDEX: "touch thumb tip to index tip and hold",
    Pose.PINCH_MIDDLE: "touch thumb tip to middle fingertip and hold (index stays up)",
}
POSES = tuple(INSTRUCTIONS)


def collect(
    tracker: HandTracker,
    hand: str,
    n: int,
    *,
    timeout_s: float = 20.0,
    clock: Callable[[], float] = time.monotonic,
    on_frame: Callable[[HandObservation | None], None] | None = None,
) -> tuple[list[HandObservation], int]:
    """Read until ``n`` frames show the requested hand. Returns (hands, other-hand frame count)."""
    got: list[HandObservation] = []
    wrong = 0
    deadline = clock() + timeout_s
    while len(got) < n and clock() < deadline:
        frame = tracker.read()
        if frame is None:
            continue
        if on_frame is not None:
            on_frame(frame.hand)
        if frame.hand is None:
            continue
        if frame.hand.handedness != hand:
            wrong += 1
            continue
        got.append(frame.hand)
    return got, wrong


def fixture_doc(pose: Pose, hand: str, hands: Sequence[HandObservation], camera: str) -> dict:
    return {
        "version": FIXTURE_VERSION,
        "pose": pose.value,
        "hand": hand,
        "source": "recorded",
        "aspect": round(hands[0].aspect, 4) if hands else 4 / 3,
        "mirrored": True,
        "recorded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "camera": camera,
        "frames": [frame_dict(h) for h in hands],
    }


def output_path(out: Path, pose: Pose, hand: str, name: str | None, force: bool) -> Path:
    stem = f"recorded_{pose.value}_{hand.lower()}" + (f"_{name}" if name else "")
    path = out / f"{stem}.json"
    i = 2
    while path.exists() and not force:
        path = out / f"{stem}_{i}.json"
        i += 1
    return path


def agreement(pose: Pose, hands: Sequence[HandObservation]) -> tuple[float, Counter[str]]:
    seen: Counter[str] = Counter()
    prev: Pose | None = None
    for h in hands:
        prev = classify(h, prev)
        seen[prev.value] += 1
    return (seen[pose.value] / len(hands) if hands else 0.0), seen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--hand", required=True, choices=["right", "left"])
    parser.add_argument("--pose", action="append", choices=[p.value for p in POSES])
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--camera", type=int, default=None, help="default: settings.yaml")
    parser.add_argument("--name", help="suffix for the file names, e.g. your initials")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args(argv)

    from grace.config import load_settings
    from grace.input.gestures import open_mediapipe_tracker

    gs = load_settings().gestures
    if args.camera is not None:
        gs = gs.model_copy(update={"camera_index": args.camera})
    hand = args.hand.capitalize()
    poses = [Pose(p) for p in args.pose] if args.pose else list(POSES)
    args.out.mkdir(parents=True, exist_ok=True)

    tracker = open_mediapipe_tracker(gs)
    try:
        for pose in poses:
            print(f"\n== {pose.value} ({hand} hand): {INSTRUCTIONS[pose]}")
            input("   Press Enter, then hold the pose... ")
            end = time.monotonic() + args.countdown
            while time.monotonic() < end:  # keep the camera stream flowing during the countdown
                tracker.read()
            print("   recording", end="", flush=True)

            def dot(h: HandObservation | None) -> None:
                print("." if h is not None else "_", end="", flush=True)

            hands, wrong = collect(tracker, hand, args.frames, on_frame=dot)
            print()
            if wrong:
                print(f"   ignored {wrong} frames labelled as the other hand")
            if len(hands) < args.frames:
                print(f"   only {len(hands)}/{args.frames} frames with a {hand} hand; skipped")
                continue
            share, seen = agreement(pose, hands)
            path = output_path(args.out, pose, hand, args.name, args.force)
            doc = fixture_doc(pose, hand, hands, f"camera {gs.camera_index}")
            path.write_text(dump_fixture(doc), encoding="utf-8")
            print(f"   saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
            note = "" if share == 1 else f"  <-- classifier saw {dict(seen)}"
            print(f"   classifier agreement: {share:.0%}{note}")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        tracker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
