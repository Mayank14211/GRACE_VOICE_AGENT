"""Pick a wake-word threshold from demo-room recordings (docs/wakeword.md section 4).

    uv run python scripts/tune_wakeword.py --model models/wakeword/grace.onnx \
        --positives data/wakeword/test_positive --negatives data/wakeword/test_negative

positives: WAVs that each contain the wake word once. negatives: room audio without it (any
length; long recordings of people talking are best). Prints the false-reject rate and false
accepts per hour for thresholds 0.2-0.9 at the given patience, and a suggestion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace.input.audio import load_wav  # noqa: E402
from grace.input.wakeword import ThresholdResult, evaluate_threshold, load_wake_word  # noqa: E402
from grace.models import MODELS_DIR, resolve  # noqa: E402


def suggest(results: list[ThresholdResult], max_false_per_hour: float) -> ThresholdResult | None:
    """Among thresholds under the false-accept bar, the fewest misses; of those, the middle one
    (a margin both ways: test recordings never cover every voice or every noise)."""
    ok = [r for r in results if r.false_accepts_per_hour <= max_false_per_hour]
    if not ok:
        return None
    fewest = min(r.false_reject_rate for r in ok)
    tied = [r for r in ok if r.false_reject_rate == fewest]
    return tied[len(tied) // 2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "GRACE").splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("models/wakeword/grace.onnx"))
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument("--negatives", type=Path, required=True)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-false-per-hour", type=float, default=0.5)
    args = parser.parse_args(argv)

    det, why = load_wake_word(
        True, resolve(args.model), models_dir=MODELS_DIR, threshold=0.5, patience=args.patience
    )
    if det is None:
        print(why)
        return 1
    pos_files = sorted(args.positives.glob("*.wav"))
    neg_files = sorted(args.negatives.glob("*.wav"))
    if not pos_files or not neg_files:
        print("need *.wav files in both --positives and --negatives")
        return 1
    pos = [det.raw_scores(load_wav(p)) for p in pos_files]
    neg = [det.raw_scores(load_wav(p)) for p in neg_files]
    hours = sum(len(s) for s in neg) * 0.08 / 3600
    print(f"{det.name}: {len(pos)} wake-word clips, {hours * 60:.1f} min of room audio, "
          f"patience {args.patience}")  # fmt: skip
    print(f"{'threshold':>9}  {'missed':>7}  {'false/hour':>10}")
    results = [evaluate_threshold(pos, neg, x / 100, args.patience) for x in range(20, 95, 5)]
    for r in results:
        missed, per_hour = r.false_reject_rate, r.false_accepts_per_hour
        print(f"{r.threshold:>9.2f}  {missed:>7.1%}  {per_hour:>10.2f}")
    best = suggest(results, args.max_false_per_hour)
    if best is None:
        print(f"No threshold keeps false accepts under {args.max_false_per_hour}/hour: retrain "
              "with more negatives (docs/wakeword.md section 3).")  # fmt: skip
        return 1
    print(f"Suggested: wakeword.threshold: {best.threshold} (misses {best.false_reject_rate:.0%}, "
          f"{best.false_accepts_per_hour:.2f} false/hour)")  # fmt: skip
    if hours < 0.15:
        print("Note: under 10 minutes of room audio; false/hour is a rough estimate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
