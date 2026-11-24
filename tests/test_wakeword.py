"""grace/input/wakeword.py: patience gate, loading fallbacks, the real openWakeWord path."""

from __future__ import annotations

from pathlib import Path

import pytest

from grace.input.audio import frames, load_wav
from grace.input.wakeword import PatienceGate, load_wake_word
from grace.models import MODELS_DIR, OWW_DEMO, OWW_EMBEDDING, OWW_MELSPEC

FIX = Path(__file__).resolve().parent / "fixtures" / "audio"


def test_patience_needs_consecutive_scores() -> None:
    gate = PatienceGate(threshold=0.5, patience=2)
    assert [gate.push(s) for s in (0.9, 0.1, 0.9, 0.9)] == [False, False, False, True]


def test_one_word_gives_one_detection() -> None:
    gate = PatienceGate(threshold=0.5, patience=1, refractory=3)
    assert [gate.push(0.9) for _ in range(6)] == [True, False, False, False, True, False]


def test_patience_one_fires_on_the_first_high_score() -> None:
    assert PatienceGate(threshold=0.5, patience=1).push(0.5)


def test_load_reports_why_there_is_no_detector(tmp_path: Path) -> None:
    kw = {"models_dir": tmp_path, "threshold": 0.5, "patience": 2}
    det, why = load_wake_word(False, tmp_path / "grace.onnx", **kw)
    assert det is None and "turned off" in why
    det, why = load_wake_word(True, tmp_path / "grace.onnx", **kw)
    assert det is None and "no wake-word model" in why
    (tmp_path / "grace.onnx").write_bytes(b"x")
    det, why = load_wake_word(True, tmp_path / "grace.onnx", **kw)
    assert det is None and "fetch_models" in why


def test_a_broken_model_does_not_raise(tmp_path: Path) -> None:
    for rel in (OWW_MELSPEC, OWW_EMBEDDING):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"not onnx")
    (tmp_path / "grace.onnx").write_bytes(b"not onnx")
    det, why = load_wake_word(
        True, tmp_path / "grace.onnx", models_dir=tmp_path, threshold=0.5, patience=2
    )
    assert det is None and "failed to load" in why


needs_demo_model = pytest.mark.skipif(
    not all((MODELS_DIR / p).is_file() for p in (OWW_DEMO, OWW_MELSPEC, OWW_EMBEDDING)),
    reason="run scripts/fetch_models.py --demo-wakeword",
)


@needs_demo_model
def test_hey_jarvis_model_detects_once() -> None:
    det, why = load_wake_word(
        True, MODELS_DIR / OWW_DEMO, models_dir=MODELS_DIR, threshold=0.5, patience=2
    )
    assert det is not None, why
    hits = [i for i, f in enumerate(frames(load_wav(FIX / "hey_jarvis.wav"))) if det.process(f)]
    assert len(hits) == 1
    assert 0.8 <= hits[0] * 0.032 <= 2.2  # just after "hey jarvis" (0.8 s - 1.6 s)


@needs_demo_model
def test_hey_jarvis_model_ignores_other_speech() -> None:
    det, _ = load_wake_word(
        True, MODELS_DIR / OWW_DEMO, models_dir=MODELS_DIR, threshold=0.5, patience=2
    )
    assert det is not None
    for name in ("one_command.wav", "two_commands.wav", "silence.wav"):
        assert not any(det.process(f) for f in frames(load_wav(FIX / name))), name


def test_evaluate_threshold_counts_misses_and_false_wakes() -> None:
    from grace.input.wakeword import evaluate_threshold

    positives = [[0.1, 0.9, 0.9], [0.1, 0.6, 0.2]]  # the second never holds 2 frames
    hour = [0.0] * 45000  # one hour of 80 ms chunks
    negatives = [[0.9, 0.9] + hour[2:]]
    r = evaluate_threshold(positives, negatives, threshold=0.5, patience=2)
    assert r.false_reject_rate == 0.5 and r.false_accepts_per_hour == pytest.approx(1.0)
    r = evaluate_threshold(positives, negatives, threshold=0.95, patience=2)
    assert r.false_reject_rate == 1.0 and r.false_accepts_per_hour == 0.0


@needs_demo_model
def test_raw_scores_peak_on_the_wake_word() -> None:
    det, _ = load_wake_word(
        True, MODELS_DIR / OWW_DEMO, models_dir=MODELS_DIR, threshold=0.5, patience=2
    )
    assert det is not None
    scores = det.raw_scores(load_wav(FIX / "hey_jarvis.wav"))
    assert max(scores) > 0.9 and len(scores) == int(4.18 / 0.08)


def test_tune_suggests_the_middle_of_the_best_range() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from tune_wakeword import suggest

    from grace.input.wakeword import ThresholdResult as R

    results = [R(0.3, 0, 2.0), R(0.4, 0, 0.4), R(0.5, 0, 0), R(0.6, 0, 0), R(0.7, 0.1, 0)]
    assert suggest(results, 0.5).threshold == 0.5  # 0.3 wakes too often; 0.7 misses more
    assert suggest([R(0.3, 0.0, 2.0)], 0.5) is None
