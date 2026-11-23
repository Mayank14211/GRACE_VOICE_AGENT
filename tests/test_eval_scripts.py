"""scripts/eval_stt.py, bench_threads.py, results_md.py: the pure parts, plus the M5 acceptance
run on the team's recordings (marked hardware: run it manually on the base machine)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import bench_threads  # noqa: E402
import eval_stt  # noqa: E402
import results_md  # noqa: E402

from grace.config import Settings  # noqa: E402
from grace.intent.grammar import default_grammar  # noqa: E402


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("Set the volume to forty percent.", ["set", "the", "volume", "to", "40", "percent"]),
        ("Volume 70%", ["volume", "70", "percent"]),
        ("seventy-five", ["75"]),
        ("scroll down five", ["scroll", "down", "5"]),
        ("one hundred", ["100"]),
        ("What's up?", ["what's", "up"]),
    ],
)
def test_normalize(text: str, words: list[str]) -> None:
    assert eval_stt.normalize(text) == words


def test_word_errors() -> None:
    assert eval_stt.word_errors(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert eval_stt.word_errors(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert eval_stt.word_errors(["a", "b", "c"], ["a", "c"]) == 1
    assert eval_stt.word_errors(["a"], ["a", "b", "c"]) == 2
    assert eval_stt.word_errors([], ["a"]) == 1


def test_intent_key_ignores_case_and_punctuation() -> None:
    g = default_grammar()
    assert eval_stt.intent_key(g, "Open Notepad.") == eval_stt.intent_key(g, "open notepad")
    assert eval_stt.intent_key(g, "Volume 40.") != eval_stt.intent_key(g, "volume 30")
    assert eval_stt.intent_key(g, "what a nice day") is None


def test_score_counts_wer_intents_and_latency() -> None:
    g = default_grammar()
    samples = [
        eval_stt.Sample(Path("a.wav"), "open notepad", "x"),
        eval_stt.Sample(Path("b.wav"), "volume 40", "x"),
    ]
    outcomes = [
        eval_stt.Outcome("Open Notepad.", eos_ms=[100.0], speech_end_ms=[600.0]),
        eval_stt.Outcome("volume 14", eos_ms=[300.0], speech_end_ms=[800.0]),
    ]
    r = eval_stt.score("moonshine", samples, outcomes, g)
    assert r.wer == pytest.approx(1 / 4) and r.intent_acc == 0.5 and r.exact == 0.5
    assert r.eos_median == 200.0 and r.speech_end_median == 700.0
    assert r.worst == [("b.wav", "volume 40", "volume 14")]
    text = eval_stt.markdown([r], samples, "unit test", Settings())
    assert "**MET**" in text and "| moonshine | 2 | 25.0% |" in text


def test_load_samples_skips_missing_files(tmp_path: Path) -> None:
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "transcripts.csv").write_text(
        "file,text,speaker\na.wav,open notepad,mayank\ngone.wav,volume 40,mayank\n"
    )
    (only,) = eval_stt.load_samples(tmp_path)
    assert (only.file.name, only.text, only.speaker) == ("a.wav", "open notepad", "mayank")
    assert eval_stt.load_samples(tmp_path / "nothing") == []


def test_percentile_and_median() -> None:
    values = [float(v) for v in range(1, 11)]
    assert results_md.percentile(values, 90) == 9.0
    assert results_md.percentile(values, 50) == 5.0
    assert results_md.percentile([], 90) == 0.0 and results_md.median([]) == 0.0


def test_replace_section_inserts_then_replaces() -> None:
    text = results_md.replace_section("# Results\n", "eval", "first")
    assert "<!-- eval:start -->\nfirst\n<!-- eval:end -->" in text
    text = results_md.replace_section(text + "\nafter\n", "eval", "second")
    assert "second" in text and "first" not in text and text.rstrip().endswith("after")


def test_recommend_prefers_fewer_threads_when_close() -> None:
    assert bench_threads.recommend({4: 410.0, 6: 400.0, 8: 395.0}) == 4
    assert bench_threads.recommend({4: 600.0, 6: 400.0, 8: 390.0}) == 6


def test_bench_refuses_logical_thread_counts() -> None:
    with pytest.raises(SystemExit):
        bench_threads.main(["--threads", "4,12"])


# --------------------------------------------------------------------------- acceptance (manual)
@pytest.mark.hardware
@pytest.mark.skipif(not eval_stt.load_samples(eval_stt.AUDIO_DIR), reason="no team recordings yet")
def test_m5_acceptance_moonshine_median_under_500_ms(capsys: pytest.CaptureFixture[str]) -> None:
    """Real recordings, real time, base machine: median end of speech -> final text < 500 ms."""
    assert eval_stt.main(["--engines", "moonshine"]) == 0
    out = capsys.readouterr().out
    assert "**MET**" in out, out
