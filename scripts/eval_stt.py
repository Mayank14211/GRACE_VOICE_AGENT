"""Evaluate speech-to-text engines on recorded commands: accuracy and latency.

    uv run python scripts/eval_stt.py                       # tests/audio/*.wav + transcripts.csv
    uv run python scripts/eval_stt.py --engines moonshine --save
    uv run python scripts/eval_stt.py --audio-dir data/synthetic_eval --label synthetic
    uv run python scripts/eval_stt.py --engines groq,moonshine --save --section groq_vs_moonshine \
        --label "team recordings, normal network"                 # M5b (cloud-first), then again
                                                                  # on a phone hotspot

Engines: groq (GROQ_API_KEY; real requests that count against the free tier in the app's own
quota store, logs/quota.json), moonshine (extra [stt-local]), whisper_local (extra [stt-whisper]).
Uploads go through the same guards as the app (speech plus 200 ms, nothing under 300 ms).

Each recording is replayed in real time through the same path as the app: Silero VAD finds the
utterance, the STT worker thread transcribes it while it "is spoken", and the clock runs from the
moment VAD decides speech has ended to the moment the final text is ready (eos -> final, the M5
acceptance number). speech end -> final adds the end_of_speech_ms silence GRACE waits for.

Accuracy: WER (lower case, punctuation dropped, "forty" == "40") and intent accuracy (the
grammar finds the same command in the transcript as in the reference text; for commands this is
what matters). ``--save`` writes the table to docs/results/stt.md.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from results_md import (  # noqa: E402
    STT_RESULTS,
    median,
    percentile,
    stamp,
    table,
    use_utf8_output,
    write_section,
)

AUDIO_DIR = ROOT / "tests" / "audio"
LEAD_S, TAIL_S = 0.3, 1.5  # silence around each recording so VAD can open and close it
BAR_MS = 500.0

# --------------------------------------------------------------------------- text (pure)
_UNITS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS = {
    w: 10 * i
    for i, w in enumerate("- - twenty thirty forty fifty sixty seventy eighty ninety".split())
    if w != "-"
}


def _numbers_to_digits(words: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in _TENS:
            value = _TENS[w]
            if i + 1 < len(words) and words[i + 1] in _UNITS and _UNITS[words[i + 1]] < 10:
                value += _UNITS[words[i + 1]]
                i += 1
            out.append(str(value))
        elif w in _UNITS:
            out.append(str(_UNITS[w]))
        elif w == "hundred" and out and out[-1].isdigit():
            out[-1] = str(int(out[-1]) * 100)
        else:
            out.append(w)
        i += 1
    return out


def normalize(text: str) -> list[str]:
    """Words for WER: lower case, no punctuation, number words as digits, % as 'percent'."""
    text = text.lower().replace("%", " percent ").replace("-", " ")
    text = re.sub(r"[^\w\s']", " ", text)
    return _numbers_to_digits(text.split())


def word_errors(ref: list[str], hyp: list[str]) -> int:
    """Word-level edit distance (substitutions + insertions + deletions)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def intent_key(grammar: Any, text: str) -> tuple[str, tuple] | None:
    intent = grammar.parse(text)
    if intent is None:
        return None
    return intent.name, tuple(sorted((k, str(v)) for k, v in intent.args.items()))


# --------------------------------------------------------------------------- data
@dataclass(frozen=True)
class Sample:
    file: Path
    text: str
    speaker: str


def load_samples(audio_dir: Path) -> list[Sample]:
    csv_path = audio_dir / "transcripts.csv"
    if not csv_path.is_file():
        return []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [
        Sample(audio_dir / r["file"], r["text"], r.get("speaker", ""))
        for r in rows
        if r.get("file") and (audio_dir / r["file"]).is_file()
    ]


@dataclass
class Outcome:
    text: str = ""
    eos_ms: list[float] = field(default_factory=list)  # one per utterance
    speech_end_ms: list[float] = field(default_factory=list)
    fallback: str | None = None
    error: str | None = None


@dataclass
class EngineReport:
    engine: str
    samples: int
    wer: float
    intent_acc: float
    exact: float
    eos_median: float
    eos_p90: float
    speech_end_median: float
    errors: int
    worst: list[tuple[str, str, str]]  # (file, reference, heard)


def score(
    engine: str, samples: list[Sample], outcomes: list[Outcome], grammar: Any
) -> EngineReport:
    edits = ref_words = intents = exact = errors = 0
    eos: list[float] = []
    spe: list[float] = []
    ranked: list[tuple[float, tuple[str, str, str]]] = []
    for sample, out in zip(samples, outcomes, strict=True):
        ref, hyp = normalize(sample.text), normalize(out.text)
        e = word_errors(ref, hyp)
        edits += e
        ref_words += len(ref)
        exact += ref == hyp
        intents += intent_key(grammar, sample.text) == intent_key(grammar, out.text)
        errors += out.error is not None
        eos += out.eos_ms
        spe += out.speech_end_ms
        ranked.append((e / max(1, len(ref)), (sample.file.name, sample.text, out.text)))
    ranked.sort(key=lambda r: -r[0])
    n = max(1, len(samples))
    return EngineReport(
        engine=engine,
        samples=len(samples),
        wer=edits / max(1, ref_words),
        intent_acc=intents / n,
        exact=exact / n,
        eos_median=median(eos),
        eos_p90=percentile(eos, 90),
        speech_end_median=median(spe),
        errors=errors,
        worst=[r for s, r in ranked[:5] if s > 0],
    )


# --------------------------------------------------------------------------- running
def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def replay(
    engine: Any, audio: np.ndarray, settings: Any, vad_path: Path, realtime: bool
) -> Outcome:
    """One recording through VAD + the STT worker thread, paced like a live microphone."""
    from grace.input.audio import FRAME_S, Segmenter, SileroVad, frames
    from grace.input.voice import ListenPipeline, SttWorker

    out = Outcome()
    texts: list[str] = []
    lock = threading.Lock()

    def on_final(utt: Any, final: Any) -> None:
        with lock:
            texts.append(final.text)
            ms = (final.t_final - utt.eos_t) * 1000
            out.eos_ms.append(ms)
            out.speech_end_ms.append(ms + utt.hangover_s * 1000)
            out.fallback = final.fallback_from or out.fallback

    def on_problem(why: str) -> None:
        out.error = why

    pipe_ref: list[ListenPipeline] = []
    stt = SttWorker(lambda _dictation: engine, lambda: pipe_ref[0], on_final=on_final,
                    on_unavailable=on_problem)  # fmt: skip
    a = settings.audio
    pipe = ListenPipeline(
        vad=SileroVad(vad_path),
        segmenter=Segmenter(
            threshold=a.vad_threshold,
            end_of_speech_ms=a.end_of_speech_ms,
            max_s=a.max_utterance_s,
            pre_roll_ms=a.pre_roll_ms,
            min_speech_ms=a.min_speech_ms,
        ),
        submit=stt.submit,
        wake_mode="off",
    )
    pipe_ref.append(pipe)
    stt.start()
    padded = np.concatenate([silence(LEAD_S), audio, silence(TAIL_S)])
    due = time.perf_counter()
    for frame in frames(padded):
        if realtime:
            due += FRAME_S
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        pipe.feed(frame)
    deadline = time.perf_counter() + 30
    while pipe._pending and time.perf_counter() < deadline:
        time.sleep(0.01)
    stt.stop()
    out.text = " ".join(t for t in texts if t)
    return out


def make_engine(name: str, settings: Any, quota: Any = None) -> Any:
    """``quota``: the app's governor, so Groq requests made by the eval are counted (and refused
    when the free tier is spent) exactly like the app's own."""
    from grace.input.stt import engine_factories

    engine = engine_factories(settings, quota=quota)[name]()
    engine.load()
    return engine


def available_engines(settings: Any) -> list[str]:
    from grace.input.stt import moonshine_installed, whisper_installed

    names = []
    if settings.api_key("groq"):
        names.append("groq")  # the default engine since S06b
    if moonshine_installed():
        names.append("moonshine")
    if whisper_installed():
        names.append("whisper_local")
    return names


def markdown(reports: list[EngineReport], samples: list[Sample], label: str, settings: Any) -> str:
    speakers = sorted({s.speaker for s in samples if s.speaker})
    rows = [
        [
            r.engine,
            str(r.samples),
            f"{r.wer:.1%}",
            f"{r.intent_acc:.1%}",
            f"{r.exact:.1%}",
            f"{r.eos_median:.0f}",
            f"{r.eos_p90:.0f}",
            f"{r.speech_end_median:.0f}",
            str(r.errors),
        ]
        for r in reports
    ]
    body = [
        f"Run {stamp()}. Audio: {label}, {len(samples)} recordings"
        + (f", speakers: {', '.join(speakers)}" if speakers else "")
        + f". end_of_speech_ms {settings.audio.end_of_speech_ms}, Moonshine "
        f"{settings.stt.moonshine_model}, faster-whisper {settings.stt.whisper_model} at "
        f"{settings.inference.threads} threads.",
        "",
        table(
            [
                "engine",
                "files",
                "WER",
                "intent acc.",
                "exact",
                "eos→final median ms",
                "eos→final p90 ms",
                "speech end→final median ms",
                "errors",
            ],
            rows,
        ),  # fmt: skip
    ]
    moon = next((r for r in reports if r.engine == "moonshine"), None)
    if moon is not None:
        verdict = "MET" if moon.eos_median < BAR_MS else "NOT MET"
        body += [
            "",
            f"**Acceptance (M5):** Moonshine median end-of-speech → final text "
            f"{moon.eos_median:.0f} ms vs bar {BAR_MS:.0f} ms: **{verdict}** ({label}).",
        ]
    groq = next((r for r in reports if r.engine == "groq"), None)
    if groq is not None:
        line = (
            f"**Cloud-first (M5b):** Groq end-of-speech → final text median "
            f"{groq.eos_median:.0f} ms, p90 {groq.eos_p90:.0f} ms ({label})"
        )
        if moon is not None:
            line += (
                f"; Moonshine on the same recordings median {moon.eos_median:.0f} ms, "
                f"p90 {moon.eos_p90:.0f} ms"
            )
        body += ["", line + ". The team sets the pass bar from these numbers."]
    for r in reports:
        if r.worst:
            body += ["", f"Worst {r.engine} transcripts:", ""]
            body += [f'- `{f}`: "{ref}" → "{hyp}"' for f, ref, hyp in r.worst]
    return "\n".join(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "GRACE").splitlines()[0])
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--engines", default=None, help="comma list (default: all available)")
    parser.add_argument("--fast", action="store_true", help="no real-time pacing (latency invalid)")
    parser.add_argument("--save", action="store_true", help="write docs/results/stt.md")
    parser.add_argument("--label", default=None, help="name of the audio set in the results")
    parser.add_argument("--section", default="eval_stt", help="results section name")
    args = parser.parse_args(argv)
    use_utf8_output()

    from grace.config import load_settings
    from grace.input.stt.base import SttError
    from grace.intent.grammar import default_grammar
    from grace.models import MODELS_DIR, SILERO_VAD

    settings = load_settings()
    samples = load_samples(args.audio_dir)
    if not samples:
        print(f"No recordings in {args.audio_dir} (see tests/audio/README.md).")
        return 1
    vad_path = MODELS_DIR / SILERO_VAD
    if not vad_path.is_file():
        print("Silero VAD missing: run scripts/fetch_models.py")
        return 1
    names = args.engines.split(",") if args.engines else available_engines(settings)
    if not names:
        print("No engine: set GROQ_API_KEY in .env or uv sync --extra stt-local.")
        return 1
    quota = None
    if "groq" in names:
        from grace.core.quota import build_governor

        quota = build_governor(settings)  # the real counters: eval calls spend real free quota
    grammar = default_grammar()
    from grace.input.audio import load_wav

    audio = [load_wav(s.file) for s in samples]
    total_s = sum(len(a) for a in audio) / 16000
    print(f"{len(samples)} recordings ({total_s:.0f} s), engines: {', '.join(names)}"
          + ("" if not args.fast else " (fast: latency not meaningful)"))  # fmt: skip
    reports = []
    for name in names:
        try:
            engine = make_engine(name, settings, quota)
        except (SttError, KeyError) as exc:
            print(f"  {name}: skipped ({exc})")
            continue
        outcomes = []
        for i, wav in enumerate(audio, 1):
            outcomes.append(replay(engine, wav, settings, vad_path, realtime=not args.fast))
            print(f"\r  {name}: {i}/{len(samples)}", end="", flush=True)
        print()
        reports.append(score(name, samples, outcomes, grammar))
    if not reports:
        return 1
    audio_dir = args.audio_dir.resolve()
    label = args.label or (
        audio_dir.relative_to(ROOT).as_posix() if audio_dir.is_relative_to(ROOT) else str(audio_dir)
    )
    text = markdown(reports, samples, label, settings)
    print()
    print(text)
    if args.save:
        write_section(STT_RESULTS, args.section, f"### {args.section} ({label})\n\n{text}")
        print(f"\nSaved to {STT_RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
