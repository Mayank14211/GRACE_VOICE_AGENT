"""Record 16 kHz mono WAV clips from the microphone (wake-word data, STT test commands).

    # STT test set (scripts/eval_stt.py): one clip per phrase, transcripts.csv updated.
    uv run python scripts/record_audio.py --phrases tests/audio/phrases.txt --prefix mayank

    # Wake-word positives: press Enter, say the word; the clip ends when you stop talking.
    uv run python scripts/record_audio.py --out data/wakeword/positive --prefix mayank --count 20

    # Room noise / negatives: fixed-length clips.
    uv run python scripts/record_audio.py --out data/wakeword/negative --prefix room \
        --seconds 60 --count 10

Speech clips end on Silero VAD's end of speech (like GRACE itself) and keep 0.3 s before and
after. Voices are personal data: data/ is git-ignored; tests/audio/ is committed only with every
speaker's consent (tests/audio/README.md).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace.input.audio import FRAME_S, SAMPLE_RATE, MicSource, Segmenter, SileroVad  # noqa: E402
from grace.models import MODELS_DIR, SILERO_VAD  # noqa: E402

TAIL_S = 0.3


def record_fixed(mic: MicSource, seconds: float) -> np.ndarray:
    out: list[np.ndarray] = []
    while len(out) * FRAME_S < seconds:
        frame = mic.read(1.0)
        if frame is not None:
            out.append(frame)
    return np.concatenate(out)


def record_utterance(mic: MicSource, vad: SileroVad, timeout_s: float = 8.0) -> np.ndarray | None:
    """From the start of speech (plus pre-roll) to end of speech (plus a short tail)."""
    seg = Segmenter(max_s=15.0, min_speech_ms=150)
    vad.reset()
    out: list[np.ndarray] = []
    started = time.monotonic()
    while True:
        frame = mic.read(1.0)
        if frame is None:
            continue
        step = seg.push(frame, vad.prob(frame))
        out.extend(step.audio)  # empty unless an utterance is open
        if step.end is not None:
            if step.end.too_short:
                out.clear()
                continue
            break
        if not out and time.monotonic() - started > timeout_s:
            return None
    return np.concatenate([*out, record_fixed(mic, TAIL_S)])


def save(path: Path, audio: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1, 1), SAMPLE_RATE, subtype="PCM_16")


def next_index(folder: Path, prefix: str) -> int:
    used = [int(p.stem.rsplit("_", 1)[1]) for p in folder.glob(f"{prefix}_*.wav")
            if p.stem.rsplit("_", 1)[1].isdigit()]  # fmt: skip
    return max(used, default=0) + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "audio")
    parser.add_argument("--prefix", required=True, help="speaker or room name, e.g. mayank")
    parser.add_argument("--phrases", type=Path, help="one phrase per line: record each once")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=0, help="fixed length (noise clips)")
    parser.add_argument("--device", default=None, help="sounddevice input index or name")
    args = parser.parse_args(argv)

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    vad = SileroVad(MODELS_DIR / SILERO_VAD)
    mic = MicSource(device)
    phrases = (
        [line.strip() for line in args.phrases.read_text(encoding="utf-8").splitlines()]
        if args.phrases
        else []
    )
    phrases = [p for p in phrases if p and not p.startswith("#")]
    jobs = phrases or [""] * args.count
    index = next_index(args.out, args.prefix)
    rows: list[tuple[str, str]] = []
    try:
        for n, phrase in enumerate(jobs, 1):
            prompt = f'Say: "{phrase}"' if phrase else ("Record" if args.seconds else "Speak")
            input(f"[{n}/{len(jobs)}] {prompt} - press Enter, then talk (Ctrl+C stops) ")
            mic.start()
            try:
                if args.seconds:
                    audio = record_fixed(mic, args.seconds)
                else:
                    audio = record_utterance(mic, vad)
            finally:
                mic.stop()
            if audio is None:
                print("  nothing heard, skipped")
                continue
            name = f"{args.prefix}_{index:03d}.wav"
            save(args.out / name, audio)
            rows.append((name, phrase))
            index += 1
            print(f"  saved {name} ({len(audio) / SAMPLE_RATE:.1f} s)")
    except KeyboardInterrupt:
        print()
    if phrases and rows:
        csv_path = args.out / "transcripts.csv"
        new = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new:
                writer.writerow(["file", "text", "speaker"])
            writer.writerows((name, text, args.prefix) for name, text in rows)
        print(f"{len(rows)} rows added to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
