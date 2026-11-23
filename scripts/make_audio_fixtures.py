"""Regenerate the VAD test fixtures in tests/fixtures/audio/ (Windows: uses SAPI voices).

    uv run python scripts/make_audio_fixtures.py

Writes 16 kHz mono 16-bit WAVs with seeded low-level room noise:
    silence.wav       4 s of noise, no speech
    one_command.wav   "open notepad"
    two_commands.wav  "volume up" ... 1.5 s pause ... "take a screenshot"
    hey_jarvis.wav    "hey jarvis" ... "open notepad" (wake-word test with the demo model)
and copies Silero VAD (hash-checked, from models/) to tests/fixtures/audio/silero_vad.onnx so
the tests run without models/. These are synthetic voices: the team's real recordings for
accuracy and latency go in tests/audio/ (see tests/audio/README.md).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace.input.audio import SAMPLE_RATE, load_wav  # noqa: E402
from grace.models import MODELS_DIR, SILERO_VAD, files_in, sha256  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "audio"
NOISE = 0.003  # about -50 dBFS: a quiet room


def say(text: str, rate: int = 170) -> np.ndarray:
    import pyttsx3

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "say.wav"
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.save_to_file(text, str(path))
        engine.runAndWait()
        speech = load_wav(path)
    nz = np.flatnonzero(np.abs(speech) > 0.01)  # trim SAPI's own leading/trailing silence
    speech = speech[nz[0] : nz[-1] + 1]
    return 0.5 * speech / max(1e-6, float(np.abs(speech).max()))


def gap(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def write(name: str, parts: list[np.ndarray], rng: np.random.Generator) -> None:
    import soundfile as sf

    audio = np.concatenate(parts)
    audio = audio + rng.normal(0.0, NOISE, len(audio)).astype(np.float32)
    sf.write(OUT / name, np.clip(audio, -1, 1), SAMPLE_RATE, subtype="PCM_16")
    print(f"  {name}: {len(audio) / SAMPLE_RATE:.2f} s")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(6)
    write("silence.wav", [gap(4.0)], rng)
    write("one_command.wav", [gap(0.8), say("open notepad"), gap(1.5)], rng)
    write(
        "two_commands.wav",
        [gap(0.6), say("volume up"), gap(1.5), say("take a screenshot"), gap(1.2)],
        rng,
    )
    # Wake-word path test with the downloadable hey_jarvis model (fetch_models --demo-wakeword).
    write(
        "hey_jarvis.wav",
        [gap(0.8), say("hey jarvis", rate=150), gap(0.6), say("open notepad"), gap(1.2)],
        np.random.default_rng(7),
    )
    (vad,) = files_in("vad")
    src = MODELS_DIR / SILERO_VAD
    if not src.is_file() or sha256(src) != vad.sha256:
        print("models/vad/silero_vad.onnx missing or wrong: run scripts/fetch_models.py first")
        return 1
    shutil.copyfile(src, OUT / "silero_vad.onnx")
    print("  silero_vad.onnx copied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
