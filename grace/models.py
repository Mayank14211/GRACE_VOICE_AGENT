"""Model files GRACE downloads into models/ (git-ignored), with pinned URLs and sha256 hashes.

    uv run python scripts/fetch_models.py            # VAD + wake-word features + Moonshine
    uv run python scripts/fetch_models.py --whisper  # + faster-whisper (extra [stt-whisper])

The doctor checks every file listed here. Paths are relative to ``MODELS_DIR``. The team's own
wake-word model (wakeword.model_path) is trained per docs/wakeword.md and is not listed: its hash
is whatever the team trains.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class ModelFile:
    path: str  # relative to MODELS_DIR
    url: str
    sha256: str
    group: str  # what needs it: vad | wakeword | moonshine:<arch>
    licence: str


SILERO_VAD = "vad/silero_vad.onnx"
OWW_MELSPEC = "openwakeword/melspectrogram.onnx"
OWW_EMBEDDING = "openwakeword/embedding_model.onnx"
OWW_DEMO = "openwakeword/hey_jarvis_v0.1.onnx"  # stand-in for testing before the team's model

_OWW = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
_MOONSHINE = "https://download.moonshine.ai/model"

# Moonshine English models: arch name -> (directory under models/moonshine, ModelArch value).
# The directory mirrors moonshine-voice's own cache layout (download.moonshine.ai/...).
_MOONSHINE_CACHE = "moonshine/download.moonshine.ai/model"
MOONSHINE_ARCHS: dict[str, tuple[str, int]] = {
    "tiny-streaming": (f"{_MOONSHINE_CACHE}/tiny-streaming-en/quantized_26_11_02", 2),
    "small-streaming": (f"{_MOONSHINE_CACHE}/small-streaming-en/quantized_26_11_02", 4),
    "medium-streaming": (f"{_MOONSHINE_CACHE}/medium-streaming-en/quantized_26_11_02", 5),
}

_SMALL = MOONSHINE_ARCHS["small-streaming"][0]
_SMALL_URL = f"{_MOONSHINE}/small-streaming-en/quantized_26_11_02"
_SMALL_GROUP = "moonshine:small-streaming"

MODEL_FILES: tuple[ModelFile, ...] = (
    ModelFile(
        SILERO_VAD,
        "https://github.com/snakers4/silero-vad/raw/v6.3/src/silero_vad/data/silero_vad.onnx",
        "f6f4407b41e05e277e7f5cfd9b9ae957b1dfe763c6076a934734c0b448e09b31",
        "vad",
        "MIT",
    ),
    ModelFile(
        OWW_MELSPEC,
        f"{_OWW}/melspectrogram.onnx",
        "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
        "wakeword",
        "Apache-2.0",
    ),
    ModelFile(
        OWW_EMBEDDING,
        f"{_OWW}/embedding_model.onnx",
        "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
        "wakeword",
        "Apache-2.0",
    ),
    ModelFile(
        OWW_DEMO,
        f"{_OWW}/hey_jarvis_v0.1.onnx",
        "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
        "wakeword-demo",
        "CC-BY-NC-SA-4.0 (testing only)",
    ),
    *(
        ModelFile(f"{_SMALL}/{name}", f"{_SMALL_URL}/{name}", sha, _SMALL_GROUP, "MIT")
        for name, sha in (
            ("adapter.ort", "100d4806a3ba19b739fa9ae5c952e4d5267983090eea4a085e08d3e0bfd6875b"),
            ("cross_kv.ort", "aa0ed033b7065ef744c36dce00228c708b43f2f2410e8ec782a960105c8595f7"),
            ("decoder_kv.ort", "e4fae3869851b861177b5e1619c77ec9262365b2747ec322298d11753e88b38d"),
            ("encoder.ort", "f61486dc7b017cee99196366dcc977a173f45f2f387b0b7977695d0f0565d234"),
            (
                "frontend.model.ort",
                "4a8d5ef7e99a5e3011dbf09ffac4176056587f2aec20f885d606b9f55fbed408",
            ),
            (
                "frontend.weights.ort",
                "5379f3dfebe849c6cbacf84f0aa7441956f9572efe1720d44aa267e7dff93056",
            ),
            (
                "streaming_config.json",
                "26f02b6afb22d60871a5efd85c3d38e569cc0ddb6c5eb6e93d3260152ae8a47a",
            ),
            ("tokenizer.bin", "6884b35fd6377d4c4d32336a0bc152f36b64d1e45b6503683cdc238250a8472d"),
        )
    ),
)

# Local LLM for the router (llama-server, S07). Gemma 4 E4B instruction-tuned, Q4_K_M. The
# official ggml-org repo ships only Q4_0 / Q8_0 / BF16, so this is bartowski's imatrix Q4_K_M
# (same llama.cpp quantisation). 5.4 GB. Gemma licence (Gemma Terms of Use).
LLM_GGUF = "llm/google_gemma-4-E4B-it-Q4_K_M.gguf"
MODEL_FILES = (
    *MODEL_FILES,
    ModelFile(
        LLM_GGUF,
        "https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF/resolve/main/"
        "google_gemma-4-E4B-it-Q4_K_M.gguf",
        "d35a3aa7a6d47de237fd488bb6f7f3efd62908d828f85a63da0ee93acdbefe9f",
        "llm",
        "Gemma Terms of Use",
    ),
)
LLM_GGUF_SIZE = 5_405_170_144

# Needed for voice to start at all (the demo wake-word model is optional).
REQUIRED_GROUPS = ("vad", "wakeword", "moonshine:small-streaming")

WHISPER_DIR = "whisper"  # faster-whisper download_root (Hugging Face cache layout)


def model_path(rel: str | Path, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / rel


def resolve(path: str | Path) -> Path:
    """A settings path (relative = relative to the repo root) as an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def moonshine_dir(arch: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / MOONSHINE_ARCHS[arch][0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, expected: str, *, stamp_suffix: str = ".sha256ok") -> bool:
    """sha256 check that remembers a good result in ``<file>.sha256ok`` (hash, size, mtime),
    so multi-GB files (the LLM) are hashed once, not on every doctor / fetch run."""
    if not path.is_file():
        return False
    st = path.stat()
    token = f"{expected.lower()} {st.st_size} {st.st_mtime_ns}"
    stamp = path.with_name(path.name + stamp_suffix)
    try:
        if stamp.read_text(encoding="utf-8").strip() == token:
            return True
    except OSError:
        pass
    if sha256(path) != expected.lower():
        stamp.unlink(missing_ok=True)
        return False
    try:
        stamp.write_text(token, encoding="utf-8")
    except OSError:
        pass
    return True


def by_path(rel: str) -> ModelFile | None:
    return next((f for f in MODEL_FILES if f.path == rel), None)


def files_in(group: str) -> tuple[ModelFile, ...]:
    return tuple(f for f in MODEL_FILES if f.group == group)


def missing(groups: tuple[str, ...] = REQUIRED_GROUPS, models_dir: Path = MODELS_DIR) -> list[str]:
    """Relative paths of required files that are not on disk (hashes are the doctor's job)."""
    return [
        f.path for f in MODEL_FILES if f.group in groups and not (models_dir / f.path).is_file()
    ]
