"""Download GRACE's model files into models/ and check their sha256 hashes.

    uv run python scripts/fetch_models.py                 # Silero VAD, openWakeWord features,
                                                          # Moonshine small-streaming (English)
    uv run python scripts/fetch_models.py --demo-wakeword # + "hey jarvis" model to test the
                                                          #   wake-word path before ours is trained
    uv run python scripts/fetch_models.py --whisper       # + faster-whisper base (needs
                                                          #   uv sync --extra stt-whisper)
    uv run python scripts/fetch_models.py --moonshine tiny-streaming  # another Moonshine size

Files that are already present with the right hash are skipped. A hash mismatch is an error
(the file is not replaced silently). Everything lands in models/ (git-ignored).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace.models import (  # noqa: E402
    MODEL_FILES,
    MODELS_DIR,
    REQUIRED_GROUPS,
    WHISPER_DIR,
    ModelFile,
    moonshine_dir,
    sha256,
)

TIMEOUT_S = 60.0


def fetch(f: ModelFile, models_dir: Path = MODELS_DIR) -> str:
    """Download ``f`` unless it is already there. Returns ok | skipped | error text."""
    import httpx

    dest = models_dir / f.path
    if dest.is_file():
        if sha256(dest) == f.sha256:
            return "present, hash ok"
        return f"HASH MISMATCH (delete {dest} and run again)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        try:
            with httpx.stream("GET", f.url, follow_redirects=True, timeout=TIMEOUT_S) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes(1 << 16):
                    tmp.write(chunk)
        except httpx.HTTPError as exc:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            return f"download failed: {type(exc).__name__}: {exc}"
    got = sha256(tmp_path)
    if got != f.sha256:
        tmp_path.unlink(missing_ok=True)
        return f"downloaded file has the wrong hash ({got[:12]}...), not saved"
    tmp_path.replace(dest)
    return "downloaded, hash ok"


def fetch_moonshine_arch(arch: str, models_dir: Path = MODELS_DIR) -> str:
    """Other Moonshine sizes via moonshine-voice's own downloader (it checks CRC32C)."""
    from moonshine_voice.download import download_model_from_info, find_model_info
    from moonshine_voice.moonshine_api import ModelArch

    from grace.models import MOONSHINE_ARCHS

    info = find_model_info("en", ModelArch(MOONSHINE_ARCHS[arch][1]))
    download_model_from_info(info, cache_root=models_dir / "moonshine")
    return f"{moonshine_dir(arch, models_dir)} (not hash-pinned; moonshine checks CRC32C)"


def fetch_whisper(size: str, models_dir: Path = MODELS_DIR) -> str:
    try:
        from faster_whisper.utils import download_model
    except ImportError:
        return "faster-whisper not installed: uv sync --extra stt-whisper"
    path = download_model(size, cache_dir=str(models_dir / WHISPER_DIR))
    return f"{path} (Hugging Face Systran/faster-whisper-{size})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo-wakeword", action="store_true", help="also the hey_jarvis model")
    parser.add_argument("--whisper", nargs="?", const="base", default=None, metavar="SIZE",
                        help="also faster-whisper (default size: base)")  # fmt: skip
    parser.add_argument("--moonshine", action="append", default=[], metavar="ARCH",
                        choices=["tiny-streaming", "medium-streaming"],
                        help="also another Moonshine size")  # fmt: skip
    args = parser.parse_args(argv)

    groups = set(REQUIRED_GROUPS) | ({"wakeword-demo"} if args.demo_wakeword else set())
    failures = 0
    for f in MODEL_FILES:
        if f.group not in groups:
            continue
        result = fetch(f)
        failures += not result.endswith("hash ok")
        print(f"  {f.path:<95} {result}")
    for arch in args.moonshine:
        print(f"  moonshine {arch}: {fetch_moonshine_arch(arch)}")
    if args.whisper:
        print(f"  faster-whisper {args.whisper}: {fetch_whisper(args.whisper)}")
    print("All model files present." if not failures else f"{failures} file(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
