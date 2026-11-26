"""The local llama-server: where the binary is, the command line, and whether it is up.

    uv run python -m grace.intent.local_llm --print-command   # JSON argv (run_llama_server.ps1)
    uv run python -m grace.intent.local_llm --check           # is it answering?

The command follows docs/PLAN.md: the GGUF from router.local.gguf, ``--jinja`` (the model's own
chat template, needed for tool calls), reasoning off, ``-t inference.threads`` (4 on the
i5-12450H P-cores, 6 on the Ryzen 5 5500U; never 12), one slot so the prompt cache is reused
between requests. llama.cpp renamed the reasoning switch over time, so the flag is chosen from
``llama-server --help``: ``--reasoning off`` where it exists, else ``--reasoning-budget 0``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from grace.models import ROOT, resolve

BINARY_NAMES = ("llama-server.exe", "llama-server")
TOOLS_DIR = ROOT / "tools" / "llama.cpp"  # git-ignored; unzip a llama.cpp release here
_REASONING_SWITCH = re.compile(r"(?m)^\s*(?:-\S+,\s*)*--reasoning[ ,=]")


def find_binary(configured: Path | None = None, which: Callable[[str], str | None] = shutil.which):
    """llama-server: router.local.binary, else PATH (winget install ggml.llamacpp), else
    tools/llama.cpp/. None when it isn't installed."""
    if configured is not None:
        path = resolve(configured)
        return path if path.is_file() else None
    for name in BINARY_NAMES:
        found = which(name)
        if found:
            return Path(found)
    for name in BINARY_NAMES:
        candidate = TOOLS_DIR / name
        if candidate.is_file():
            return candidate
    return None


def reasoning_flags(help_text: str) -> list[str]:
    """Flags that switch the model's thinking off, for this llama-server version. Pure."""
    if _REASONING_SWITCH.search(help_text or ""):
        return ["--reasoning", "off"]
    return ["--reasoning-budget", "0"]


def host_port(base_url: str) -> tuple[str, int]:
    url = urlparse(base_url)
    return url.hostname or "127.0.0.1", url.port or 8080


def server_command(
    binary: Path | str,
    gguf: Path | str,
    *,
    threads: int,
    ctx_size: int = 8192,
    base_url: str = "http://127.0.0.1:8080/v1",
    help_text: str = "",
) -> list[str]:
    """argv for llama-server. Pure."""
    if not 1 <= threads <= 8:
        raise ValueError("threads must be 1-8 (never the logical-thread count)")
    host, port = host_port(base_url)
    return [
        str(binary),
        "-m", str(gguf),
        "--host", host,
        "--port", str(port),
        "-t", str(threads),
        "-c", str(ctx_size),
        "-np", "1",  # one slot: every request reuses the same cached prompt prefix
        "--jinja",
        *reasoning_flags(help_text),
    ]  # fmt: skip


def read_help(binary: Path | str, run: Callable[..., Any] = subprocess.run) -> str:
    try:
        proc = run([str(binary), "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def is_up(base_url: str, timeout_s: float = 2.0) -> tuple[bool, str]:
    """(answering, detail). GET {base_url}/models: the OpenAI-compatible listing."""
    import httpx

    try:
        response = httpx.get(base_url.rstrip("/") + "/models", timeout=timeout_s)
    except httpx.HTTPError as exc:
        return False, f"not reachable at {base_url} ({type(exc).__name__})"
    if response.status_code == 503:
        return False, "loading the model"
    if response.status_code != 200:
        return False, f"HTTP {response.status_code} from {base_url}/models"
    try:
        models = [m.get("id", "?") for m in response.json().get("data", [])]
    except ValueError:
        models = []
    return True, f"{base_url}, model {models[0] if models else '?'}"


def wait_until_up(base_url: str, timeout_s: float = 300.0, poll_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if is_up(base_url, timeout_s=2.0)[0]:
            return True
        time.sleep(poll_s)
    return False


def command_from_settings(
    settings: Any, *, threads: int | None = None, base_url: str = ""
) -> list[str]:
    """llama-server argv from settings. Raises FileNotFoundError for a missing binary / model."""
    local = settings.router.local
    binary = find_binary(local.binary)
    if binary is None:
        raise FileNotFoundError(
            "llama-server not found. Install it with 'winget install ggml.llamacpp', or unzip a "
            "llama.cpp Windows release into tools/llama.cpp/, or set router.local.binary."
        )
    gguf = resolve(local.gguf)
    if not gguf.is_file():
        raise FileNotFoundError(
            f"model {gguf} missing: uv run python scripts/fetch_models.py --llm --only-llm"
        )
    return server_command(
        binary,
        gguf,
        threads=threads or settings.inference.threads,
        ctx_size=local.ctx_size,
        base_url=base_url or local.base_url,
        help_text=read_help(binary),
    )


def main(argv: list[str] | None = None) -> int:
    from grace.config import load_settings

    parser = argparse.ArgumentParser(description="GRACE local llama-server helper")
    parser.add_argument("--print-command", action="store_true", help="print llama-server argv")
    parser.add_argument("--check", action="store_true", help="is llama-server answering?")
    parser.add_argument("--threads", type=int, default=None, help="override inference.threads")
    args = parser.parse_args(argv)
    settings = load_settings()
    if args.check:
        up, detail = is_up(settings.router.local.base_url)
        print(("up: " if up else "down: ") + detail)
        return 0 if up else 1
    try:
        command = command_from_settings(settings, threads=args.threads)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(command))
    return 0


if __name__ == "__main__":
    sys.exit(main())
