"""Shared helpers for scripts that write measured results into docs/results/*.md."""

from __future__ import annotations

import platform
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STT_RESULTS = ROOT / "docs" / "results" / "stt.md"
ROUTER_RESULTS = ROOT / "docs" / "results" / "router.md"


def replace_section(text: str, name: str, body: str) -> str:
    """Replace what is between ``<!-- name:start -->`` and ``<!-- name:end -->`` (appended at
    the end if the markers are missing)."""
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    block = f"{start}\n{body.strip()}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if pattern.search(text):
        return pattern.sub(lambda _: block, text)
    return text.rstrip() + "\n\n" + block + "\n"


def write_section(path: Path, name: str, body: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(replace_section(text, name, body), encoding="utf-8")


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0-100); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, round(p / 100 * len(ordered) + 0.4999))
    return ordered[min(rank, len(ordered)) - 1]


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def os_name() -> str:
    """platform.platform() calls Windows 11 "Windows-10"; the build number tells them apart."""
    if sys.platform == "win32":
        build = sys.getwindowsversion().build
        return f"Windows {'11' if build >= 22000 else '10'} build {build}"
    return platform.platform(terse=True)


def stamp() -> str:
    return f"{datetime.now():%Y-%m-%d %H:%M} on {cpu_name()} ({os_name()})"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def use_utf8_output() -> None:
    """Result tables contain arrows; a redirected Windows console would crash on them (cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
