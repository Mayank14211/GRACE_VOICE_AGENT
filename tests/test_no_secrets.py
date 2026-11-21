"""Fail if anything in the working tree looks like an API key (AIza..., sk-..., hf_..., gsk_...).

Scans every file under the repo root, tracked or not, except:
- tool/cache dirs (.venv, .git, caches, models/, logs/);
- the local, git-ignored key files (.env, .env.*) — that is where keys are supposed to live.
  .env.example IS scanned and must stay empty.
Matches are reported as file:line and pattern name only; the value is never printed.
Fake keys in tests must be assembled at runtime (e.g. "gsk" + "_" + ...).
"""

from __future__ import annotations

import os
from pathlib import Path

from grace.redaction import find_secrets

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    "models",
    "logs",
}
MAX_BYTES = 5 * 1024 * 1024


def is_local_key_file(name: str) -> bool:
    return (name == ".env" or name.startswith(".env.")) and name != ".env.example"


def iter_text_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if is_local_key_file(name):
                continue
            path = Path(dirpath) / name
            try:
                if path.stat().st_size > MAX_BYTES:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:8192]:  # binary
                continue
            yield path, data.decode("utf-8", errors="ignore")


def scan(root: Path) -> list[str]:
    findings = []
    for path, text in iter_text_files(root):
        for pattern_name, offset in find_secrets(text):
            line = text.count("\n", 0, offset) + 1
            findings.append(f"{path.relative_to(root).as_posix()}:{line} ({pattern_name})")
    return findings


def test_repository_contains_no_secrets() -> None:
    findings = scan(ROOT)
    assert not findings, (
        "Key-like strings found (remove them, put real keys in .env, and REVOKE the key):\n  "
        + "\n  ".join(findings)
    )


def test_scanner_finds_planted_key_and_skips_local_env(tmp_path: Path) -> None:
    fake = "gsk" + "_" + "P" * 52
    (tmp_path / "leak.py").write_text(f'KEY = "{fake}"\n')
    (tmp_path / ".env").write_text(f"GROQ_API_KEY={fake}\n")
    (tmp_path / ".env.local").write_text(f"GROQ_API_KEY={fake}\n")
    (tmp_path / ".env.example").write_text(f"GROQ_API_KEY={fake}\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "x.py").write_text(fake)
    (tmp_path / "blob.bin").write_bytes(b"\0" + fake.encode())

    findings = scan(tmp_path)

    assert sorted(findings) == [".env.example:1 (groq_api_key)", "leak.py:1 (groq_api_key)"]
    assert all(fake not in f for f in findings)
