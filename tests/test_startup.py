"""Acceptance (M1): the app starts with an empty .env and reports disabled providers clearly."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from grace.config import ENV_KEYS

ROOT = Path(__file__).resolve().parent.parent


def run_main(*args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in ENV_KEYS.values()}
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.windows
def test_starts_with_empty_env_and_reports_disabled_providers(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("")
    proc = run_main(
        "--check-config", "--env-file", str(env_file), "--settings", str(tmp_path / "none.yaml")
    )
    assert proc.returncode == 0, proc.stderr
    for var in ENV_KEYS.values():
        assert f"{var} not set in .env" in proc.stdout
        assert f"disabled ({var} not set in .env)" in proc.stderr  # logged warning
    assert "no LLM provider enabled" in proc.stderr


@pytest.mark.windows
def test_invalid_settings_exit_with_readable_error(tmp_path: Path) -> None:
    bad = tmp_path / "settings.yaml"
    bad.write_text("inference: {threads: 12}\n")
    proc = run_main("--check-config", "--settings", str(bad), "--env-file", str(tmp_path / ".env"))
    assert proc.returncode == 2
    assert "Configuration error" in proc.stderr
    assert "inference.threads" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.windows
def test_cli_no_longer_accepts_keys() -> None:
    proc = run_main("--help")
    assert proc.returncode == 0, proc.stderr
    for flag in ("--gemini-key", "--openai-key"):
        assert flag not in proc.stdout
    assert "--check-config" in proc.stdout
