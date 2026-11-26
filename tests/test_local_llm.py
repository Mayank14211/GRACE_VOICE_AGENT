"""llama-server helpers (command line, binary lookup, health) and the cached model hash check."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from grace import doctor, models
from grace.intent import local_llm
from grace.intent.local_llm import (
    find_binary,
    host_port,
    is_up,
    reasoning_flags,
    server_command,
)

NEW_HELP = """----- sampling params -----
-rea,  --reasoning [on|off|auto]         use reasoning (thinking) if the model supports it
--reasoning-budget N                     deprecated
"""
OLD_HELP = """--reasoning-format FORMAT   controls whether thought tags are allowed
--reasoning-budget N        controls the amount of thinking allowed; currently only one of:
                            -1 for unrestricted thinking budget, or 0 to disable thinking
"""


def test_reasoning_flag_follows_the_installed_version() -> None:
    assert reasoning_flags(NEW_HELP) == ["--reasoning", "off"]
    assert reasoning_flags(OLD_HELP) == ["--reasoning-budget", "0"]  # --reasoning-format isn't it
    assert reasoning_flags("") == ["--reasoning-budget", "0"]


def test_server_command() -> None:
    cmd = server_command(
        "llama-server.exe",
        "models/llm/m.gguf",
        threads=4,
        ctx_size=8192,
        base_url="http://127.0.0.1:8080/v1",
        help_text=NEW_HELP,
    )
    assert cmd == [
        "llama-server.exe", "-m", "models/llm/m.gguf", "--host", "127.0.0.1", "--port", "8080",
        "-t", "4", "-c", "8192", "-np", "1", "--jinja", "--reasoning", "off",
    ]  # fmt: skip
    for bad in (0, 12):  # never the logical-thread count
        with pytest.raises(ValueError):
            server_command("x", "y", threads=bad)


def test_host_port() -> None:
    assert host_port("http://127.0.0.1:8089/v1") == ("127.0.0.1", 8089)
    assert host_port("http://localhost/v1") == ("localhost", 8080)


def test_find_binary(tmp_path: Path, monkeypatch) -> None:
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"")
    assert find_binary(exe) == exe
    assert find_binary(tmp_path / "missing.exe") is None
    assert find_binary(None, which=lambda name: str(exe)) == exe
    monkeypatch.setattr(local_llm, "TOOLS_DIR", tmp_path)
    assert find_binary(None, which=lambda name: None) == exe
    monkeypatch.setattr(local_llm, "TOOLS_DIR", tmp_path / "nothing")
    assert find_binary(None, which=lambda name: None) is None


def test_is_up(monkeypatch) -> None:
    def fake_get(url, timeout):
        request = httpx.Request("GET", url)
        if "8080" in url:
            return httpx.Response(200, json={"data": [{"id": "gemma"}]}, request=request)
        if "8081" in url:
            return httpx.Response(503, request=request)
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(httpx, "get", fake_get)
    assert is_up("http://127.0.0.1:8080/v1") == (True, "http://127.0.0.1:8080/v1, model gemma")
    assert is_up("http://127.0.0.1:8081/v1") == (False, "loading the model")
    up, detail = is_up("http://127.0.0.1:8082/v1")
    assert not up and "not reachable" in detail


def test_print_command_explains_what_is_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(local_llm, "find_binary", lambda configured=None: None)
    assert local_llm.main(["--print-command"]) == 1
    assert "winget install ggml.llamacpp" in capsys.readouterr().err


def test_print_command_is_json(monkeypatch, tmp_path: Path, capsys) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setattr(local_llm, "find_binary", lambda configured=None: Path("llama-server"))
    monkeypatch.setattr(local_llm, "resolve", lambda p: gguf)
    monkeypatch.setattr(local_llm, "read_help", lambda binary: OLD_HELP)
    assert local_llm.main(["--print-command", "--threads", "6"]) == 0
    argv = json.loads(capsys.readouterr().out)
    assert argv[argv.index("-t") + 1] == "6" and "--jinja" in argv


# --------------------------------------------------------------------------- hash cache
def test_verified_hashes_once_and_notices_changes(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "m.gguf"
    f.write_bytes(b"model bytes")
    good = models.sha256(f)
    calls: list[Path] = []
    real = models.sha256
    monkeypatch.setattr(models, "sha256", lambda p: calls.append(p) or real(p))
    assert models.verified(f, good) and models.verified(f, good)
    assert len(calls) == 1  # second check used the stamp
    f.write_bytes(b"other bytes!")  # size / mtime change -> hashed again, and it fails
    assert not models.verified(f, good) and len(calls) == 2
    assert not (tmp_path / "m.gguf.sha256ok").exists()
    assert not models.verified(tmp_path / "missing", good)


def test_llm_is_in_the_manifest_but_not_required() -> None:
    entry = models.by_path(models.LLM_GGUF)
    assert entry is not None and entry.group == "llm" and len(entry.sha256) == 64
    assert "llm" not in models.REQUIRED_GROUPS
    assert all(path != models.LLM_GGUF for path, _ in doctor.MODELS)


# --------------------------------------------------------------------------- doctor
def test_doctor_llama_server_not_configured(tmp_path: Path) -> None:
    [result] = doctor.check_llama_server(tmp_path / "none.yaml", tmp_path / ".env")
    assert result.status == doctor.NOT_CONFIGURED


def test_doctor_llama_server_down_is_a_failure_only_in_demo_mode(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(local_llm, "is_up", lambda url, timeout_s=2.0: (False, "not reachable"))
    monkeypatch.setattr(local_llm, "find_binary", lambda configured=None: Path("llama-server"))
    enabled = tmp_path / "on.yaml"
    enabled.write_text(
        f"router: {{local: {{enabled: true, gguf: '{(tmp_path / 'x.gguf').as_posix()}'}}}}\n"
    )
    statuses = {r.name: r.status for r in doctor.check_llama_server(enabled, tmp_path / ".env")}
    assert statuses == {"model x.gguf": doctor.WARN, "llama-server": doctor.WARN}
    demo = tmp_path / "demo.yaml"
    demo.write_text(
        f"demo_mode: true\nrouter: {{local: {{gguf: '{(tmp_path / 'x.gguf').as_posix()}'}}}}\n"
    )
    statuses = {r.name: r.status for r in doctor.check_llama_server(demo, tmp_path / ".env")}
    assert set(statuses.values()) == {doctor.FAIL}


def test_doctor_llama_server_up(tmp_path: Path, monkeypatch) -> None:
    gguf = tmp_path / "custom.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setattr(local_llm, "is_up", lambda url, timeout_s=2.0: (True, "ok, model m"))
    cfg = tmp_path / "on.yaml"
    cfg.write_text(f"router: {{local: {{enabled: true, gguf: '{gguf.as_posix()}'}}}}\n")
    results = doctor.check_llama_server(cfg, tmp_path / ".env")
    assert [(r.status, r.detail) for r in results] == [
        (doctor.PASS, "custom model (hash not pinned)"),
        (doctor.PASS, "ok, model m"),
    ]


def test_doctor_router_check(tmp_path: Path) -> None:
    result = doctor.check_router(tmp_path / "none.yaml", tmp_path / ".env")
    assert result.status == doctor.NOT_CONFIGURED and "8 tools" in result.detail
    assert "never (exact phrase only)" in result.detail
    cfg = tmp_path / "s.yaml"
    cfg.write_text("demo_mode: true\nrouter: {allow_risky: true}\n")
    result = doctor.check_router(cfg, tmp_path / ".env")
    assert result.status == doctor.PASS and "chain: local" in result.detail
    assert "confirmation only" in result.detail
