"""feedback/earcons.py and metrics/log.py."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import pytest

from grace.feedback.earcons import TONES, BeepEarcons, FakeEarcons, create_earcons
from grace.metrics.log import MetricsLog, session_path


def test_beep_earcons_play_off_the_calling_thread() -> None:
    played: list[tuple[int, int, str]] = []
    done = threading.Event()

    def beep(freq: int, ms: int) -> None:
        played.append((freq, ms, threading.current_thread().name))
        if len(played) == len(TONES["panic"]):
            done.set()

    earcons = BeepEarcons(beep)
    earcons.play("panic")
    assert done.wait(2)
    earcons.close()
    assert [(f, m) for f, m, _ in played] == list(TONES["panic"])
    assert {t for _, _, t in played} == {"grace-earcons"}


def test_beep_failure_is_silent_and_the_thread_survives() -> None:
    calls: list[int] = []
    done = threading.Event()

    def beep(freq: int, ms: int) -> None:
        calls.append(freq)
        if len(calls) == 2:
            done.set()
        raise RuntimeError("no audio device")

    earcons = BeepEarcons(beep)
    earcons.play("on")
    earcons.play("off")
    assert done.wait(2)
    earcons.close()


def test_unknown_earcon_is_a_programming_error() -> None:
    with pytest.raises(KeyError):
        FakeEarcons().play("trumpet")


def test_disabled_earcons_are_silent() -> None:
    create_earcons(enabled=False).play("on")  # no sound, no error


def test_metrics_rows_are_jsonl_with_time_and_kind(tmp_path: Path) -> None:
    path = tmp_path / "m" / "s.jsonl"
    metrics = MetricsLog(path, clock=lambda: 1.23456)
    metrics.write("toggle", what="mic", on=False, latency_ms=12.5)
    metrics.write("marked_wrong", action={"name": "volume.set", "ok": True})
    metrics.close()
    metrics.write("ignored", after="close")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"t": 1.235, "kind": "toggle", "what": "mic", "on": False, "latency_ms": 12.5},
        {"t": 1.235, "kind": "marked_wrong", "action": {"name": "volume.set", "ok": True}},
    ]


def test_metrics_redact_secrets(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    fake_key = "gsk_" + "a1B2" * 12
    metrics = MetricsLog(path, secrets=["my-configured-secret-value"])
    metrics.write("error", detail=f"bad key {fake_key}", other=["my-configured-secret-value"])
    metrics.close()
    text = path.read_text(encoding="utf-8")
    assert fake_key not in text and "my-configured-secret-value" not in text


def test_session_path_is_timestamped(tmp_path: Path) -> None:
    p = session_path(tmp_path, datetime(2026, 11, 23, 9, 5, 7))
    assert p == tmp_path / "session-20261123-090507.jsonl"
