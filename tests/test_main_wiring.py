"""M2: the legacy main.py loop runs on grammar + registry (no substring matching)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import main
from grace.actions import create_fake_registry


@dataclass
class FakeTTS:
    spoken: list[str] = field(default_factory=list)

    def speak_async(self, text: str) -> None:
        self.spoken.append(text)

    speak = speak_async


class FakeScreenReader:
    def read_screen_text(self) -> str:
        return "screen text"

    def analyze_code(self, code: str) -> str:
        return f"analysis of {code}"

    def debug_code(self, code: str) -> str:
        return f"debug of {code}"


@pytest.fixture
def agent():
    registry, backends = create_fake_registry()
    a = main.GraceVoiceAgent(
        registry=registry, tts=FakeTTS(), recognizer=object(), screen_reader=FakeScreenReader()
    )
    yield a, backends
    a.gate.cancel(announce=False)


@pytest.mark.parametrize(
    ("raw", "stripped"),
    [
        ("execute open notepad", "open notepad"),
        ("open notepad execute", "open notepad"),
        ("Execute, volume 40.", "volume 40"),
        ("execute", ""),
        ("volume 40", "volume 40"),
        ("executed the plan", "executed the plan"),
    ],
)
def test_strip_execute_word(raw: str, stripped: str) -> None:
    assert main.strip_execute_word(raw) == stripped


def test_commands_go_through_registry(agent) -> None:
    a, b = agent
    a._process_command("execute volume 40")
    a._process_command("set brightness to 70 percent")
    a._process_command("open notepad")
    a._process_command("scroll down a lot")
    assert b.volume.level == 40 and b.brightness.level == 70
    assert b.launcher.launched == ["notepad"]
    assert b.input.calls == [("scroll", -1200)]


def test_shutdown_needs_confirmation(agent) -> None:
    a, b = agent
    a._process_command("shut down the computer")
    assert b.power.calls == []
    assert "Confirm shutdown? Say yes or press Ctrl+Alt+Y." in a.tts.spoken
    a._process_command("yes")
    assert b.power.calls == ["shutdown"]


@pytest.mark.parametrize(
    "text", ["turn off the light", "shutter speed", "close call", "I'm shutting the door", "shut"]
)
def test_old_substring_triggers_are_gone(agent, text: str) -> None:
    a, b = agent
    a._process_command(text)
    assert b.power.calls == [] and b.input.calls == [] and a.gate.pending is None
    assert a.tts.spoken[-1] == "I'm not sure how to help with that."


def test_dictation_mode(agent) -> None:
    a, b = agent
    a._process_command("start dictation")
    assert a.mode == "dictation"
    a._process_dictation("Dear team, the volume 40 report is attached.")
    a._process_dictation("stop dictation")
    assert a.mode == "command"
    assert b.input.calls == [("write", "Dear team, the volume 40 report is attached.")]
    assert b.volume.calls == []


def test_app_intents(agent) -> None:
    a, b = agent
    b.clipboard.text = "print(1)"
    a._process_command("debug this code")
    assert a.tts.spoken[-1] == "debug of print(1)"
    a._process_command("exit")
    assert a.is_running is False
