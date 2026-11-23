"""The voice agent (was main.py): grammar + registry (no substring matching), dictation, replies.

Listening (wake word, follow-ups, push-to-talk) is tested in tests/test_voice_worker.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from grace import legacy_voice
from grace.actions import create_fake_registry
from grace.intent.assistant import AssistantReply


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
    a = legacy_voice.GraceVoiceAgent(
        registry=registry, tts=FakeTTS(), screen_reader=FakeScreenReader()
    )
    yield a, backends
    a.gate.cancel(announce=False)


@pytest.mark.parametrize(
    ("raw", "stripped"),
    [
        ("execute open notepad", "open notepad"),
        ("open notepad execute", "open notepad"),
        ("Execute, volume 40.", "volume 40."),  # the grammar ignores the "."
        ("open notepad execute.", "open notepad"),
        ("type Hello, World!", "type Hello, World!"),
        ("execute", ""),
        ("volume 40", "volume 40"),
        ("executed the plan", "executed the plan"),
    ],
)
def test_strip_execute_word(raw: str, stripped: str) -> None:
    assert legacy_voice.strip_execute_word(raw) == stripped


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
    assert a.tts.spoken[-1] == "Code analysis needs a GEMINI_API_KEY in the .env file."
    a.assistant = FakeAssistant()
    a._process_command("debug this code")
    assert a.tts.spoken[-1] == "review of ```\nprint(1)\n```"
    a._process_command("exit")
    assert a.is_running is False


# --------------------------------------------------------------------------- Gemini fallback
class FakeAssistant:
    def __init__(self) -> None:
        self.asked: list[str] = []
        self.resets = 0

    def ask(self, text: str) -> AssistantReply:
        self.asked.append(text)
        return AssistantReply(f"answer to {text}", sources=[("Src", "https://s.example")])

    def complete(self, prompt: str, *, instruction: str | None = None) -> str:
        return f"review of {prompt}"

    def reset(self) -> None:
        self.resets += 1


def test_unmatched_speech_goes_to_the_assistant(agent) -> None:
    a, b = agent
    a.assistant = FakeAssistant()
    a._process_command("what is the capital of France")
    assert a.assistant.asked == ["what is the capital of France"]
    assert a.tts.spoken[-1] == "answer to what is the capital of France"
    a._process_command("volume 40")  # grammar hits never reach the LLM
    assert a.assistant.asked == ["what is the capital of France"] and b.volume.level == 40


def test_unmatched_speech_cancels_a_pending_risky_action(agent) -> None:
    a, b = agent
    a.assistant = FakeAssistant()
    a._process_command("shut down the computer")
    a._process_command("tell me a joke")
    a._process_command("yes")
    assert b.power.calls == []


def test_start_over_resets_the_assistant(agent) -> None:
    a, _ = agent
    a.assistant = FakeAssistant()
    a._process_command("start over")
    assert a.assistant.resets == 1


# --------------------------------------------------------------------------- from the pipeline
def test_handle_utterance_runs_a_command(agent) -> None:
    a, b = agent
    heard: list[str] = []
    a._on_heard = heard.append
    a.handle_utterance("Volume 40.")
    a.handle_utterance("   ")  # nothing said: nothing happens
    a.handle_utterance("execute")
    assert b.volume.level == 40 and heard == ["Volume 40."]


def test_handle_utterance_after_exit_does_nothing(agent) -> None:
    a, b = agent
    a.is_running = False
    a.handle_utterance("volume 40")
    assert b.volume.calls == []


def test_dictation_separates_sentences(agent) -> None:
    a, b = agent
    a.handle_utterance("start dictation")
    a.handle_utterance("Dear team,")
    a.handle_utterance("the report is attached.")
    a.handle_utterance("stop dictation")
    a.handle_utterance("start dictation")
    a.handle_utterance("New note.")
    assert b.input.calls == [
        ("write", "Dear team,"),
        ("write", " the report is attached."),
        ("write", "New note."),
    ]


def test_mic_off_cancels_dictation_quietly(agent) -> None:
    a, _ = agent
    modes: list[str] = []
    a._on_mode = modes.append
    a.handle_utterance("start dictation")
    spoken = len(a.tts.spoken)
    a.cancel_dictation()
    assert a.mode == "command" and modes == ["dictation", "command"]
    assert len(a.tts.spoken) == spoken  # no speech: the user just turned the mic off
    a.cancel_dictation()  # not dictating: nothing
    assert modes == ["dictation", "command"]


def test_stop_listening_tells_the_pipeline(agent) -> None:
    a, _ = agent
    ended: list[bool] = []
    a._on_end_session = lambda: ended.append(True)
    a.handle_utterance("stop listening")
    assert ended == [True] and "Say the wake word" in a.tts.spoken[-1]
