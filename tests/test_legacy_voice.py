"""M2: the legacy main.py loop runs on grammar + registry (no substring matching)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import main
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
        ("Execute, volume 40.", "volume 40."),  # the grammar ignores the "."
        ("open notepad execute.", "open notepad"),
        ("type Hello, World!", "type Hello, World!"),
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


# --------------------------------------------------------------------------- follow-ups
class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class ScriptedRecognizer:
    """Each listen takes 3 s of fake time and returns the next utterance ("" = silence)."""

    def __init__(self, clock: Clock, utterances: list[str]) -> None:
        self.clock, self.queue, self.listens = clock, list(utterances), 0

    def listen_for_command(self, phrase_time_limit: int = 5) -> str:
        self.listens += 1
        assert self.listens < 100, "conversation loop did not end"
        self.clock.now += 3
        return self.queue.pop(0) if self.queue else ""


def conversation_agent(utterances: list[str], follow_up_s: float = 10.0):
    clock = Clock()
    rec = ScriptedRecognizer(clock, utterances)
    registry, b = create_fake_registry()
    a = main.GraceVoiceAgent(
        registry=registry,
        tts=FakeTTS(),
        recognizer=rec,
        screen_reader=FakeScreenReader(),
        follow_up_s=follow_up_s,
        clock=clock,
    )
    a.is_running = True
    return a, b, rec


def test_follow_up_commands_need_no_wake_word() -> None:
    """The reported bug: open notepad, type, close the window, with one wake word."""
    a, b, rec = conversation_agent(["type Hello, World!", "close this window"])
    a._on_wake_word_detected("open notepad")  # said in the same breath as the wake word
    assert b.launcher.launched == ["notepad"]
    assert b.input.calls == [("write", "Hello, World!"), ("hotkey", "alt", "f4")]
    assert not a._session_open
    assert rec.listens == 2 + 4  # 2 follow-ups, then silence until 10 s idle (4 x 3 s)


def test_wake_word_alone_asks_and_listens() -> None:
    a, b, _ = conversation_agent(["volume 40"])
    a._on_wake_word_detected("")
    assert a.tts.spoken[0] == "Yes?" and b.volume.level == 40


def test_a_pause_shorter_than_the_window_keeps_the_conversation() -> None:
    a, b, _ = conversation_agent(["volume 40", "", "", "volume 20"])  # 6 s of silence
    a._on_wake_word_detected("")
    assert b.volume.calls == [("set_level", 40), ("set_level", 20)]


def test_a_long_pause_ends_the_conversation() -> None:
    a, b, rec = conversation_agent(["volume 40", "", "", "", "", "volume 20"])  # 12 s silence
    a._on_wake_word_detected("")
    assert b.volume.calls == [("set_level", 40)] and rec.queue == ["volume 20"]


def test_stop_listening_ends_the_conversation() -> None:
    a, b, rec = conversation_agent(["volume 40", "stop listening", "volume 10"])
    a._on_wake_word_detected("")
    assert b.volume.level == 40 and rec.queue == ["volume 10"]
    assert "Say the wake word" in a.tts.spoken[-1]


def test_follow_up_zero_means_one_command_per_wake_word() -> None:
    a, b, rec = conversation_agent(["type hi"], follow_up_s=0)
    a._on_wake_word_detected("open notepad")
    assert b.launcher.launched == ["notepad"] and b.input.calls == [] and rec.queue == ["type hi"]


def test_dictation_survives_pauses_until_stopped() -> None:
    a, b, _ = conversation_agent(
        ["start dictation", "", "", "", "", "", "Dear team,", "stop dictation"]
    )
    a._on_wake_word_detected("")
    assert b.input.calls == [("write", "Dear team,")] and a.mode == "command"


def test_waits_for_tts_before_listening(monkeypatch) -> None:
    a, _, rec = conversation_agent(["volume 40"])
    order: list[str] = []
    a.tts.wait_until_done = lambda: order.append("wait")
    original = rec.listen_for_command
    rec.listen_for_command = lambda phrase_time_limit=5: (order.append("listen"), original())[1]
    monkeypatch.setattr(main.time, "sleep", lambda s: None)
    a._on_wake_word_detected("")
    assert order[:2] == ["wait", "listen"]


def test_split_wake_word() -> None:
    from speech_recognition_module import split_wake_word

    assert split_wake_word("Listen, open notepad", "listen") == "open notepad"
    assert split_wake_word("listen", "listen") == ""
    assert split_wake_word("I was listening to music", "listen") is None
    assert split_wake_word("hey LISTEN type Hi", "listen") == "type Hi"
    assert split_wake_word(None, "listen") is None
