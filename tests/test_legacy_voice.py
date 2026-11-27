"""The voice agent (was main.py): grammar + registry (no substring matching), dictation, replies.

Listening (wake word, follow-ups, push-to-talk) is tested in tests/test_voice_worker.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from grace import legacy_voice
from grace.actions import create_fake_registry
from grace.intent.assistant import ExtraResult
from grace.intent.router import RouteResult
from grace.intent.types import Intent


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
    assert a.tts.spoken[-1] == "Code analysis needs a Groq or OpenRouter key in the .env file."
    a.extras = FakeExtras()
    a._process_command("debug this code")
    assert a.tts.spoken[-1] == "review of ```\nprint(1)\n```"
    a._process_command("exit")
    assert a.is_running is False


# --------------------------------------------------------------------------- router fallback
class FakeExtras:
    def __init__(self) -> None:
        self.searched: list[str] = []

    def complete(self, prompt: str, *, instruction: str | None = None) -> str:
        return f"review of {prompt}"

    def web_search(self, query: str) -> ExtraResult:
        self.searched.append(query)
        return ExtraResult("It is sunny. Twenty degrees. Light wind.", sources=[("S", "https://s")])

    def generate_image(self, prompt: str) -> ExtraResult:
        return ExtraResult(f"Image of {prompt} saved.")


class FakeRouter:
    """Returns scripted intents; records what it was asked and the context it got."""

    def __init__(self, *intents: Intent, reply: str = "") -> None:
        self.intents = list(intents)
        self.reply = reply
        self.asked: list[str] = []
        self.lasts: list = []

    def route(self, text, dialogue=None) -> RouteResult:
        self.asked.append(text)
        self.lasts.append(dialogue.last() if dialogue is not None else None)
        if not self.intents:
            return RouteResult(reply=self.reply or "Sorry.", ok=False)
        return RouteResult(list(self.intents), provider="fake")


def answer(text: str) -> Intent:
    return Intent("app.answer", {"text": text}, source="router")


def test_unmatched_speech_goes_to_the_router(agent) -> None:
    a, b = agent
    a.router = FakeRouter(answer("Paris."))
    a._process_command("what is the capital of France")
    assert a.router.asked == ["what is the capital of France"]
    assert a.tts.spoken[-1] == "Paris."
    a._process_command("volume 40")  # grammar hits never reach the LLM
    assert a.router.asked == ["what is the capital of France"] and b.volume.level == 40


def test_without_a_router_unmatched_speech_is_declined(agent) -> None:
    a, b = agent
    a._process_command("tell me a joke")
    assert a.tts.spoken[-1] == "I'm not sure how to help with that."


def test_router_failure_reply_is_spoken(agent) -> None:
    a, _ = agent
    a.router = FakeRouter(reply="I can't reach an assistant right now.")
    a._process_command("tell me a joke")
    assert a.tts.spoken[-1] == "I can't reach an assistant right now."


def test_router_intents_run_through_the_registry_in_order(agent) -> None:
    a, b = agent
    a.router = FakeRouter(
        Intent("volume.mute", {"mute": True}, source="router"),
        Intent("apps.open", {"name": "youtube"}, source="router"),
    )
    a._process_command("mute it and put youtube on")
    assert b.volume.muted is True and b.launcher.urls == ["https://www.youtube.com"]
    assert a.dialogue.last().intent.name == "apps.open"


def test_risky_router_intent_needs_confirmation_and_runs_last(agent) -> None:
    a, b = agent
    a.router = FakeRouter(
        Intent("system.shutdown", source="router"),
        Intent("volume.set", {"level": 10}, source="router"),
    )
    a._process_command("lower the volume and turn everything off for tonight")
    assert b.volume.level == 10 and b.power.calls == []  # safe one ran, risky one is held
    assert a.gate.pending is not None and a.gate.pending.name == "system.shutdown"
    assert "Confirm shutdown" in a.tts.spoken[-1]
    assert a.dialogue.last().intent.name == "volume.set"  # risky actions are never follow-ups
    a._process_command("yes")
    assert b.power.calls == ["shutdown"]
    assert a.dialogue.last() is None


def test_unmatched_speech_cancels_a_pending_risky_action(agent) -> None:
    a, b = agent
    a.router = FakeRouter(answer("Ha."))
    a._process_command("shut down the computer")
    a._process_command("tell me a joke")
    a._process_command("yes")
    assert b.power.calls == []


def test_start_over_resets_the_dialogue(agent) -> None:
    a, _ = agent
    a._process_command("volume 40")
    assert a.dialogue.last() is not None and a.dialogue.history()
    a._process_command("start over")
    assert a.dialogue.last() is None
    assert [e.user for e in a.dialogue.history()] == ["start over"]


def test_exchanges_are_remembered_for_the_router(agent) -> None:
    a, _ = agent
    a.router = FakeRouter(answer("Paris."))
    a._process_command("what is the capital of France")
    a._process_command("volume 40")
    [first, second] = a.dialogue.history()
    assert (first.user, first.grace) == ("what is the capital of France", "Paris.")
    assert (second.user, second.grace) == ("volume 40", "Volume set to 40 percent.")


def test_web_search_and_images_from_the_router(agent) -> None:
    a, _ = agent
    a.router = FakeRouter(Intent("app.web_search", {"query": "weather"}, source="router"))
    a._process_command("what's the weather like")
    assert a.tts.spoken[-1] == "Web search needs a TAVILY_API_KEY in the .env file."
    a.extras = FakeExtras()
    a._process_command("what's the weather like")
    assert a.extras.searched == ["weather"]
    assert a.tts.spoken[-1] == "It is sunny. Twenty degrees."  # two sentences at most
    a.router = FakeRouter(Intent("app.generate_image", {"prompt": "a cat"}, source="router"))
    a._process_command("draw me a cat")
    assert a.tts.spoken[-1] == "Image of a cat saved."


# --------------------------------------------------------------------------- follow-ups
def test_followups_after_a_grammar_command(agent) -> None:
    a, b = agent
    a.router = FakeRouter(answer("unused"))
    a._process_command("volume 40")
    a._process_command("a bit more")
    assert b.volume.level == 45
    a._process_command("now the brightness")
    assert b.brightness.level == 55  # brightness.step +5 from the fake's 50
    a._process_command("undo that")
    assert b.brightness.level == 50
    assert a.router.asked == []  # all resolved without an LLM


def test_undo_that_restores_the_value_a_set_replaced(agent) -> None:
    a, b = agent
    b.volume.level = 70
    a._process_command("volume 20")
    a._process_command("undo that")
    assert b.volume.level == 70
    a._process_command("undo that")  # undoing the undo
    assert b.volume.level == 20


def test_plain_undo_is_still_ctrl_z(agent) -> None:
    a, b = agent
    a._process_command("volume 20")
    a._process_command("undo")
    assert b.volume.level == 20 and ("hotkey", "ctrl", "z") in b.input.calls


def test_undo_that_without_a_reversible_action_is_ctrl_z(agent) -> None:
    a, b = agent
    a._process_command("type hello")
    a._process_command("undo that")
    assert ("hotkey", "ctrl", "z") in b.input.calls


def test_unresolved_followup_goes_to_the_router_with_the_last_action(agent) -> None:
    a, _ = agent
    a.router = FakeRouter(answer("Which one?"))
    a._process_command("volume 40")
    a._process_command("make the second one louder")
    assert a.router.asked == ["make the second one louder"]
    assert a.router.lasts[0].intent.name == "volume.set"


def test_risky_actions_are_never_repeated(agent) -> None:
    a, b = agent
    a.router = FakeRouter(answer("No."))
    a._process_command("shut down the computer")
    a._process_command("yes")
    a._process_command("again")
    assert b.power.calls == ["shutdown"] and a.router.asked == ["again"]


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
