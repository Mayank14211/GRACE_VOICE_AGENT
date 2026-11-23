"""grace/input/voice.py: one owner of the mic; mic gating, push-to-talk, cancel drops results."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from grace import legacy_voice
from grace.actions import create_fake_registry
from grace.input.voice import VoiceWorker, split_wake_word


@dataclass
class FakeTTS:
    spoken: list[str] = field(default_factory=list)

    def speak_async(self, text: str) -> None:
        self.spoken.append(text)

    speak = speak_async


class Mic:
    """Fake legacy recognizer: wake passes and command listens pop scripted utterances.

    ``during`` runs inside a listen, to simulate a key press or toggle while listening.
    """

    def __init__(self, passes: list[str | None], commands: list[str] | None = None) -> None:
        self.passes = list(passes)
        self.commands = list(commands or [])
        self.during: Callable[[], None] | None = None
        self.log: list[str] = []

    def _during(self) -> None:
        if self.during is not None:
            hook, self.during = self.during, None
            hook()

    def listen_for_wake_phrase(self) -> str | None:
        self.log.append("wake")
        self._during()
        return self.passes.pop(0) if self.passes else None

    def listen_for_command(self, phrase_time_limit: int = 5) -> str:
        self.log.append("command")
        self._during()
        return self.commands.pop(0) if self.commands else ""


class MicSwitch:
    def __init__(self, on: bool = True) -> None:
        self.on = on

    def __call__(self) -> bool:
        return self.on


def setup(passes: list[str | None], commands: list[str] | None = None, mic_on: bool = True):
    mic = Mic(passes, commands)
    registry, b = create_fake_registry()
    heard: list[str] = []
    agent = legacy_voice.GraceVoiceAgent(
        registry=registry,
        tts=FakeTTS(),
        recognizer=mic,
        screen_reader=object(),
        follow_up_s=0,  # one command per wake word keeps scripts short
        on_heard=heard.append,
    )
    agent.is_running = True
    switch = MicSwitch(mic_on)
    worker = VoiceWorker(agent, mic, mic_on=switch)
    return worker, agent, mic, b, switch, heard


def test_split_wake_word_is_pure() -> None:
    assert split_wake_word("Listen, open notepad", "listen") == "open notepad"
    assert split_wake_word("listening to music", "listen") is None


def test_wake_word_then_command() -> None:
    worker, _, mic, b, _, heard = setup(["random chatter", "listen volume 30"])
    worker.step()
    assert b.volume.calls == []  # no wake word: nothing happens
    worker.step()
    assert b.volume.level == 30 and heard == ["volume 30"]
    assert mic.log == ["wake", "wake"]


def test_mic_off_opens_nothing_and_waits() -> None:
    worker, _, mic, _, _, _ = setup(["listen volume 30"], mic_on=False)
    worker.wake()  # pre-set so the wait returns at once
    worker.step()
    assert mic.log == []


def test_mic_turned_off_during_a_pass_drops_the_result() -> None:
    worker, _, mic, b, switch, heard = setup(["listen volume 30"])

    def turn_off() -> None:
        switch.on = False
        worker.cancel()

    mic.during = turn_off
    worker.step()
    assert b.volume.calls == [] and heard == []


def test_cancel_during_a_conversation_ends_it_and_drops_the_text() -> None:
    worker, agent, mic, b, _, heard = setup(["listen"], commands=["volume 30", "volume 40"])
    agent.follow_up_s = 10
    mic.during = None
    # The wake word alone opens a conversation; cancel while the first command is captured.
    original = mic.listen_for_command

    def cancel_then_listen(phrase_time_limit: int = 5) -> str:
        worker.cancel()
        return original(phrase_time_limit)

    mic.listen_for_command = cancel_then_listen  # type: ignore[method-assign]
    worker.step()
    assert b.volume.calls == [] and heard == []
    assert not agent._session_open


def test_push_to_talk_works_with_the_mic_off() -> None:
    worker, _, mic, b, _, heard = setup([], commands=["volume 25"], mic_on=False)
    worker.request_ptt()
    worker.step()
    assert b.volume.level == 25 and heard == ["volume 25"]
    assert mic.log == ["command"]  # no wake-word pass, no wake word needed


def test_push_to_talk_pressed_during_a_wake_pass_uses_what_was_heard() -> None:
    worker, _, mic, b, _, _ = setup(["volume 15"])  # no wake word said
    mic.during = worker.request_ptt
    worker.step()
    assert b.volume.level == 15
    assert mic.log == ["wake"]  # no second listen


def test_push_to_talk_strips_a_wake_word_if_said_anyway() -> None:
    worker, _, mic, b, _, _ = setup(["listen, volume 35"])
    mic.during = worker.request_ptt
    worker.step()
    assert b.volume.level == 35


def test_push_to_talk_silence_does_nothing() -> None:
    worker, _, _, b, _, heard = setup([], commands=[""])
    worker.request_ptt()
    worker.step()
    assert heard == [] and b.volume.calls == []


def test_cancel_clears_a_pending_push_to_talk() -> None:
    worker, _, mic, _, _, _ = setup([], commands=["volume 25"], mic_on=False)
    worker.request_ptt()
    worker.cancel()
    worker.wake()
    worker.step()
    assert mic.log == []


def test_push_to_talk_during_a_conversation_is_ignored() -> None:
    worker, agent, _, _, _, _ = setup([])
    agent._session_open = True
    worker.request_ptt()
    assert not worker._take_ptt()


@pytest.mark.timeout(5)
def test_thread_runs_and_stops() -> None:
    worker, agent, _, b, switch, heard = setup(["listen volume 30"])
    worker.start()
    tick = threading.Event()
    for _ in range(200):
        if b.volume.level == 30:
            break
        tick.wait(0.01)
    switch.on = False
    worker.stop()
    assert b.volume.level == 30 and heard == ["volume 30"]
    assert agent.is_running is False


def test_a_crashing_pass_does_not_kill_the_worker() -> None:
    worker, _, mic, _, switch, _ = setup([])
    calls = []

    def boom() -> str | None:
        calls.append(1)
        switch.on = False
        worker._stop.set()
        raise OSError("mic unplugged")

    mic.listen_for_wake_phrase = boom  # type: ignore[method-assign]
    worker._run()  # returns once stop is set; the exception was logged, not raised
    assert calls == [1]
