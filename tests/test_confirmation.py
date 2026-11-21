"""M2: risky intents need the exact phrase AND confirmation; 8 s timeout cancels."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from grace.actions import create_fake_registry
from grace.actions.confirmation import ConfirmationGate, Dispatcher, format_hotkey
from grace.intent.grammar import parse
from grace.intent.types import Intent


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@dataclass
class FakeTimer:
    delay: float
    callback: object
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


@dataclass
class Harness:
    registry: object
    backends: object
    gate: ConfirmationGate
    dispatcher: Dispatcher
    clock: Clock
    said: list[str] = field(default_factory=list)
    timers: list[FakeTimer] = field(default_factory=list)

    def say_text(self, text: str):
        intent = parse(text)
        assert intent is not None, text
        return self.dispatcher.handle(intent)


@pytest.fixture
def h() -> Harness:
    registry, backends = create_fake_registry()
    said: list[str] = []
    timers: list[FakeTimer] = []
    clock = Clock()

    def start_timer(delay, cb):
        timers.append(FakeTimer(delay, cb))
        return timers[-1]

    gate = ConfirmationGate(
        registry, say=said.append, clock=clock, start_timer=start_timer, confirm_hotkey="ctrl+alt+y"
    )
    return Harness(
        registry, backends, gate, Dispatcher(registry, gate, said.append), clock, said, timers
    )


def test_prompt_text_and_no_execution(h: Harness) -> None:
    assert h.say_text("shut down the computer") is None
    assert h.said == ["Confirm shutdown? Say yes or press Ctrl+Alt+Y."]
    assert h.backends.power.calls == []
    assert h.gate.pending is not None and h.gate.pending.name == "system.shutdown"
    assert h.timers[0].delay == 8.0


def test_voice_yes_confirms_once(h: Harness) -> None:
    h.say_text("restart the computer")
    result = h.say_text("yes")
    assert result is not None and result.ok
    assert h.backends.power.calls == ["restart"]
    assert h.timers[0].cancelled
    h.say_text("yes")  # nothing pending any more
    assert h.backends.power.calls == ["restart"]
    assert h.said[-1] == "There's nothing to confirm."


def test_hotkey_confirm_path(h: Harness) -> None:
    h.say_text("put the computer to sleep")
    assert h.gate.confirm().ok  # what Ctrl+Alt+Y calls
    assert h.backends.power.calls == ["sleep"]


def test_timeout_cancels(h: Harness) -> None:
    h.say_text("shut down the computer")
    h.clock.now += 8.0
    h.timers[0].callback()  # the 8 s timer fires
    assert h.gate.pending is None
    assert h.said[-1] == "Cancelled: shutdown was not confirmed in time."
    assert h.gate.confirm() is None
    assert h.backends.power.calls == []


def test_late_yes_after_deadline_is_refused_even_if_timer_did_not_fire(h: Harness) -> None:
    h.say_text("shut down the computer")
    h.clock.now += 8.01
    h.say_text("yes")
    assert h.backends.power.calls == []
    assert "not confirmed in time" in h.said[-1]


def test_yes_just_before_deadline_works(h: Harness) -> None:
    h.say_text("shut down the computer")
    h.clock.now += 7.9
    h.say_text("yes")
    assert h.backends.power.calls == ["shutdown"]


def test_no_cancels(h: Harness) -> None:
    h.say_text("shut down the computer")
    h.say_text("no")
    assert h.said[-1] == "Cancelled shutdown."
    h.say_text("yes")
    assert h.backends.power.calls == []


def test_any_other_command_cancels_pending(h: Harness) -> None:
    h.say_text("shut down the computer")
    h.say_text("volume 30")
    assert h.gate.pending is None
    assert h.backends.volume.level == 30
    h.say_text("yes")
    assert h.backends.power.calls == []


def test_new_risky_request_replaces_old(h: Harness) -> None:
    h.say_text("shut down the computer")
    h.say_text("restart the computer")
    h.timers[0].callback()  # stale timer from the first request must not cancel the second
    assert h.gate.pending is not None and h.gate.pending.name == "system.restart"
    h.say_text("yes")
    assert h.backends.power.calls == ["restart"]


def test_negative_phrases_never_reach_the_gate(h: Harness) -> None:
    for text in ["turn off the light", "I'm shutting the door", "shut down", "close call"]:
        assert parse(text) is None
    assert h.gate.pending is None and h.backends.power.calls == []


def test_dispatcher_runs_safe_actions_directly(h: Harness) -> None:
    result = h.say_text("set volume to 150 percent")
    assert result.ok and h.backends.volume.level == 100
    assert h.said == ["Volume set to 100 percent."]


def test_dispatcher_unknown_intent(h: Harness) -> None:
    result = h.dispatcher.handle(Intent("nothing.here"))
    assert result is not None and not result.ok


def test_expire_without_token_checks_the_clock(h: Harness) -> None:
    h.say_text("shut down the computer")
    assert not h.gate.expire()
    h.clock.now += 9
    assert h.gate.expire()


def test_format_hotkey() -> None:
    assert format_hotkey("ctrl+alt+y") == "Ctrl+Alt+Y"


def test_real_thread_timer_cancels() -> None:
    registry, backends = create_fake_registry()
    said: list[str] = []
    gate = ConfirmationGate(registry, say=said.append, timeout_s=0.05)
    gate.request(Intent("system.shutdown"))
    import time

    deadline = time.monotonic() + 2
    while gate.pending is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert gate.pending is None and backends.power.calls == []
    assert said[-1] == "Cancelled: shutdown was not confirmed in time."
