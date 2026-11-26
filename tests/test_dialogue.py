"""Dialogue memory and deterministic follow-up resolution. Pure, no hardware."""

from __future__ import annotations

import pytest

from grace.intent.dialogue import Dialogue, LastAction, is_undo_phrase, resolve_followup
from grace.intent.grammar import default_grammar
from grace.intent.types import Intent

G = default_grammar()


def last(action: str, before: dict | None = None, **args) -> LastAction:
    return LastAction(Intent(action, args), before or {})


def resolve(text: str, prev: LastAction | None):
    return resolve_followup(
        G.normalize(text),
        prev,
        is_risky=lambda n: n.startswith("system."),
        steps=G.steps,
        scroll=G.scroll,
    )


VOL_SET = last("volume.set", {"level": 70}, level=40)
VOL_UP = last("volume.step", delta=10)
VOL_DOWN_SMALL = last("volume.step", delta=-5)
BRIGHT_SET = last("brightness.set", {"level": 30}, level=80)
SCROLL_DOWN = last("input.scroll", direction="down", notches=3)
MUTE = last("volume.mute", mute=True)
OPEN = last("apps.open", name="notepad")
VOL_GET = last("volume.get")


@pytest.mark.parametrize(
    ("text", "prev", "name", "args"),
    [
        # "more" keeps the direction; the size word sets the amount, else the last amount
        ("a bit more", VOL_SET, "volume.step", {"delta": 5}),
        ("more", VOL_UP, "volume.step", {"delta": 10}),
        ("a lot more", VOL_UP, "volume.step", {"delta": 30}),
        ("Okay, a bit more please.", VOL_DOWN_SMALL, "volume.step", {"delta": -5}),
        ("even more", VOL_DOWN_SMALL, "volume.step", {"delta": -5}),
        ("more", SCROLL_DOWN, "input.scroll", {"direction": "down", "notches": 3}),
        ("keep going", SCROLL_DOWN, "input.scroll", {"direction": "down", "notches": 3}),
        # "less" reverses it, by a small step unless a size is said
        ("a bit less", VOL_UP, "volume.step", {"delta": -5}),
        ("less", BRIGHT_SET, "brightness.step", {"delta": -5}),
        ("a lot less", VOL_UP, "volume.step", {"delta": -30}),
        ("too much", SCROLL_DOWN, "input.scroll", {"direction": "up", "notches": 1}),
        # direction words apply to the last target
        ("turn it up", BRIGHT_SET, "brightness.step", {"delta": 10}),
        ("down a bit", VOL_SET, "volume.step", {"delta": -5}),
        ("higher", VOL_GET, "volume.step", {"delta": 10}),
        ("up", SCROLL_DOWN, "input.scroll", {"direction": "up", "notches": 3}),
        # undo reverses the change (a set goes back to the value it replaced)
        ("undo that", VOL_SET, "volume.set", {"level": 70}),
        ("undo that", VOL_UP, "volume.step", {"delta": -10}),
        ("put it back", BRIGHT_SET, "brightness.set", {"level": 30}),
        ("revert that", MUTE, "volume.mute", {"mute": False}),
        ("undo it", SCROLL_DOWN, "input.scroll", {"direction": "up", "notches": 3}),
        # transfer to the other target with the same arguments
        ("now the brightness", VOL_SET, "brightness.set", {"level": 40}),
        ("and the volume", BRIGHT_SET, "volume.set", {"level": 80}),
        ("brightness too", VOL_UP, "brightness.step", {"delta": 10}),
        ("same for the brightness", VOL_DOWN_SMALL, "brightness.step", {"delta": -5}),
        ("what about the brightness", VOL_GET, "brightness.get", {}),
        # a bare level sets the last target
        ("make it 30", VOL_UP, "volume.set", {"level": 30}),
        ("thirty percent", BRIGHT_SET, "brightness.set", {"level": 30}),
        ("set it to 75", VOL_GET, "volume.set", {"level": 75}),
        # repeat
        ("again", OPEN, "apps.open", {"name": "notepad"}),
        ("do it again", VOL_UP, "volume.step", {"delta": 10}),
        ("one more time", SCROLL_DOWN, "input.scroll", {"direction": "down", "notches": 3}),
    ],
)
def test_followups_resolve(text: str, prev: LastAction, name: str, args: dict) -> None:
    intent = resolve(text, prev)
    assert intent is not None, text
    assert (intent.name, intent.args, intent.source) == (name, args, "followup")


@pytest.mark.parametrize(
    ("text", "prev"),
    [
        ("a bit more", None),  # nothing to follow up
        ("a bit more", OPEN),  # "more" of opening an app means nothing
        ("a bit more", MUTE),  # mute has no direction
        ("a bit more", VOL_GET),  # neither does reading a value
        ("now the volume", VOL_SET),  # already the volume
        ("now the brightness", SCROLL_DOWN),
        ("now the brightness", MUTE),
        ("make it 300", VOL_SET),
        ("make it 5", SCROLL_DOWN),
        ("undo that", OPEN),  # not reversible: the grammar's Ctrl+Z handles it
        ("undo that", last("volume.set", level=40)),  # the old value is unknown
        ("again", last("system.shutdown")),  # risky actions are never repeated
        ("undo that", last("system.restart")),
        ("brightness", VOL_SET),  # a bare target is a question for the router, not a transfer
        ("I want a bit more space on my desk", VOL_SET),  # whole phrase only
        ("more coffee", VOL_UP),
        ("go back", VOL_SET),  # the grammar's browser back, never an undo
    ],
)
def test_non_followups_return_none(text: str, prev: LastAction | None) -> None:
    assert resolve(text, prev) is None


def test_undo_phrases() -> None:
    assert is_undo_phrase("undo that") and is_undo_phrase("put it back")
    assert not is_undo_phrase("undo") and not is_undo_phrase("undo my homework")


# --------------------------------------------------------------------------- memory
class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_keeps_the_last_n_exchanges_as_messages() -> None:
    d = Dialogue(max_turns=2, clock=Clock())
    for i in range(3):
        d.add(f"q{i}", f"a{i}")
    d.add("q3", "")  # GRACE said nothing: only the user message
    assert d.messages() == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
    ]


def test_forgets_after_a_long_pause_and_on_reset() -> None:
    clock = Clock()
    d = Dialogue(memory_reset_s=600, followup_s=120, clock=clock)
    d.add("hello", "hi")
    d.record(Intent("volume.set", {"level": 40}))
    clock.now = 119
    assert d.last() is not None
    clock.now = 121
    assert d.last() is None  # follow-ups refer to recent actions only
    assert d.history()
    clock.now = 700
    assert d.history() == []
    d.add("again", "")
    d.record(Intent("volume.set", {"level": 40}))
    d.reset()
    assert d.history() == [] and d.last() is None


def test_the_last_action_is_forgotten_once_something_else_was_said() -> None:
    d = Dialogue(clock=Clock())
    d.record(Intent("volume.set", {"level": 40}))
    assert d.last() is not None  # during the exchange that ran it
    d.add("volume 40", "Volume set to 40 percent.")
    assert d.last() is not None  # the next utterance may follow it up
    d.add("what time is it", "Two o'clock.")
    assert d.last() is None  # stale: "a bit more" now is not about the volume


def test_zero_turns_keeps_nothing() -> None:
    d = Dialogue(max_turns=0, clock=Clock())
    d.add("hello", "hi")
    assert d.messages() == []
