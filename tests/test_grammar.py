"""M2: deterministic whole-phrase grammar. Pure tests: no hardware, no OS calls."""

from __future__ import annotations

import pytest

from grace.actions import create_fake_registry
from grace.actions.input_ctl import ALLOWED_KEYS, HotkeyArgs
from grace.intent.grammar import (
    RULES,
    Grammar,
    GrammarError,
    compile_template,
    default_grammar,
    normalize,
    parse,
    words_to_digits,
)
from grace.intent.types import APP_INTENTS

RISKY = {"system.shutdown", "system.restart", "system.sleep"}

POSITIVE: list[tuple[str, str, dict]] = [
    # volume: absolute
    ("volume 40", "volume.set", {"level": 40}),
    ("Volume 40%", "volume.set", {"level": 40}),
    ("set volume to 40", "volume.set", {"level": 40}),
    ("set the volume to 70 percent", "volume.set", {"level": 70}),
    ("Set the volume to forty five percent.", "volume.set", {"level": 45}),
    ("change the sound to 30", "volume.set", {"level": 30}),
    ("volume at 25", "volume.set", {"level": 25}),
    ("turn the volume up to 80", "volume.set", {"level": 80}),
    ("set volume to max", "volume.set", {"level": 100}),
    ("volume zero", "volume.set", {"level": 0}),
    ("volume a hundred", "volume.set", {"level": 100}),
    ("volume 150", "volume.set", {"level": 150}),  # clamped by the action, see test_registry
    ("please set volume to 20", "volume.set", {"level": 20}),
    # volume: relative
    ("volume up", "volume.step", {"delta": 10}),
    ("volume down", "volume.step", {"delta": -10}),
    ("volume up 20", "volume.step", {"delta": 20}),
    ("volume down by 15 percent", "volume.step", {"delta": -15}),
    ("turn the volume up", "volume.step", {"delta": 10}),
    ("turn up the volume a lot", "volume.step", {"delta": 30}),
    ("turn down the sound a little", "volume.step", {"delta": -5}),
    ("increase volume", "volume.step", {"delta": 10}),
    ("increase the volume by twenty", "volume.step", {"delta": 20}),
    ("lower the volume", "volume.step", {"delta": -10}),
    ("louder", "volume.step", {"delta": 10}),
    ("a bit quieter", "volume.step", {"delta": -5}),
    # volume: mute / query
    ("mute", "volume.mute", {"mute": True}),
    ("mute the sound", "volume.mute", {"mute": True}),
    ("unmute", "volume.mute", {"mute": False}),
    ("turn off the sound", "volume.mute", {"mute": True}),
    ("what's the volume", "volume.get", {}),
    ("what is the current volume", "volume.get", {}),
    # brightness
    ("brightness 70", "brightness.set", {"level": 70}),
    ("set brightness to 70 percent", "brightness.set", {"level": 70}),
    ("set the screen brightness to 35", "brightness.set", {"level": 35}),
    ("brightness to half", "brightness.set", {"level": 50}),
    ("brightness up", "brightness.step", {"delta": 10}),
    ("brightness down a lot", "brightness.step", {"delta": -30}),
    ("increase brightness by 5", "brightness.step", {"delta": 5}),
    ("dim the screen", "brightness.step", {"delta": -10}),
    ("make the screen a bit brighter", "brightness.step", {"delta": 5}),
    ("darker", "brightness.step", {"delta": -10}),
    ("what is the brightness", "brightness.get", {}),
    # risky: exact phrases
    ("shut down the computer", "system.shutdown", {}),
    ("Shut down the computer.", "system.shutdown", {}),
    ("shutdown my pc", "system.shutdown", {}),
    ("turn off the computer", "system.shutdown", {}),
    ("power off the laptop", "system.shutdown", {}),
    ("restart the computer", "system.restart", {}),
    ("reboot my pc", "system.restart", {}),
    ("put the computer to sleep", "system.sleep", {}),
    ("sleep the computer", "system.sleep", {}),
    # mouse
    ("click", "input.click", {"button": "left", "count": 1}),
    ("left click", "input.click", {"button": "left", "count": 1}),
    ("click here", "input.click", {"button": "left", "count": 1}),
    ("double click", "input.click", {"button": "left", "count": 2}),
    ("double-click", "input.click", {"button": "left", "count": 2}),
    ("right click", "input.click", {"button": "right", "count": 1}),
    ("scroll", "input.scroll", {"direction": "down", "notches": 3}),
    ("scroll down", "input.scroll", {"direction": "down", "notches": 3}),
    ("scroll up", "input.scroll", {"direction": "up", "notches": 3}),
    ("scroll down a lot", "input.scroll", {"direction": "down", "notches": 10}),
    ("scroll up a little", "input.scroll", {"direction": "up", "notches": 1}),
    ("scroll down 5 times", "input.scroll", {"direction": "down", "notches": 5}),
    # keyboard
    ("press enter", "input.hotkey", {"keys": ["enter"]}),
    ("hit the enter key", "input.hotkey", {"keys": ["enter"]}),
    ("press control c", "input.hotkey", {"keys": ["ctrl", "c"]}),
    ("press ctrl+shift+escape", "input.hotkey", {"keys": ["ctrl", "shift", "esc"]}),
    ("press alt f four", "input.hotkey", {"keys": ["alt", "f4"]}),
    ("press windows key plus d", "input.hotkey", {"keys": ["win", "d"]}),
    ("press page down", "input.hotkey", {"keys": ["pagedown"]}),
    ("undo", "input.hotkey", {"keys": ["ctrl", "z"]}),
    ("select all", "input.hotkey", {"keys": ["ctrl", "a"]}),
    ("save the file", "input.hotkey", {"keys": ["ctrl", "s"]}),
    ("cut that", "input.hotkey", {"keys": ["ctrl", "x"]}),
    ("show desktop", "input.hotkey", {"keys": ["win", "d"]}),
    ("open a new tab", "input.hotkey", {"keys": ["ctrl", "t"]}),
    ("open the start menu", "input.hotkey", {"keys": ["win"]}),
    ("minimize the window", "input.hotkey", {"keys": ["win", "down"]}),
    # typing keeps original casing and punctuation
    ("type Hello, World!", "input.type", {"text": "Hello, World!"}),
    ("please type my name is Grace", "input.type", {"text": "my name is Grace"}),
    # clipboard
    ("copy", "clipboard.copy", {}),
    ("copy the selection", "clipboard.copy", {}),
    ("paste", "clipboard.paste", {}),
    # apps and windows
    ("open notepad", "apps.open", {"name": "notepad"}),
    ("open the calculator", "apps.open", {"name": "the calculator"}),
    ("launch google chrome", "apps.open", {"name": "google chrome"}),
    ("open youtube", "apps.open", {"name": "youtube"}),
    ("go to github dot com", "apps.open", {"name": "github dot com"}),
    ("open github.com", "apps.open", {"name": "github.com"}),
    ("close window", "apps.close_window", {}),
    ("close this window", "apps.close_window", {}),
    ("close the current tab", "apps.close_tab", {}),
    ("close tab", "apps.close_tab", {}),
    # screen
    ("take a screenshot", "screen.screenshot", {}),
    ("screenshot", "screen.screenshot", {}),
    ("capture the screen", "screen.screenshot", {}),
    # app-level
    ("yes", "app.confirm", {}),
    ("Yes.", "app.confirm", {}),
    ("confirm", "app.confirm", {}),
    ("no", "app.cancel", {}),
    ("cancel", "app.cancel", {}),
    ("help", "app.help", {}),
    ("start dictation", "app.dictation_start", {}),
    ("stop dictation", "app.dictation_stop", {}),
    ("read the screen", "app.read_screen", {}),
    ("read the selection", "app.read_selection", {}),
    ("analyze the code", "app.analyze_code", {}),
    ("debug this code", "app.debug_code", {}),
    ("exit", "app.exit", {}),
    ("stop listening", "app.end_session", {}),
    ("That's all.", "app.end_session", {}),
    ("start over", "app.reset_conversation", {}),
]

NEGATIVE: list[str] = [
    # required by the milestone: must NOT map to shutdown / close
    "turn off the light",
    "shutter speed",
    "close call",
    "I'm shutting the door",
    # incomplete or embedded risky phrases
    "shut down",
    "shutdown",
    "turn off",
    "restart",
    "reboot",
    "shut the door",
    "shut down the computer later",
    "please don't shut down the computer",
    "do not shut down the computer",
    "how do I shut down the computer",
    "I want to shut down the computer",
    "never shut down the computer",
    "turn off the computer screen",
    "restart the video",
    "restart the song",
    "the computer is shutting down",
    "go to sleep",
    "sleep",
    # "open" only claims apps / sites / domains it can open; the rest goes to the router
    "open the door",
    "run the tests",
    "start the computer",
    "open shutdown",
    # close: only "close [the|this] window/tab"
    "close",
    "close all windows",
    "close the door",
    "close enough",
    "that was a close one",
    # command words inside ordinary sentences
    "I said volume 40 yesterday",
    "the volume of a sphere",
    "what a loud noise",
    "click bait",
    "she will copy my homework",
    "scroll of the dead sea",
    "this is a brightness test",
    "mute point",
    "open",
    "type",
    "press",
    "press the big red thing",
    "",
    "   ",
    "...",
    "yes I am shutting the door",
]


@pytest.mark.parametrize(("text", "name", "args"), POSITIVE)
def test_positive(text: str, name: str, args: dict) -> None:
    intent = parse(text)
    assert intent is not None, f"{text!r} did not parse"
    assert (intent.name, intent.args) == (name, args)
    assert intent.source == "grammar"


@pytest.mark.parametrize("text", NEGATIVE)
def test_negative(text: str) -> None:
    assert parse(text) is None, f"{text!r} must not parse, got {parse(text)}"


def test_case_count_meets_gate() -> None:
    assert len(POSITIVE) + len(NEGATIVE) >= 60
    assert len(NEGATIVE) >= 20


@pytest.mark.parametrize("text", ["shut down the computer", "restart the computer"])
def test_risky_phrases_do_not_match_with_extra_words(text: str) -> None:
    assert parse(text) is not None
    for wrapped in (f"{text} now", f"not {text}", f"{text} tomorrow", f"maybe {text}"):
        intent = parse(wrapped)
        assert intent is None or intent.name not in RISKY, wrapped


def test_every_grammar_intent_exists() -> None:
    registry, _ = create_fake_registry()
    g = default_grammar()
    names = {r.intent for r in RULES} | {name for name, _ in g.phrases} | {"input.type"}
    unknown = {n for n in names if n not in registry and n not in APP_INTENTS}
    assert not unknown


def test_risky_intents_match_risky_registry_actions() -> None:
    registry, _ = create_fake_registry()
    assert {s.name for s in registry if s.risky} == RISKY


def test_shortcut_and_key_tables_use_valid_keys() -> None:
    g = default_grammar()
    for keys, _ in g.shortcuts:
        assert HotkeyArgs(keys=list(keys)).keys == list(keys)
    for _, canon in g._key_phrases:
        assert canon in ALLOWED_KEYS


def test_normalize() -> None:
    assert normalize("Set the Volume to 40%!") == "set the volume to 40 percent"
    assert normalize("What's up?") == "what is up"
    assert normalize("open GitHub.com.") == "open github.com"
    assert normalize("please, volume up please", ["please"], ["please"]) == "volume up"
    assert normalize("“double-click”") == "double click"


def test_words_to_digits() -> None:
    assert words_to_digits("forty five percent".split()) == ["45", "percent"]
    assert words_to_digits("a hundred".split()) == ["100"]
    assert words_to_digits("twenty".split()) == ["20"]
    assert words_to_digits("ninety nine".split()) == ["99"]
    assert words_to_digits("a dog".split()) == ["a", "dog"]


def test_parse_keys_rejects_unknown_and_repeated_keys() -> None:
    g = default_grammar()
    assert g.parse_keys("control alt delete") == ["ctrl", "alt", "delete"]
    assert g.parse_keys("f 13") is None
    assert g.parse_keys("control control") is None
    assert g.parse_keys("banana") is None
    assert g.parse_keys("a b c d e") is None  # more than 4 keys


def test_templates_reject_unknown_groups_and_bad_tokens() -> None:
    with pytest.raises(GrammarError):
        compile_template("{nope}", {})
    with pytest.raises(GrammarError):
        compile_template("open (.*)", {})
    with pytest.raises(GrammarError):
        compile_template("", {})


def test_yaml_booleans_are_rejected(tmp_path) -> None:
    bad = tmp_path / "g.yaml"
    bad.write_text(
        "groups: {}\n"
        "steps: {small: 1, normal: 2, large: 3}\n"
        "scroll: {small: 1, normal: 2, large: 3}\n"
        "keys: {}\nphrases:\n  app.confirm: [yes]\n"
    )
    with pytest.raises(GrammarError, match="quote yes/no"):
        Grammar.from_yaml(bad)
