"""input/hotkeys.py with a fake keyboard: registration from settings, latch, push-to-talk."""

from __future__ import annotations

import pytest

from grace.config import HotkeySettings
from grace.input.hotkeys import (
    HOTKEY_NAMES,
    PTT_RELEASE,
    STALE_S,
    FakeKeyboard,
    HotkeyManager,
)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make(settings: HotkeySettings | None = None, reject: tuple[str, ...] = ()):
    kb = FakeKeyboard(reject=reject)
    clock = Clock()
    calls: list[tuple[str, float]] = []
    handlers = {
        name: (lambda t, name=name: calls.append((name, t)))
        for name in (*HOTKEY_NAMES, PTT_RELEASE)
    }
    manager = HotkeyManager(settings or HotkeySettings(), handlers, kb, clock=clock).start()
    return manager, kb, clock, calls


def names(calls: list[tuple[str, float]]) -> list[str]:
    return [n for n, _ in calls]


def test_registers_every_hotkey_from_settings() -> None:
    _, kb, _, _ = make()
    assert set(kb.hotkeys) == {
        "ctrl+alt+m",
        "ctrl+alt+g",
        "ctrl+alt+space",
        "ctrl+alt+q",
        "ctrl+alt+y",
        "ctrl+alt+h",
        "ctrl+alt+x",
    }


def test_combos_come_from_settings_yaml_values() -> None:
    custom = HotkeySettings(mic_toggle="Ctrl+Shift+F9", panic_stop="ctrl+alt+p")
    manager, kb, _, calls = make(custom)
    assert "ctrl+shift+f9" in kb.hotkeys and "ctrl+alt+m" not in kb.hotkeys
    kb.press("ctrl+shift+f9")
    kb.press("ctrl+alt+p")
    assert names(calls) == ["mic_toggle", "panic_stop"]
    assert manager.combos["mic_toggle"] == "ctrl+shift+f9"


@pytest.mark.parametrize(
    ("combo", "name"),
    [
        ("ctrl+alt+m", "mic_toggle"),
        ("ctrl+alt+g", "gesture_toggle"),
        ("ctrl+alt+q", "panic_stop"),
        ("ctrl+alt+y", "confirm"),
        ("ctrl+alt+h", "history"),
        ("ctrl+alt+x", "mark_wrong"),
        ("ctrl+alt+space", "push_to_talk"),
    ],
)
def test_each_hotkey_calls_its_handler_with_the_press_time(combo: str, name: str) -> None:
    _, kb, clock, calls = make()
    clock.now = 123.25
    kb.press(combo)
    assert calls == [(name, 123.25)]


def test_auto_repeat_while_held_fires_once() -> None:
    _, kb, clock, calls = make()
    for _ in range(30):  # one second of Windows auto-repeat
        kb.press("ctrl+alt+m")
        clock.now += 0.033
    assert names(calls) == ["mic_toggle"]
    kb.release("m")
    kb.press("ctrl+alt+m")
    assert names(calls) == ["mic_toggle", "mic_toggle"]


def test_releasing_any_key_of_the_combo_unlatches() -> None:
    _, kb, _, calls = make()
    kb.press("ctrl+alt+g")
    kb.release("ctrl")
    kb.press("ctrl+alt+g")
    kb.release("x")  # unrelated key: still unlatched only by its own keys
    assert names(calls) == ["gesture_toggle", "gesture_toggle"]


def test_lost_key_up_expires_after_quiet_period() -> None:
    _, kb, clock, calls = make()
    kb.press("ctrl+alt+h")
    clock.now += STALE_S + 0.01  # no repeats: the key was released but we never saw it
    kb.press("ctrl+alt+h")
    assert names(calls) == ["history", "history"]


def test_push_to_talk_press_and_release() -> None:
    _, kb, clock, calls = make()
    kb.press("ctrl+alt+space")
    for _ in range(10):
        clock.now += 0.03
        kb.press("ctrl+alt+space")  # held: repeats
    clock.now += 0.03
    kb.release("space")
    assert names(calls) == ["push_to_talk", PTT_RELEASE]
    assert calls[1][1] == pytest.approx(100.33)
    kb.release("space")  # a second key-up is ignored
    assert len(calls) == 2


def test_push_to_talk_lost_release_is_recovered() -> None:
    manager, kb, clock, calls = make()
    kb.press("ctrl+alt+space")
    clock.now += STALE_S + 0.5
    manager.expire_stale()
    assert names(calls) == ["push_to_talk", PTT_RELEASE]
    # And a new press after a lost release ends the old hold first.
    kb.press("ctrl+alt+space")
    clock.now += STALE_S + 0.5
    kb.press("ctrl+alt+space")
    assert names(calls)[2:] == ["push_to_talk", PTT_RELEASE, "push_to_talk"]


def test_expire_stale_keeps_fresh_latches() -> None:
    manager, kb, clock, calls = make()
    kb.press("ctrl+alt+space")
    clock.now += STALE_S / 2
    manager.expire_stale()
    kb.press("ctrl+alt+space")  # repeat, still latched
    assert names(calls) == ["push_to_talk"]


def test_stop_unregisters_and_releases_a_held_push_to_talk() -> None:
    manager, kb, _, calls = make()
    kb.press("ctrl+alt+space")
    manager.stop()
    assert kb.hotkeys == {} and kb.key_up is None
    assert names(calls) == ["push_to_talk", PTT_RELEASE]


def test_a_bad_combo_is_reported_and_the_rest_still_work() -> None:
    manager, kb, _, calls = make(reject=("ctrl+alt+x",))
    assert set(manager.failed) == {"mark_wrong"}
    kb.press("ctrl+alt+m")
    assert names(calls) == ["mic_toggle"]


def test_a_crashing_handler_never_reaches_the_hook(caplog: pytest.LogCaptureFixture) -> None:
    kb = FakeKeyboard()
    handlers = {name: (lambda t: None) for name in (*HOTKEY_NAMES, PTT_RELEASE)}

    def boom(t: float) -> None:
        raise RuntimeError("bad")

    handlers["confirm"] = boom
    HotkeyManager(HotkeySettings(), handlers, kb).start()
    kb.press("ctrl+alt+y")  # must not raise
    assert "hotkey handler confirm failed" in caplog.text


def test_missing_handlers_are_rejected() -> None:
    with pytest.raises(ValueError, match="mark_wrong"):
        HotkeyManager(HotkeySettings(), {n: print for n in HOTKEY_NAMES[:-1]}, FakeKeyboard())


@pytest.mark.windows
def test_keyboard_backend_resolves_scan_codes() -> None:
    from grace.input.hotkeys import KeyboardLibBackend

    backend = KeyboardLibBackend()
    codes = backend.scan_codes("ctrl+alt+space")
    assert 57 in codes  # space
    assert len(codes) >= 3
