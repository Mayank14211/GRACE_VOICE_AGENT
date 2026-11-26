"""M2: action registry and OS actions, exercised through in-memory fakes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from grace.actions import create_fake_registry
from grace.actions.apps import UrlArgs, resolve_target
from grace.actions.fakes import FakeBackends
from grace.actions.input_ctl import WHEEL_DELTA, HotkeyArgs, ScrollArgs
from grace.actions.registry import (
    ActionError,
    ActionRegistry,
    ActionResult,
    ActionSpec,
    ConfirmationRequiredError,
    NoArgs,
)
from grace.actions.system import LevelArgs, PycawVolume, SbcBrightness, StepArgs, WindowsPower


@pytest.fixture
def reg(tmp_path: Path):
    return create_fake_registry(screenshot_dir=tmp_path / "shots")


# --------------------------------------------------------------------------- registry core
def test_register_and_lookup() -> None:
    r = ActionRegistry()

    @r.register("demo.hello", description="Say hi.")
    def hello(_: NoArgs) -> ActionResult:
        return ActionResult(True, "hi")

    assert "demo.hello" in r and r.names() == ["demo.hello"]
    assert r.spec("demo.hello").description == "Say hi."
    assert r.execute("demo.hello").message == "hi"
    with pytest.raises(ValueError):
        r.register("demo.hello")(hello)


def test_unknown_action_and_invalid_args_do_not_raise(reg) -> None:
    registry, b = reg
    assert not registry.execute("nope.nothing").ok
    result = registry.execute("volume.set", {"level": "loud"})
    assert not result.ok and "level" in result.message
    assert not registry.execute("volume.set", {"level": 10, "extra": 1}).ok  # extra=forbid
    assert b.volume.calls == []


def test_handler_failures_become_results() -> None:
    r = ActionRegistry()

    @r.register("x.expected")
    def expected(_: NoArgs) -> ActionResult:
        raise ActionError("Nope, not today.")

    @r.register("x.crash")
    def crash(_: NoArgs) -> ActionResult:
        raise RuntimeError("boom")

    assert r.execute("x.expected") == ActionResult(False, "Nope, not today.")
    assert r.execute("x.crash") == ActionResult(False, "Sorry, that didn't work.")


def test_risky_spec_needs_a_confirm_label() -> None:
    with pytest.raises(ValueError):
        ActionSpec("x.risky", NoArgs, lambda a: ActionResult(True, ""), risky=True)


@pytest.mark.parametrize("name", ["system.shutdown", "system.restart", "system.sleep"])
def test_risky_actions_never_execute_without_confirmation(reg, name: str) -> None:
    registry, b = reg
    assert registry.is_risky(name)
    with pytest.raises(ConfirmationRequiredError):
        registry.execute(name)
    with pytest.raises(ConfirmationRequiredError):
        registry.execute(name, {}, confirmed=False)
    assert b.power.calls == []
    assert registry.execute(name, confirmed=True).ok
    assert b.power.calls == [name.split(".")[1]]


def test_only_power_actions_are_risky(reg) -> None:
    registry, _ = reg
    assert {s.name for s in registry if s.risky} == {
        "system.shutdown",
        "system.restart",
        "system.sleep",
    }


def test_every_action_exports_a_json_schema(reg) -> None:
    registry, _ = reg
    for spec in registry:
        assert spec.args_model.model_json_schema()["type"] == "object"
        assert spec.description


# --------------------------------------------------------------------------- clamping
@pytest.mark.parametrize(("raw", "clamped"), [(150, 100), (100, 100), (40, 40), (0, 0), (-5, 0)])
def test_level_clamps(raw: int, clamped: int) -> None:
    assert LevelArgs(level=raw).level == clamped


def test_volume_150_sets_100(reg) -> None:
    registry, b = reg
    result = registry.execute("volume.set", {"level": 150})
    assert result.ok and b.volume.level == 100 and result.data["level"] == 100


def test_brightness_minus_5_sets_0(reg) -> None:
    registry, b = reg
    result = registry.execute("brightness.set", {"level": -5})
    assert result.ok and b.brightness.level == 0


def test_steps_clamp_at_the_ends(reg) -> None:
    registry, b = reg
    b.volume.level, b.brightness.level = 95, 3
    registry.execute("volume.step", {"delta": 30})
    registry.execute("brightness.step", {"delta": -10})
    assert (b.volume.level, b.brightness.level) == (100, 0)
    assert StepArgs(delta=500).delta == 100 and StepArgs(delta=-500).delta == -100


def test_scroll_notches_clamp() -> None:
    assert ScrollArgs(notches=0).notches == 1
    assert ScrollArgs(notches=999).notches == 50


# --------------------------------------------------------------------------- system actions
def test_volume_set_exact_and_unmutes(reg) -> None:
    registry, b = reg
    b.volume.muted = True
    assert registry.execute("volume.set", {"level": 37}).message == "Volume set to 37 percent."
    assert b.volume.level == 37 and not b.volume.muted


def test_volume_set_zero_keeps_mute_state(reg) -> None:
    registry, b = reg
    b.volume.muted = True
    registry.execute("volume.set", {"level": 0})
    assert b.volume.muted


def test_volume_get_mute_and_step(reg) -> None:
    registry, b = reg
    b.volume.level = 40
    assert registry.execute("volume.get").data == {"level": 40, "muted": False}
    registry.execute("volume.mute", {"mute": True})
    assert b.volume.muted
    registry.execute("volume.step", {"delta": -15})
    assert b.volume.level == 25


def test_brightness_get_and_step(reg) -> None:
    registry, b = reg
    b.brightness.level = 60
    assert registry.execute("brightness.get").data == {"level": 60}
    registry.execute("brightness.step", {"delta": 10})
    assert b.brightness.calls == [70]


# --------------------------------------------------------------------------- input actions
def test_click_and_scroll(reg) -> None:
    registry, b = reg
    assert registry.execute("input.click").message == "Click."
    assert registry.execute("input.click", {"button": "right", "count": 2}).message == (
        "Double right click."
    )
    registry.execute("input.scroll", {"direction": "up", "notches": 2})
    registry.execute("input.scroll", {"direction": "down"})
    assert b.input.calls == [
        ("click", "left", 1),
        ("click", "right", 2),
        ("scroll", 2 * WHEEL_DELTA),
        ("scroll", -3 * WHEEL_DELTA),
    ]


def test_pointer_actions_for_gestures(reg) -> None:
    registry, b = reg
    assert registry.execute("input.move_to", {"x": 10, "y": 20}).ok
    assert registry.execute("input.mouse_down").ok
    assert registry.execute("input.mouse_up", {"button": "left"}).ok
    assert not registry.execute("input.move_to", {"x": -1, "y": 0}).ok
    assert not registry.execute("input.mouse_down", {"button": "thumb"}).ok
    assert b.input.calls == [("move_to", 10, 20), ("mouse_down", "left"), ("mouse_up", "left")]
    assert registry.spec("input.move_to").quiet and not registry.spec("input.click").quiet


def test_quiet_actions_log_at_debug(reg, caplog: pytest.LogCaptureFixture) -> None:
    registry, _ = reg
    with caplog.at_level("DEBUG", logger="grace.actions.registry"):
        registry.execute("input.move_to", {"x": 1, "y": 1})
        registry.execute("input.click")
    levels = {r.getMessage().split()[1]: r.levelname for r in caplog.records}
    assert levels == {"input.move_to": "DEBUG", "input.click": "INFO"}


def test_pointer_primitives_are_not_llm_tools(reg) -> None:
    from grace.intent.tools import build_toolset

    offered = build_toolset(reg[0], extras=True, allow_risky=True).intents()
    assert not {"input.move_to", "input.mouse_down", "input.mouse_up"} & offered


def test_type_ascii_writes_once(reg) -> None:
    registry, b = reg
    registry.execute("input.type", {"text": "Hello, World!"})
    assert b.input.calls == [("write", "Hello, World!")]


def test_type_non_ascii_pastes_once_and_restores_clipboard(reg) -> None:
    registry, b = reg
    b.clipboard.text = "previous"
    registry.execute("input.type", {"text": "नमस्ते café"})
    assert b.input.calls == [("hotkey", "ctrl", "v")]  # not typed AND pasted (old bug)
    assert b.clipboard.history == ["नमस्ते café", "previous"]


def test_hotkey_validation(reg) -> None:
    registry, b = reg
    assert registry.execute("input.hotkey", {"keys": ["Control", "Escape"]}).ok
    assert b.input.calls == [("hotkey", "ctrl", "esc")]
    assert not registry.execute("input.hotkey", {"keys": ["ctrl", "rm -rf"]}).ok
    assert not registry.execute("input.hotkey", {"keys": []}).ok
    with pytest.raises(ValidationError):
        HotkeyArgs(keys=["a", "b", "c", "d", "e"])


def test_clipboard_copy_reports_selection(reg) -> None:
    registry, b = reg
    b.clipboard.text = ""
    assert not registry.execute("clipboard.copy").ok
    b.clipboard.text = "selected words"
    result = registry.execute("clipboard.copy")
    assert result.ok and result.data["text"] == "selected words"
    assert b.input.calls == [("hotkey", "ctrl", "c")] * 2


# --------------------------------------------------------------------------- apps / screen
@pytest.mark.parametrize(
    ("name", "target"),
    [
        ("notepad", ("app", "notepad")),
        ("the calculator", ("app", "calc")),
        ("Google Chrome", ("app", "chrome")),
        ("the chrome browser", ("app", "chrome")),
        ("youtube", ("url", "https://www.youtube.com")),
        ("github dot com", ("url", "https://github.com")),
        ("docs.python.org", ("url", "https://docs.python.org")),
        ("the door", None),
        ("notepad & del c:", None),
        ("calc; shutdown", None),
    ],
)
def test_resolve_target(name: str, target) -> None:
    assert resolve_target(name) == target


def test_open_app_site_and_unknown(reg) -> None:
    registry, b = reg
    assert registry.execute("apps.open", {"name": "notepad"}).ok
    assert registry.execute("apps.open", {"name": "youtube"}).ok
    result = registry.execute("apps.open", {"name": "the door"})
    assert not result.ok and "don't know" in result.message
    assert b.launcher.launched == ["notepad"]
    assert b.launcher.urls == ["https://www.youtube.com"]


@pytest.mark.parametrize("url", ["file:///C:/Windows", "javascript:alert(1)", "notaurl", "ftp://x"])
def test_open_url_rejects_non_http(url: str) -> None:
    with pytest.raises(ValidationError):
        UrlArgs(url=url)


def test_close_window_and_tab(reg) -> None:
    registry, b = reg
    registry.execute("apps.close_window")
    registry.execute("apps.close_tab")
    assert b.input.calls == [("hotkey", "alt", "f4"), ("hotkey", "ctrl", "w")]


def test_screenshot_saves_into_folder(tmp_path: Path) -> None:
    registry, b = create_fake_registry(screenshot_dir=tmp_path / "shots")
    result = registry.execute("screen.screenshot")
    assert result.ok
    [path] = b.screenshots.saved
    assert path.parent == tmp_path / "shots" and path.suffix == ".png"
    assert (tmp_path / "shots").is_dir()


# --------------------------------------------------------------------------- real backends
class FakeEndpoint:
    def __init__(self, events: list[str], scalar: float = 0.4, fail: bool = False) -> None:
        self.events, self.scalar, self.muted, self.fail = events, scalar, 0, fail

    def GetMasterVolumeLevelScalar(self) -> float:  # noqa: N802 (COM method names)
        if self.fail:
            raise OSError("device gone")
        return self.scalar

    def SetMasterVolumeLevelScalar(self, value: float, ctx: object) -> None:  # noqa: N802
        self.scalar = value

    def GetMute(self) -> int:  # noqa: N802
        return self.muted

    def SetMute(self, value: int, ctx: object) -> None:  # noqa: N802
        self.muted = value

    def __del__(self) -> None:
        self.events.append("release")


def _fake_com(events: list[str]):
    from contextlib import contextmanager

    @contextmanager
    def com():
        events.append("init")
        try:
            yield
        finally:
            events.append("uninit")

    return com


def test_pycaw_volume_converts_and_pairs_com_calls() -> None:
    events: list[str] = []
    endpoint = FakeEndpoint(events, scalar=0.404)
    vol = PycawVolume(endpoint_factory=lambda: endpoint, com=_fake_com(events))
    assert vol.get_level() == 40
    vol.set_level(73)
    assert endpoint.scalar == pytest.approx(0.73)
    vol.set_level(150)
    assert endpoint.scalar == 1.0
    vol.set_muted(True)
    assert vol.is_muted()
    assert events == ["init", "uninit"] * 5


def test_pycaw_volume_releases_endpoint_before_uninit_on_error() -> None:
    events: list[str] = []
    vol = PycawVolume(
        endpoint_factory=lambda: FakeEndpoint(events, fail=True), com=_fake_com(events)
    )
    with pytest.raises(ActionError, match="device gone"):
        vol.get_level()
    assert events == ["init", "release", "uninit"]


def test_com_initialized_pairs_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    from grace.actions import system

    calls: list[str] = []
    fake = SimpleNamespace(
        CoInitialize=lambda: calls.append("init"), CoUninitialize=lambda: calls.append("uninit")
    )
    monkeypatch.setitem(sys.modules, "comtypes", fake)
    with system.com_initialized():
        calls.append("work")
    assert calls == ["init", "work", "uninit"]

    def changed_mode() -> None:
        err = OSError("changed mode")
        err.winerror = system.RPC_E_CHANGED_MODE
        raise err

    calls.clear()
    fake.CoInitialize = changed_mode
    with system.com_initialized():
        calls.append("work")
    assert calls == ["work"]  # someone else owns this thread's COM; don't uninitialise it


def test_sbc_brightness(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"level": 55}
    fake = SimpleNamespace(
        get_brightness=lambda display=None: [state["level"]],
        set_brightness=lambda value, display=None: state.update(level=value),
    )
    monkeypatch.setitem(sys.modules, "screen_brightness_control", fake)
    b = SbcBrightness()
    assert b.get_level() == 55
    b.set_level(120)
    assert state["level"] == 100

    def broken(display=None):
        raise RuntimeError("no WMI")

    fake.get_brightness = broken
    with pytest.raises(ActionError):
        b.get_level()


@pytest.mark.windows
def test_windows_power_commands() -> None:
    calls: list[list[str]] = []
    power = WindowsPower(run=lambda cmd, **kw: calls.append(cmd))
    power.shutdown()
    power.restart()
    assert calls == [["shutdown", "/s", "/t", "0"], ["shutdown", "/r", "/t", "0"]]


def test_fake_backends_default_construct() -> None:
    b = FakeBackends()
    assert b.volume.level == 50 and b.power.calls == []
