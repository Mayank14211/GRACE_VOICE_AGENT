"""grace/runtime.py with fakes: toggles, push-to-talk, panic stop, confirm, history, mark wrong."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from grace.actions import create_fake_registry
from grace.actions.confirmation import ConfirmationGate
from grace.actions.executor import ActionExecutor
from grace.actions.registry import ActionResult
from grace.config import HotkeySettings
from grace.core.bus import ACTION, QUIT, REPLY, STATE, Event, EventBus
from grace.core.state import AppState
from grace.feedback.earcons import FakeEarcons
from grace.input.hotkeys import FakeKeyboard, HotkeyManager
from grace.intent.types import Intent
from grace.metrics.log import MetricsLog
from grace.runtime import CpuMeter, Runtime


class FakeGestures:
    def __init__(self) -> None:
        self.enabled: list[bool] = []

    def set_enabled(self, on: bool) -> None:
        self.enabled.append(on)


class FakeVoice:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def wake(self) -> None:
        self.calls.append("wake")

    def request_ptt(self) -> None:
        self.calls.append("ptt")

    def release_ptt(self) -> None:
        self.calls.append("release")

    def cancel(self) -> None:
        self.calls.append("cancel")


class Harness:
    def __init__(self, tmp_path: Path, *, mic_on: bool = True, gesture_on: bool = False) -> None:
        self.bus = EventBus()
        self.state = AppState(self.bus, mic_on=mic_on, gesture_on=gesture_on)
        self.registry, self.b = create_fake_registry()
        self.executor = ActionExecutor(self.registry)
        self.gestures = FakeGestures()
        self.voice = FakeVoice()
        self.said: list[str] = []
        self.gate = ConfirmationGate(self.registry, say=self.said.append, start_timer=None)
        self.earcons = FakeEarcons()
        self.metrics_path = tmp_path / "m.jsonl"
        self.metrics = MetricsLog(self.metrics_path)
        self.spawned: list[tuple[Callable[[], Any], str]] = []
        self.stopped_speech = 0
        self.rt = Runtime(
            state=self.state,
            bus=self.bus,
            executor=self.executor,
            gestures=self.gestures,
            voice=self.voice,
            gate=self.gate,
            earcons=self.earcons,
            metrics=self.metrics,
            stop_speech=self._stop_speech,
            spawn=lambda fn, name: self.spawned.append((fn, name)),
        )
        self.events: list[Event] = []
        self.bus.subscribe("*", self.events.append)

    def _stop_speech(self) -> None:
        self.stopped_speech += 1

    def rows(self) -> list[dict[str, Any]]:
        self.metrics.close()
        return [json.loads(x) for x in self.metrics_path.read_text(encoding="utf-8").splitlines()]

    def replies(self) -> list[str]:
        self.bus.pump()
        return [e.data["text"] for e in self.events if e.topic == REPLY]


@pytest.fixture
def h(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def test_mic_toggle(h: Harness) -> None:
    h.rt.toggle_mic(5.0)
    assert h.state.mic_on is False
    assert h.voice.calls == ["cancel", "wake"]  # what the current listen returns is dropped
    h.rt.toggle_mic(6.0)
    assert h.state.mic_on is True and h.voice.calls[-1] == "wake"
    assert h.earcons.played == ["off", "on"]
    h.bus.pump()
    causes = [e.data["cause_t"] for e in h.events if e.topic == STATE]
    assert causes == [5.0, 6.0]
    assert [r["on"] for r in h.rows() if r["kind"] == "toggle"] == [False, True]


def test_gesture_toggle_drives_the_loop(h: Harness) -> None:
    h.rt.toggle_gestures(1.0)
    h.rt.toggle_gestures(2.0)
    assert h.gestures.enabled == [True, False]
    assert h.earcons.played == ["on", "off"]


def test_setting_the_same_value_is_a_no_op(h: Harness) -> None:
    h.rt.set_gestures(False)
    h.rt.set_mic(True)
    assert h.gestures.enabled == [] and h.voice.calls == [] and h.earcons.played == []


def test_push_to_talk(h: Harness) -> None:
    h.rt.ptt_press(1.0)
    assert h.state.ptt_active and h.voice.calls == ["ptt"]
    h.rt.ptt_release(2.0)
    h.rt.ptt_release(2.1)  # duplicate release: ignored
    assert not h.state.ptt_active and h.voice.calls == ["ptt", "release"]
    assert h.earcons.played == ["ptt_start", "ptt_end"]


def test_panic_stops_everything_and_releases_the_mouse(h: Harness) -> None:
    h.rt.set_gestures(True)
    h.rt.ptt_press()
    h.registry.execute("input.mouse_down", {"button": "left"})  # a drag in progress
    for x in range(5):
        h.executor.submit(Intent("input.move_to", {"x": x, "y": 0}, source="gesture"))
    h.executor.submit(Intent("input.click", {"button": "left", "count": 1}, source="gesture"))
    h.gate.request(Intent("system.shutdown"))
    h.b.input.calls.clear()

    h.rt.panic(9.0)
    h.executor.run_pending()

    snap = h.state.snapshot()
    assert (snap.mic_on, snap.gesture_on, snap.ptt_active) == (False, False, False)
    assert h.gestures.enabled[-1] is False
    assert "cancel" in h.voice.calls
    assert h.b.input.calls == [("mouse_up", "left")]  # pending moves / click dropped
    assert h.gate.pending is None and h.b.power.calls == []
    assert h.stopped_speech == 1
    assert h.earcons.played[-1] == "panic"
    assert h.replies()[-1] == "Stopped. Mic and gestures are off."
    assert any(r["kind"] == "panic" and r["dropped"] == 2 for r in h.rows())


def test_panic_keeps_the_app_usable(h: Harness) -> None:
    h.rt.panic()
    h.rt.toggle_mic()
    h.rt.toggle_gestures()
    assert h.state.mic_on and h.state.gesture_on
    h.bus.pump()
    assert not any(e.topic == QUIT for e in h.events)


def test_confirm_runs_the_gate_off_the_hook_thread(h: Harness) -> None:
    h.gate.request(Intent("system.restart"))
    h.rt.confirm()
    assert h.b.power.calls == []  # not run on the calling (keyboard) thread
    [(fn, name)] = h.spawned
    assert name == "grace-confirm"
    fn()
    assert h.b.power.calls == ["restart"]


def test_confirm_with_nothing_pending(h: Harness) -> None:
    h.rt.confirm()
    assert h.spawned == [] and h.replies() == ["There's nothing to confirm."]


def test_history_toggle(h: Harness) -> None:
    h.rt.toggle_history(1.0)
    assert h.state.snapshot().history_visible
    h.rt.toggle_history(2.0)
    assert not h.state.snapshot().history_visible


def test_mark_wrong_logs_the_last_real_action(h: Harness) -> None:
    h.rt.mark_wrong()
    assert h.replies() == ["No action to mark yet."]
    h.rt.on_action(Intent("volume.set", {"level": 40}), ActionResult(True, "Volume 40."))
    h.rt.on_action(
        Intent("input.move_to", {"x": 1, "y": 1}, source="gesture"), ActionResult(True, "")
    )
    h.rt.mark_wrong()
    rows = h.rows()
    [marked] = [r for r in rows if r["kind"] == "marked_wrong"]
    assert marked["action"] == "volume.set" and marked["source"] == "grammar"
    actions = [r for r in rows if r["kind"] == "action"]
    assert [a["name"] for a in actions] == ["volume.set"]  # cursor moves are not logged
    assert "level" not in json.dumps(actions)  # args never reach the metrics file
    assert h.earcons.played == ["marked"]
    h.bus.pump()
    assert [e.topic for e in h.events].count(ACTION) == 1


def test_gesture_name_reaches_state(h: Harness) -> None:
    h.rt.on_gesture("SCROLL")
    assert h.state.snapshot().gesture == "SCROLL"


def test_toggle_latency_is_recorded(h: Harness, caplog: pytest.LogCaptureFixture) -> None:
    h.rt.on_toggle_shown(frozenset({"mic_on"}), 12.0)
    h.rt.on_toggle_shown(frozenset({"gesture_on"}), 250.0)
    assert "budget 200 ms" in caplog.text
    assert "median 250 ms, max 250 ms" in h.rt.summary()
    assert [r["ms"] for r in h.rows() if r["kind"] == "toggle_latency"] == [12.0, 250.0]


def test_cpu_meter() -> None:
    cpu_t = iter([0.0, 0.5, 2.5])
    wall = iter([0.0, 1.0, 3.0])
    meter = CpuMeter(lambda: next(cpu_t), lambda: next(wall), cores=4)
    assert meter.sample() == (50.0, 12.5)
    assert meter.sample() == (100.0, 25.0)


def test_request_quit(h: Harness) -> None:
    h.rt.request_quit("voice")
    h.bus.pump()
    assert [e.data["reason"] for e in h.events if e.topic == QUIT] == ["voice"]


def test_hotkeys_end_to_end_with_a_fake_keyboard(h: Harness) -> None:
    kb = FakeKeyboard()
    HotkeyManager(HotkeySettings(), h.rt.handlers(), kb).start()
    kb.press("ctrl+alt+g")
    kb.release("g")
    kb.press("ctrl+alt+m")
    kb.press("ctrl+alt+space")
    kb.release("space")
    kb.press("ctrl+alt+h")
    kb.press("ctrl+alt+q")
    snap = h.state.snapshot()
    assert snap.history_visible and not snap.mic_on and not snap.gesture_on
    assert h.gestures.enabled == [True, False]
    assert h.voice.calls[:4] == ["cancel", "wake", "ptt", "release"]
