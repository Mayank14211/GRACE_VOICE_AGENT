"""Runtime controller: what each hotkey does, wired to state, workers, executor and feedback.

No Qt and no hardware here: ``grace/__main__.py`` builds the real parts and hands them in, tests
hand in fakes. Every hotkey handler runs on the keyboard hook thread, so each one only flips
state, wakes a worker or queues work, and returns (well under the 200 ms toggle budget).

    mic on/off      state.mic_on; the voice worker opens the mic only while it is on
    gestures on/off state.gesture_on; the gesture loop opens the camera only while it is on
    push-to-talk    one command without the wake word (works with the mic off)
    panic stop      mic + gestures + push-to-talk off, pending gesture actions dropped, every
                    mouse button GRACE holds released, pending confirmation cancelled, speech
                    stopped. The app stays alive; turn things back on with the hotkeys.
    confirm         runs the pending risky action (confirmation gate)
    history         shows / hides the history panel
    mark wrong      logs the last action as wrong (metrics)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from grace.actions.registry import ActionResult
from grace.core.bus import ACTION, QUIT, REPLY, EventBus
from grace.core.state import AppState
from grace.feedback.earcons import Earcons, SilentEarcons
from grace.input.hotkeys import PTT_RELEASE
from grace.intent.types import Intent
from grace.metrics.log import Metrics, NullMetrics

log = logging.getLogger(__name__)

QUIET_ACTIONS = frozenset({"input.move_to"})  # too frequent for metrics / "last action"


class GestureSwitch(Protocol):
    def set_enabled(self, on: bool) -> None: ...


class VoiceControl(Protocol):
    def wake(self) -> None: ...

    def request_ptt(self) -> None: ...

    def release_ptt(self) -> None: ...

    def cancel(self) -> None: ...


class Executor(Protocol):
    def submit(self, intent: Intent) -> None: ...

    def clear_pending(self) -> int: ...


class Gate(Protocol):
    @property
    def pending(self) -> Intent | None: ...

    def confirm(self) -> ActionResult | None: ...

    def cancel(self, *, announce: bool = True) -> bool: ...


@dataclass(frozen=True)
class LastAction:
    name: str
    source: str
    ok: bool
    message: str
    t: float


class CpuMeter:
    """Process CPU use between samples, from time.process_time (no psutil needed)."""

    def __init__(
        self,
        process_time: Callable[[], float] = time.process_time,
        clock: Callable[[], float] = time.monotonic,
        cores: int | None = None,
    ) -> None:
        self._process_time = process_time
        self._clock = clock
        self.cores = cores or os.cpu_count() or 1
        self._last = (process_time(), clock())

    def sample(self) -> tuple[float, float]:
        """(percent of one core, percent of the whole machine) since the previous sample."""
        cpu, now = self._process_time(), self._clock()
        last_cpu, last_now = self._last
        self._last = (cpu, now)
        wall = now - last_now
        if wall <= 0:
            return 0.0, 0.0
        one_core = 100.0 * (cpu - last_cpu) / wall
        return one_core, one_core / self.cores


def _run_in_thread(fn: Callable[[], Any], name: str) -> None:
    threading.Thread(target=fn, name=name, daemon=True).start()


class Runtime:
    def __init__(
        self,
        *,
        state: AppState,
        bus: EventBus,
        executor: Executor,
        gestures: GestureSwitch | None = None,
        voice: VoiceControl | None = None,
        gate: Gate | None = None,
        earcons: Earcons | None = None,
        metrics: Metrics | None = None,
        stop_speech: Callable[[], None] | None = None,
        spawn: Callable[[Callable[[], Any], str], None] = _run_in_thread,
        clock: Callable[[], float] = time.monotonic,
        cpu: CpuMeter | None = None,
    ) -> None:
        self.state = state
        self.bus = bus
        self.executor = executor
        self.gestures = gestures
        self.voice = voice
        self.gate = gate
        self.earcons = earcons or SilentEarcons()
        self.metrics = metrics or NullMetrics()
        self._stop_speech = stop_speech
        self._spawn = spawn
        self._clock = clock
        self._cpu = cpu or CpuMeter()
        self._lock = threading.Lock()
        self.last_action: LastAction | None = None
        self.toggle_latencies_ms: list[float] = []
        self.cpu_samples: list[tuple[float, float]] = []

    # ------------------------------------------------------------------ feedback
    def say(self, text: str) -> None:
        """Overlay-only message (no TTS): hotkey feedback must not talk over the user."""
        self.state.add_turn("grace", text)
        self.bus.publish(REPLY, text=text)

    def _toggled(self, what: str, on: bool) -> None:
        self.earcons.play("on" if on else "off")
        self.metrics.write("toggle", what=what, on=on)
        log.info("%s %s", what, "on" if on else "off")

    # ------------------------------------------------------------------ hotkey handlers
    def handlers(self) -> dict[str, Callable[[float], None]]:
        return {
            "mic_toggle": self.toggle_mic,
            "gesture_toggle": self.toggle_gestures,
            "push_to_talk": self.ptt_press,
            PTT_RELEASE: self.ptt_release,
            "panic_stop": self.panic,
            "confirm": self.confirm,
            "history": self.toggle_history,
            "mark_wrong": self.mark_wrong,
        }

    def set_mic(self, on: bool, t: float | None = None) -> None:
        if not self.state.update(t, mic_on=on):
            return
        if self.voice is not None:
            if not on:
                self.voice.cancel()
            self.voice.wake()
        self._toggled("mic", on)

    def toggle_mic(self, t: float | None = None) -> None:
        self.set_mic(not self.state.mic_on, t)

    def set_gestures(self, on: bool, t: float | None = None) -> None:
        if not self.state.update(t, gesture_on=on):
            return
        if self.gestures is not None:
            self.gestures.set_enabled(on)
        self._toggled("gestures", on)

    def toggle_gestures(self, t: float | None = None) -> None:
        self.set_gestures(not self.state.gesture_on, t)

    def ptt_press(self, t: float | None = None) -> None:
        self.state.update(t, ptt_active=True)
        if self.voice is not None:
            self.voice.request_ptt()
        self.earcons.play("ptt_start")
        self.metrics.write("ptt", down=True)

    def ptt_release(self, t: float | None = None) -> None:
        if not self.state.update(t, ptt_active=False):
            return
        if self.voice is not None:
            self.voice.release_ptt()
        self.earcons.play("ptt_end")
        self.metrics.write("ptt", down=False)

    def panic(self, t: float | None = None) -> None:
        self.state.update(t, mic_on=False, gesture_on=False, ptt_active=False, listening=False)
        if self.gestures is not None:
            self.gestures.set_enabled(False)  # the loop also releases a drag and the camera
        if self.voice is not None:
            self.voice.cancel()
        dropped = self.executor.clear_pending()
        self.executor.submit(Intent("input.release_all", source="hotkey"))
        if self.gate is not None:
            self.gate.cancel(announce=False)
        if self._stop_speech is not None:
            try:
                self._stop_speech()
            except Exception:  # noqa: BLE001
                log.debug("stopping speech failed", exc_info=True)
        self.earcons.play("panic")
        self.metrics.write("panic", dropped=dropped)
        log.warning("PANIC STOP: mic and gestures off, mouse released")
        self.say("Stopped. Mic and gestures are off.")

    def confirm(self, t: float | None = None) -> None:
        gate = self.gate
        if gate is None or gate.pending is None:
            self.say("There's nothing to confirm.")
            return
        self.earcons.play("confirm")
        # gate.confirm runs the action and waits for it: never on the keyboard hook thread.
        self._spawn(gate.confirm, "grace-confirm")

    def toggle_history(self, t: float | None = None) -> None:
        self.state.toggle("history_visible", cause_t=t)

    def mark_wrong(self, t: float | None = None) -> None:
        with self._lock:
            last = self.last_action
        if last is None:
            self.say("No action to mark yet.")
            return
        self.metrics.write(
            "marked_wrong",
            action=last.name,
            source=last.source,
            ok=last.ok,
            age_s=round(self._clock() - last.t, 2),
        )
        self.earcons.play("marked")
        self.say(f"Marked {last.name} as wrong.")

    # ------------------------------------------------------------------ worker callbacks
    def on_action(self, intent: Intent, result: ActionResult) -> None:
        """Executor result callback (executor thread)."""
        if intent.name in QUIET_ACTIONS:
            return
        with self._lock:
            self.last_action = LastAction(
                intent.name, intent.source, result.ok, result.message, self._clock()
            )
        # Args are not logged: they can hold typed text or clipboard contents.
        self.metrics.write("action", name=intent.name, source=intent.source, ok=result.ok)
        self.bus.publish(ACTION, intent=intent, result=result)

    def on_gesture(self, name: str) -> None:
        """Gesture loop callback (camera thread): the name for the status chip."""
        self.state.update(gesture=name)

    def on_toggle_shown(self, changed: frozenset[str], latency_ms: float) -> None:
        """Overlay callback: a hotkey-caused change reached the screen after ``latency_ms``."""
        with self._lock:
            self.toggle_latencies_ms.append(latency_ms)
        self.metrics.write("toggle_latency", changed=sorted(changed), ms=round(latency_ms, 1))
        if latency_ms > 200:
            log.warning("toggle took %.0f ms to reach the screen (budget 200 ms)", latency_ms)

    def sample_cpu(self) -> tuple[float, float]:
        one_core, machine = self._cpu.sample()
        with self._lock:
            self.cpu_samples.append((one_core, machine))
        self.metrics.write("cpu", one_core_pct=round(one_core, 1), machine_pct=round(machine, 1))
        log.info("cpu: %.1f%% of one core (%.1f%% of the machine)", one_core, machine)
        return one_core, machine

    def request_quit(self, reason: str = "voice") -> None:
        self.bus.publish(QUIT, reason=reason)

    def summary(self) -> str:
        with self._lock:
            lat = sorted(self.toggle_latencies_ms)
            cpu = [m for _, m in self.cpu_samples]
        parts = []
        if lat:
            median = lat[len(lat) // 2]
            parts.append(f"toggles shown: {len(lat)}, median {median:.0f} ms, max {lat[-1]:.0f} ms")
        if cpu:
            parts.append(f"cpu (machine): mean {sum(cpu) / len(cpu):.1f}%, max {max(cpu):.1f}%")
        return "; ".join(parts) or "no toggles or cpu samples recorded"
