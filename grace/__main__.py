"""GRACE entry point: voice + gestures + hotkeys + overlay in one process.

    uv run python -m grace                 # the app (main.py does the same)
    uv run python -m grace --gestures      # start with gesture control on
    uv run python -m grace --dry-run       # fake OS backends: nothing on the PC changes
    uv run python -m grace --check-config  # load settings + .env, report providers, exit
    uv run python -m grace doctor [--no-hardware] [--json]

Threads (docs/PLAN.md section 4): Qt owns the main thread (overlay). The voice worker, the
gesture camera loop, the action executor, the event bus and the keyboard hook each run on their
own thread and talk through grace.core.bus; every OS action runs on the one executor thread.

Ctrl+C in the console, the voice command "exit", or closing the console quits. Ctrl+Alt+Q is a
panic stop, not a quit: the app stays alive.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from grace.config import (
    ConfigError,
    Settings,
    format_provider_report,
    load_settings,
    report_providers,
)
from grace.redaction import configure_logging, install_redaction

log = logging.getLogger("grace")

TICK_MS = 250  # hotkey latch expiry + lets Python handle Ctrl+C while Qt runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m grace",
        description="GRACE: voice + gesture desktop assistant. API keys are read from .env only "
        "(see .env.example). 'python -m grace doctor' checks the environment.",
    )
    parser.add_argument("--settings", type=Path, default=None, help="settings file "
                        "(default: settings.yaml in the repo root)")  # fmt: skip
    parser.add_argument("--env-file", type=Path, default=None, help=".env file with API keys "
                        "(default: .env in the repo root)")  # fmt: skip
    parser.add_argument("--check-config", action="store_true",
                        help="load settings and .env, report providers, then exit")  # fmt: skip
    parser.add_argument("--wake-word", type=str, default=None,
                        help="wake word (default: runtime.wake_word in settings.yaml)")  # fmt: skip
    parser.add_argument("--gestures", action="store_true", help="start with gestures on")
    parser.add_argument("--no-voice", action="store_true", help="no microphone / TTS")
    parser.add_argument("--no-overlay", action="store_true", help="console only, no overlay")
    parser.add_argument("--dry-run", action="store_true",
                        help="fake OS backends: nothing on the PC changes")  # fmt: skip
    parser.add_argument("--seconds", type=float, default=0,
                        help="quit after N seconds (soak tests; 0 = run until exit)")  # fmt: skip
    parser.add_argument("--voice-index", type=int, default=None,
                        help="TTS voice index (see --list-voices)")  # fmt: skip
    parser.add_argument("--speech-rate", type=int, default=170, help="TTS words per minute")
    parser.add_argument("--list-voices", action="store_true",
                        help="list the available TTS voices and exit")  # fmt: skip
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["doctor"]:
        from grace.doctor import main as doctor_main

        return doctor_main(argv[1:])
    args = build_parser().parse_args(argv)

    configure_logging()
    try:
        settings = load_settings(args.settings, args.env_file)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    install_redaction(settings.secret_values())
    statuses = report_providers(settings)
    print("LLM providers:\n" + format_provider_report(statuses))
    if args.check_config:
        return 0
    if args.list_voices:
        return _list_voices()
    return run(settings, args)


def _list_voices() -> int:
    from tts import TextToSpeech

    tts = TextToSpeech()
    voices = tts.list_available_voices()
    print(f"Found {len(voices)} voices:")
    for voice in voices:
        print(f"Voice {voice['index']}: {voice['name']} (Gender: {voice['gender'] or 'Unknown'})")
        tts.change_voice(voice["index"])
        tts.speak_async(f"This is the voice of {voice['name']}.")
        time.sleep(2.5)
    return 0


def _create_qt_app(enabled: bool) -> Any:
    """QApplication first (before pyautogui is imported, for DPI awareness), or None."""
    if not enabled:
        return None
    try:
        from grace.ui.overlay import create_app
    except ImportError as exc:  # PySide6 missing: run without the overlay
        log.warning("overlay unavailable (%s); running console-only", exc)
        return None
    return create_app([sys.argv[0]])


def run(settings: Settings, args: argparse.Namespace) -> int:
    from grace.actions import create_fake_registry, create_registry
    from grace.actions.executor import ActionExecutor
    from grace.core.bus import HEARD, PARTIAL, QUIT, REPLY, STATE, EventBus, qt_bridge
    from grace.core.state import AppState
    from grace.feedback.earcons import create_earcons
    from grace.input.gestures import build_gesture_loop
    from grace.metrics.log import MetricsLog, NullMetrics, session_path
    from grace.runtime import Runtime

    rt = settings.runtime
    app = _create_qt_app(rt.overlay and not args.no_overlay)

    bus = EventBus()
    enabled = settings.enabled_providers()
    provider = enabled[0] if enabled else "offline"  # the chip shows the one that last answered
    state = AppState(bus, max_turns=rt.history_turns, mic_on=rt.mic_on_at_start, provider=provider)
    metrics = (
        MetricsLog(session_path(), secrets=settings.secret_values())
        if rt.metrics
        else NullMetrics()
    )
    if args.dry_run:
        registry, _ = create_fake_registry()
        screen = (1920, 1080)
    else:
        from grace.actions.input_ctl import primary_screen_size

        registry = create_registry()
        screen = primary_screen_size()

    runtime_ref: list[Runtime] = []  # the executor and gesture callbacks need the runtime

    executor = ActionExecutor(registry, lambda i, r: runtime_ref[0].on_action(i, r))
    gestures = build_gesture_loop(
        settings.gestures,
        screen,
        executor.submit,
        lambda name: runtime_ref[0].on_gesture(name),
        enabled=False,  # the camera opens only when gestures are turned on
    )

    def heard(text: str) -> None:
        state.add_turn("user", text)
        bus.publish(HEARD, text=text)

    def replied(text: str) -> None:
        state.add_turn("grace", text)
        bus.publish(REPLY, text=text)

    earcons = create_earcons(rt.earcons)
    agent = voice = tts = None
    voice_lines: list[str] = []
    if not args.no_voice:
        agent, voice, tts, voice_lines = _build_voice(
            settings, args, state, bus, metrics, earcons, registry, executor, runtime_ref, heard,
            replied,
        )  # fmt: skip
    if voice is None and rt.mic_on_at_start:
        state.update(mic_on=False)

    runtime = Runtime(
        state=state,
        bus=bus,
        executor=executor,
        gestures=gestures,
        voice=voice,
        gate=agent.gate if agent is not None else None,
        earcons=earcons,
        metrics=metrics,
        stop_speech=tts.stop if tts is not None else None,
    )
    runtime_ref.append(runtime)

    hotkeys = _start_hotkeys(settings, runtime)
    quit_now = threading.Event()
    bus.subscribe(QUIT, lambda e: quit_now.set())

    overlay = bridge = None
    if app is not None:
        from grace.ui.overlay import Overlay

        bridge = qt_bridge(bus, (STATE, HEARD, PARTIAL, REPLY, QUIT))
        overlay = Overlay(
            bridge,
            state.snapshot(),
            fade_after_s=rt.subtitle_fade_s,
            on_toggle_shown=runtime.on_toggle_shown,
            # No voice = nothing for the mic button to switch, so it is not shown.
            on_mic_click=(
                (lambda t: runtime.toggle_mic(t, source="button"))
                if rt.mic_button and voice is not None
                else None
            ),
            mic_hotkey=settings.hotkeys.mic_toggle,
            on_gesture_click=(
                (lambda t: runtime.toggle_gestures(t, source="button"))
                if rt.gesture_button
                else None
            ),
            gesture_hotkey=settings.hotkeys.gesture_toggle,
        )
        bridge.event.connect(lambda e: app.quit() if e.topic == QUIT else None)

    bus.start()
    executor.start()
    gestures.start()
    if voice is not None:
        voice.start()
    if rt.gestures_on_at_start or args.gestures:
        runtime.set_gestures(True)

    mode = "DRY RUN (nothing on the PC changes)" if args.dry_run else "LIVE"
    hk = settings.hotkeys
    for line in voice_lines:
        print(f"Voice: {line}")
    print(
        f"GRACE running ({mode}). Hotkeys: mic {hk.mic_toggle}, gestures "
        f"{hk.gesture_toggle}, push-to-talk {hk.push_to_talk} (hold), panic {hk.panic_stop}, "
        f"confirm {hk.confirm}, history {hk.history}, mark wrong {hk.mark_wrong}. Ctrl+C quits."
    )
    if agent is not None and voice is not None:
        wake = voice.pipeline.wake_mode
        how = {
            "model": "Say the wake word",
            "text": f"Say {agent.wake_word}",
            "off": "Just speak",
        }[wake]
        agent._say(f"GRACE is ready. {how}, or hold Control Alt Space.")
    started = time.monotonic()
    try:
        if app is not None and overlay is not None:
            _run_qt(app, overlay, runtime, hotkeys, rt.cpu_sample_s, args.seconds)
        else:
            _run_headless(quit_now, runtime, hotkeys, rt.cpu_sample_s, args.seconds)
    finally:
        _shutdown(hotkeys, voice, gestures, executor, bus, runtime, metrics, overlay, started)
    return 0


def _build_voice(
    settings: Settings,
    args: argparse.Namespace,
    state: Any,
    bus: Any,
    metrics: Any,
    earcons: Any,
    registry: Any,
    executor: Any,
    runtime_ref: list[Any],
    heard: Any,
    replied: Any,
) -> tuple[Any, Any, Any, list[str]]:
    """(agent, voice worker or None, tts, description lines). Voice that can't start (models
    missing) leaves GRACE running with gestures and hotkeys, and says why."""
    from grace.actions.executor import ExecutorRegistry
    from grace.core.bus import PARTIAL
    from grace.core.quota import build_governor
    from grace.input.voice import (
        VoiceHooks,
        VoiceUnavailable,
        build_voice_worker,
        stt_event_fields,
    )
    from grace.intent.assistant import build_extras
    from grace.intent.dialogue import Dialogue
    from grace.intent.router import build_router
    from grace.legacy_voice import GraceVoiceAgent
    from tts import TextToSpeech

    tts = TextToSpeech()
    if args.voice_index is not None:
        tts.change_voice(args.voice_index)
    if args.speech_rate != 170:
        tts.change_rate(args.speech_rate)
    voice_ref: list[Any] = []
    router_registry = ExecutorRegistry(registry, executor, "router")

    def quota_warning(key: str, counter: str, fraction: float) -> None:
        metrics.write("quota_warning", key=key, counter=counter, pct=round(fraction * 100))
        replied(f"Heads up: {key.split(':')[0]} is at {fraction:.0%} of its free {counter} limit.")

    quota = build_governor(settings, on_warning=quota_warning)
    extras = build_extras(settings, router_registry, quota=quota)
    router = build_router(
        settings,
        registry,
        search=extras.offers_search,
        images=extras.offers_images,
        quota=quota,
        metrics=metrics,
        on_provider=lambda name: state.update(provider=name),
    )
    extras.analyst = router  # code analysis runs the chain without Gemini
    r = settings.router
    dialogue = Dialogue(
        max_turns=r.history_turns, memory_reset_s=r.memory_reset_s, followup_s=r.followup_s
    )
    agent = GraceVoiceAgent(
        confirm_hotkey=settings.hotkeys.confirm,
        registry=ExecutorRegistry(registry, executor, "grammar"),
        tts=tts,
        router=router,
        extras=extras,
        dialogue=dialogue,
        wake_word=args.wake_word or settings.runtime.wake_word,
        on_heard=heard,
        on_reply=replied,
        on_mode=lambda mode: state.update(mode=mode),
        on_end_session=lambda: voice_ref[0].pipeline.end_session() if voice_ref else None,
        on_exit=lambda: runtime_ref[0].request_quit("voice"),
    )
    extras.notify = agent._say  # "Searching the web." while a slow tool runs

    def on_wake(score: float | None) -> None:
        earcons.play("wake")
        metrics.write("wake", score=None if score is None else round(score, 3))

    def on_stt(utt: Any, final: Any) -> None:
        fields = stt_event_fields(utt, final)
        metrics.write("stt", **fields)
        if fields["eos_to_final_ms"] is not None:
            log.info(
                "stt %s: %.0f ms after end of speech (%s)",
                final.engine,
                fields["eos_to_final_ms"],
                utt.kind,
            )

    hooks = VoiceHooks(
        command=agent.handle_utterance,
        partial=lambda text: bus.publish(PARTIAL, text=text),
        listening=lambda on: state.update(listening=on),
        wake=on_wake,
        stt_result=on_stt,
        problem=replied,
        is_dictation=lambda: agent.mode == "dictation",
        is_muted=tts.is_speaking,
        dictation_cancelled=agent.cancel_dictation,
    )
    try:
        voice, lines = build_voice_worker(
            settings, hooks, mic_on=lambda: state.mic_on, wake_word=agent.wake_word
        )
    except VoiceUnavailable as exc:
        print(f"Voice input unavailable: {exc}")
        replied(f"Voice input unavailable: {exc}")
        return agent, None, tts, []
    voice_ref.append(voice)
    return agent, voice, tts, lines


def _start_hotkeys(settings: Settings, runtime: Any) -> Any:
    from grace.input.hotkeys import HotkeyManager, KeyboardLibBackend

    try:
        backend = KeyboardLibBackend()
    except Exception as exc:  # noqa: BLE001 - keyboard missing / not Windows
        log.warning("global hotkeys unavailable: %s", exc)
        return None
    return HotkeyManager(settings.hotkeys, runtime.handlers(), backend).start()


def _run_qt(
    app: Any, overlay: Any, runtime: Any, hotkeys: Any, cpu_s: float, seconds: float
) -> None:
    from PySide6.QtCore import QTimer

    overlay.show()
    tick = QTimer()
    tick.timeout.connect(lambda: hotkeys.expire_stale() if hotkeys is not None else None)
    tick.start(TICK_MS)
    cpu = QTimer()
    cpu.timeout.connect(runtime.sample_cpu)
    cpu.start(int(cpu_s * 1000))
    if seconds:
        QTimer.singleShot(int(seconds * 1000), app.quit)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    app.exec()


def _run_headless(
    quit_now: threading.Event, runtime: Any, hotkeys: Any, cpu_s: float, seconds: float
) -> None:
    start = last_cpu = time.monotonic()
    try:
        while not quit_now.wait(TICK_MS / 1000):
            now = time.monotonic()
            if hotkeys is not None:
                hotkeys.expire_stale()
            if now - last_cpu >= cpu_s:
                runtime.sample_cpu()
                last_cpu = now
            if seconds and now - start >= seconds:
                break
    except KeyboardInterrupt:
        pass


def _shutdown(
    hotkeys: Any,
    voice: Any,
    gestures: Any,
    executor: Any,
    bus: Any,
    runtime: Any,
    metrics: Any,
    overlay: Any,
    started: float,
) -> None:
    print("Stopping GRACE...")
    if hotkeys is not None:
        hotkeys.stop()
    if voice is not None:
        voice.stop()
    gestures.stop()  # releases a held drag, closes the camera
    executor.stop()  # runs what is queued (a pending mouse_up), then stops
    runtime.sample_cpu()
    minutes = (time.monotonic() - started) / 60
    summary = f"ran {minutes:.1f} min; {runtime.summary()}"
    metrics.write("session_end", minutes=round(minutes, 2), summary=summary)
    print(f"GRACE stopped: {summary}")
    metrics.close()
    runtime.earcons.close()
    if overlay is not None:
        overlay.close()
    bus.stop()


if __name__ == "__main__":
    sys.exit(main())
