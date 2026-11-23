"""ui/overlay.py: pure formatting, and an offscreen pytest-qt smoke test (windows marker)."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402

from grace.core.bus import HEARD, REPLY, EventBus, qt_bridge  # noqa: E402
from grace.core.state import AppState, StateSnapshot, Turn  # noqa: E402
from grace.ui.overlay import (  # noqa: E402
    Overlay,
    chip_html,
    history_html,
    subtitle_html,
)


# --------------------------------------------------------------------------- pure
def test_subtitle_escapes_and_clips() -> None:
    out = subtitle_html("type <b>bold</b> & more", "Typed.")
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; more" in out and "<b>GRACE:</b> Typed." in out
    long = subtitle_html("x" * 500, None)
    assert "…" in long and len(long) < 400


def test_subtitle_reply_only() -> None:
    assert subtitle_html(None, "Stopped.") == "<b>GRACE:</b> Stopped."


def test_chip_shows_mic_gesture_flags_and_provider() -> None:
    snap = StateSnapshot(mic_on=True, gesture_on=True, gesture="SCROLL", provider="gemini")
    out = chip_html(snap)
    assert "Mic <b>on</b>" in out and "Gesture: <b>SCROLL</b>" in out and "gemini" in out
    off = chip_html(StateSnapshot(mic_on=False, gesture_on=False, gesture="READY"))
    assert "Mic <b>off</b>" in off and "Gesture: <b>off</b>" in off  # name hidden when off
    busy = chip_html(StateSnapshot(ptt_active=True, listening=True, mode="dictation"))
    assert "PUSH-TO-TALK · LISTENING · DICTATION" in busy


def test_history_html() -> None:
    assert "No conversation yet" in history_html(())
    out = history_html((Turn("user", "volume 40", 0), Turn("grace", "Volume <40>.", 1)))
    assert "<b>You:</b> volume 40" in out and "<b>GRACE:</b> Volume &lt;40&gt;." in out


# --------------------------------------------------------------------------- Qt smoke test
@pytest.fixture
def overlay(qtbot):  # type: ignore[no-untyped-def]
    bus = EventBus()
    state = AppState(bus, provider="offline")
    bridge = qt_bridge(bus, ("state", HEARD, REPLY))
    shown: list[tuple[frozenset[str], float]] = []
    ov = Overlay(
        bridge,
        state.snapshot(),
        fade_after_s=0.2,
        on_toggle_shown=lambda c, ms: shown.append((c, ms)),
    )
    ov.show()
    yield ov, bus, state, shown
    ov.close()
    bus.stop()


@pytest.mark.windows
def test_windows_are_click_through_frameless_on_top_without_taskbar(overlay) -> None:  # type: ignore[no-untyped-def]
    ov, *_ = overlay
    for w in (ov.subtitle, ov.chip, ov.history):
        flags = w.windowFlags()
        for flag in (
            Qt.WindowType.FramelessWindowHint,
            Qt.WindowType.WindowStaysOnTopHint,
            Qt.WindowType.Tool,
            Qt.WindowType.WindowTransparentForInput,
            Qt.WindowType.WindowDoesNotAcceptFocus,
        ):
            assert flags & flag, flag
        assert w.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        assert w.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        assert w.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert ov.chip.isVisible() and not ov.subtitle.isVisible() and not ov.history.isVisible()


@pytest.mark.windows
def test_subtitle_shows_heard_and_reply_then_fades(overlay, qtbot) -> None:  # type: ignore[no-untyped-def]
    ov, bus, *_ = overlay
    bus.publish(HEARD, text="open notepad")
    bus.publish(REPLY, text="Opening notepad.")
    bus.pump()
    assert ov.subtitle.isVisible()
    text = ov.subtitle.label.text()
    assert "You: open notepad" in text and "Opening notepad." in text
    qtbot.waitUntil(lambda: not ov.subtitle.isVisible(), timeout=3000)


@pytest.mark.windows
def test_positions_bottom_centre_and_bottom_right(overlay) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtGui import QGuiApplication

    ov, bus, *_ = overlay
    bus.publish(REPLY, text="hello")
    bus.pump()
    area = QGuiApplication.primaryScreen().availableGeometry()
    sub, chip = ov.subtitle.geometry(), ov.chip.geometry()
    assert abs(sub.center().x() - area.center().x()) <= 2
    assert sub.bottom() <= area.bottom() and sub.bottom() >= area.bottom() - 100
    assert chip.right() <= area.right() and chip.right() >= area.right() - 60


@pytest.mark.windows
def test_state_changes_update_the_chip_and_report_latency(overlay) -> None:  # type: ignore[no-untyped-def]
    ov, bus, state, shown = overlay
    state.update(cause_t=None, gesture_on=True, gesture="SCROLL")
    bus.pump()
    assert "Gesture: <b>SCROLL</b>" in ov.chip.label.text()
    assert shown == []  # no hotkey caused it: nothing to measure
    import time

    state.update(cause_t=time.monotonic(), mic_on=False)
    bus.pump()
    assert "Mic <b>off</b>" in ov.chip.label.text()
    [(changed, ms)] = shown
    assert changed == {"mic_on"} and 0 <= ms < 200


@pytest.mark.windows
def test_history_panel_toggles(overlay) -> None:  # type: ignore[no-untyped-def]
    ov, bus, state, _ = overlay
    state.add_turn("user", "volume 40")
    state.toggle("history_visible")
    bus.pump()
    assert ov.history.isVisible() and "volume 40" in ov.history.label.text()
    state.toggle("history_visible")
    bus.pump()
    assert not ov.history.isVisible()


@pytest.mark.windows
def test_events_from_worker_threads_arrive_on_the_qt_thread(overlay, qtbot) -> None:  # type: ignore[no-untyped-def]
    ov, bus, state, _ = overlay
    bus.start()
    seen_threads: list[str] = []
    original = ov._apply_state

    def spy(snap):  # type: ignore[no-untyped-def]
        seen_threads.append(threading.current_thread().name)
        original(snap)

    ov._apply_state = spy  # type: ignore[method-assign]
    worker = threading.Thread(target=lambda: state.update(gesture_on=True, gesture="VOLUME"))
    worker.start()
    worker.join()
    qtbot.waitUntil(lambda: "VOLUME" in ov.chip.label.text(), timeout=2000)
    assert seen_threads == [threading.main_thread().name]


@pytest.mark.windows
def test_a_bad_event_does_not_break_the_overlay(overlay) -> None:  # type: ignore[no-untyped-def]
    from grace.core.bus import Event

    ov, bus, *_ = overlay
    ov.on_event(Event("state", {"snapshot": None}))  # logged, not raised
    bus.publish(REPLY, text="still fine")
    bus.pump()
    assert "still fine" in ov.subtitle.label.text()
