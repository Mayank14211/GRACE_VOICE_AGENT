"""ui/overlay.py: pure formatting, and an offscreen pytest-qt smoke test (windows marker)."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402

from grace.core.bus import HEARD, REPLY, EventBus, qt_bridge  # noqa: E402
from grace.core.state import AppState, StateSnapshot, Turn  # noqa: E402
from grace.ui.overlay import (  # noqa: E402
    BUTTON_GAP,
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


# --------------------------------------------------------------------------- mic button (A-001)
@pytest.fixture
def with_button(qtbot):  # type: ignore[no-untyped-def]
    bus = EventBus()
    state = AppState(bus)
    bridge = qt_bridge(bus, ("state", HEARD, REPLY))
    clicks: list[float] = []
    ov = Overlay(bridge, state.snapshot(), on_mic_click=clicks.append, mic_hotkey="ctrl+alt+m")
    ov.show()
    yield ov, bus, state, clicks
    ov.close()


def test_no_button_without_a_click_handler(overlay) -> None:  # type: ignore[no-untyped-def]
    ov, *_ = overlay
    assert ov.mic_button is None


@pytest.mark.windows
def test_button_takes_clicks_but_never_focus(with_button) -> None:  # type: ignore[no-untyped-def]
    ov, *_ = with_button
    btn = ov.mic_button
    flags = btn.windowFlags()
    assert not flags & Qt.WindowType.WindowTransparentForInput  # it must receive clicks
    assert not btn.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    for flag in (
        Qt.WindowType.FramelessWindowHint,
        Qt.WindowType.WindowStaysOnTopHint,
        Qt.WindowType.Tool,
        Qt.WindowType.WindowDoesNotAcceptFocus,
    ):
        assert flags & flag, flag
    assert btn.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert btn.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert "ctrl+alt+m" in btn.toolTip()
    # The rest of the overlay stays click-through.
    for w in (ov.subtitle, ov.chip, ov.history):
        assert w.windowFlags() & Qt.WindowType.WindowTransparentForInput


@pytest.mark.windows
def test_click_calls_the_handler_with_the_click_time(with_button, qtbot) -> None:  # type: ignore[no-untyped-def]
    ov, _, _, clicks = with_button
    qtbot.mouseClick(ov.mic_button, Qt.MouseButton.LeftButton)
    assert len(clicks) == 1 and clicks[0] > 0
    qtbot.mouseClick(ov.mic_button, Qt.MouseButton.RightButton)  # only the left button counts
    assert len(clicks) == 1


@pytest.mark.windows
def test_press_then_release_outside_does_not_click(with_button, qtbot) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtCore import QPoint

    ov, _, _, clicks = with_button
    btn = ov.mic_button
    qtbot.mousePress(btn, Qt.MouseButton.LeftButton)
    qtbot.mouseRelease(btn, Qt.MouseButton.LeftButton, pos=QPoint(-20, -20))
    assert clicks == []


@pytest.mark.windows
def test_a_crashing_click_handler_is_contained(qtbot) -> None:  # type: ignore[no-untyped-def]
    bus = EventBus()
    bridge = qt_bridge(bus, ("state",))

    def boom(t: float) -> None:
        raise RuntimeError("bad")

    ov = Overlay(bridge, AppState(bus).snapshot(), on_mic_click=boom)
    ov.show()
    qtbot.mouseClick(ov.mic_button, Qt.MouseButton.LeftButton)  # logged, not raised
    ov.close()


@pytest.mark.windows
def test_button_mirrors_mic_and_listening_state(with_button) -> None:  # type: ignore[no-untyped-def]
    ov, bus, state, _ = with_button
    btn = ov.mic_button
    assert btn.on and not btn.listening and btn.ring() is None
    state.update(mic_on=False)
    bus.pump()
    assert not btn.on and btn.ring() is None
    state.update(mic_on=True, listening=True)
    bus.pump()
    assert btn.on and btn.ring() is not None
    assert btn.grab().width() > 0  # it paints without error (both looks)
    state.update(mic_on=False)
    bus.pump()
    assert btn.grab().width() > 0


@pytest.mark.windows
def test_button_sits_left_of_the_chip_and_follows_its_width(with_button) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtGui import QGuiApplication

    ov, bus, state, _ = with_button
    area = QGuiApplication.primaryScreen().availableGeometry()
    btn, chip = ov.mic_button, ov.chip
    assert btn.geometry().right() < chip.geometry().left()
    assert btn.geometry().bottom() <= area.bottom()
    gap = chip.x() - btn.width() - BUTTON_GAP
    assert btn.x() == max(area.x(), gap)
    state.update(gesture_on=True, gesture="BRIGHTNESS", mode="dictation", ptt_active=True)
    bus.pump()  # the chip's text (and so its width and x) changes; the button keeps its place
    assert btn.x() == max(area.x(), chip.x() - btn.width() - BUTTON_GAP)


@pytest.mark.windows
def test_click_to_screen_latency_is_reported(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Click -> Runtime.toggle_mic(t) -> state -> chip repaint: reported through on_toggle_shown."""
    bus = EventBus()
    state = AppState(bus)
    shown: list[float] = []
    bridge = qt_bridge(bus, ("state",))
    ov = Overlay(
        bridge,
        state.snapshot(),
        on_toggle_shown=lambda c, ms: shown.append(ms),
        on_mic_click=lambda t: state.update(cause_t=t, mic_on=not state.mic_on),
    )
    ov.show()
    qtbot.mouseClick(ov.mic_button, Qt.MouseButton.LeftButton)
    bus.pump()
    assert state.mic_on is False and len(shown) == 1 and 0 <= shown[0] < 200
    assert not ov.mic_button.on
    ov.close()


@pytest.mark.windows
def test_status_chip_is_a_single_line(overlay) -> None:  # type: ignore[no-untyped-def]
    """Regression: a word-wrapping QLabel made the chip three lines tall and 139 px wide."""
    ov, bus, state, _ = overlay
    state.update(gesture_on=True, gesture="BRIGHTNESS", mode="dictation", ptt_active=True)
    bus.pump()
    line = ov.chip.label.fontMetrics().height()
    assert ov.chip.height() < line * 2.5, ov.chip.height()
    assert ov.chip.width() > ov.chip.height() * 5


# --------------------------------------------------------------------------- gesture button (A-002)
@pytest.fixture
def with_buttons(qtbot):  # type: ignore[no-untyped-def]
    bus = EventBus()
    state = AppState(bus)
    bridge = qt_bridge(bus, ("state", HEARD, REPLY))
    mic: list[float] = []
    cam: list[float] = []
    ov = Overlay(
        bridge,
        state.snapshot(),
        on_mic_click=mic.append,
        on_gesture_click=cam.append,
        gesture_hotkey="ctrl+alt+g",
    )
    ov.show()
    yield ov, bus, state, mic, cam
    ov.close()


@pytest.mark.windows
def test_gesture_button_takes_clicks_but_never_focus(with_buttons, qtbot) -> None:  # type: ignore[no-untyped-def]
    ov, _, _, mic, cam = with_buttons
    btn = ov.gesture_button
    assert not btn.windowFlags() & Qt.WindowType.WindowTransparentForInput
    assert btn.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assert btn.focusPolicy() == Qt.FocusPolicy.NoFocus and "ctrl+alt+g" in btn.toolTip()
    qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)
    assert len(cam) == 1 and mic == []  # each button only calls its own handler


@pytest.mark.windows
def test_buttons_line_up_leftwards_from_the_chip(with_buttons) -> None:  # type: ignore[no-untyped-def]
    ov, bus, state, *_ = with_buttons
    mic, cam, chip = ov.mic_button, ov.gesture_button, ov.chip
    assert cam.geometry().right() < mic.geometry().left() < mic.geometry().right()
    assert mic.geometry().right() < chip.geometry().left()
    assert cam.y() == mic.y()
    state.update(gesture_on=True, gesture="SCROLL", ptt_active=True)
    bus.pump()  # the chip's text changes; both buttons keep their places relative to it
    assert mic.x() == max(0, chip.x() - mic.width() - BUTTON_GAP)
    assert cam.x() == max(0, mic.x() - cam.width() - BUTTON_GAP)


@pytest.mark.windows
def test_gesture_button_mirrors_state_and_camera_problems(with_buttons) -> None:  # type: ignore[no-untyped-def]
    ov, bus, state, *_ = with_buttons
    btn = ov.gesture_button
    assert not btn.on and btn.ring() is None  # gestures start off
    state.update(gesture_on=True, gesture="READY")
    bus.pump()
    assert btn.on and btn.ring() is None
    state.update(gesture="NO CAMERA")
    bus.pump()
    assert btn.on and btn.ring() is not None  # on, but the camera is missing / busy
    state.update(gesture_on=False, gesture="OFF")
    bus.pump()
    assert not btn.on and btn.ring() is None
    assert btn.grab().width() > 0  # paints in every look


def test_only_requested_buttons_exist(overlay) -> None:  # type: ignore[no-untyped-def]
    ov, *_ = overlay
    assert ov.mic_button is None and ov.gesture_button is None
