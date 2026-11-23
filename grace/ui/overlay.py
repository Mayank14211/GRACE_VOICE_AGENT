"""Overlay v1: transparent, click-through, always-on-top windows on the primary screen.

    app = create_app()                                  # before anything imports pyautogui
    overlay = Overlay(bridge, on_toggle_shown=runtime.on_toggle_shown)
    overlay.show()

Three small top-level windows, each frameless, translucent, always on top, without a taskbar
entry (Qt.Tool), never activated or focused, and transparent for mouse and keyboard input
(WindowTransparentForInput sets WS_EX_TRANSPARENT on Windows, so clicks reach the app below):

- subtitle bar, bottom-centre: what GRACE heard and replied; fades out ``fade_after_s`` after
  the last line;
- status chip, bottom-right: mic, gestures and the current gesture name, push-to-talk /
  listening / dictation flags, the LLM provider;
- history panel (Ctrl+Alt+H), above the chip: the last dialogue turns.

All widget code runs on the Qt main thread: events arrive through the bus bridge signal
(``grace.core.bus.qt_bridge``), which Qt queues onto this thread. Qt 6 makes the process per-
monitor DPI aware; sizes are in points / device-independent pixels, so text scales with the
display. Positions use the primary screen's available geometry (above the taskbar) and follow it
when the resolution or scaling changes.
"""

from __future__ import annotations

import html
import logging
import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEasingCurve, QObject, QPropertyAnimation, Qt, QTimer, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from grace.core.bus import HEARD, REPLY, STATE, Event
from grace.core.state import StateSnapshot, Turn

log = logging.getLogger(__name__)

MARGIN = 24  # px from the screen edges
FADE_MS = 400
SUBTITLE_MAX_CHARS = 220
SHOWN_CHANGES = frozenset({"mic_on", "gesture_on", "ptt_active", "history_visible"})

ON, OFF, ACCENT, DIM = "#5be37d", "#9aa0a6", "#ffd166", "#c8ccd0"

_PANEL_STYLE = (
    "background-color: rgba(18, 18, 20, 215); color: #f1f3f4; border-radius: 10px;"
    "padding: 8px 14px; font-family: 'Segoe UI', sans-serif;"
)


# --------------------------------------------------------------------------- pure formatting
def _clip(text: str, limit: int = SUBTITLE_MAX_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def subtitle_html(heard: str | None, reply: str | None) -> str:
    lines = []
    if heard:
        lines.append(f"<span style='color:{DIM}'>You: {html.escape(_clip(heard))}</span>")
    if reply:
        lines.append(f"<b>GRACE:</b> {html.escape(_clip(reply))}")
    return "<br>".join(lines)


def _flag(label: str, on: bool, value: str | None = None) -> str:
    colour = ON if on else OFF
    text = value if value is not None else ("on" if on else "off")
    return f"<span style='color:{colour}'>●</span> {label} <b>{html.escape(text)}</b>"


def chip_html(s: StateSnapshot) -> str:
    gesture = s.gesture if s.gesture_on else "off"
    parts = [
        _flag("Mic", s.mic_on),
        _flag("Gesture:", s.gesture_on, gesture),
    ]
    extras = []
    if s.ptt_active:
        extras.append("PUSH-TO-TALK")
    if s.listening:
        extras.append("LISTENING")
    if s.mode == "dictation":
        extras.append("DICTATION")
    if extras:
        parts.append(f"<span style='color:{ACCENT}'><b>{' · '.join(extras)}</b></span>")
    parts.append(f"<span style='color:{OFF}'>{html.escape(s.provider)}</span>")
    return "&nbsp;&nbsp;&nbsp;".join(parts)


def history_html(turns: tuple[Turn, ...]) -> str:
    if not turns:
        return f"<span style='color:{OFF}'>No conversation yet.</span>"
    rows = []
    for turn in turns:
        who = "You" if turn.speaker == "user" else "GRACE"
        colour = DIM if turn.speaker == "user" else "#f1f3f4"
        text = html.escape(_clip(turn.text, 160))
        rows.append(f"<span style='color:{colour}'><b>{who}:</b> {text}</span>")
    return "<br>".join(rows)


# --------------------------------------------------------------------------- windows
OVERLAY_FLAGS = (
    Qt.WindowType.FramelessWindowHint
    | Qt.WindowType.WindowStaysOnTopHint
    | Qt.WindowType.Tool  # no taskbar entry, no Alt+Tab
    | Qt.WindowType.WindowTransparentForInput  # click-through (WS_EX_TRANSPARENT)
    | Qt.WindowType.WindowDoesNotAcceptFocus  # never steals focus (WS_EX_NOACTIVATE)
    | Qt.WindowType.NoDropShadowWindowHint
)


class OverlayWindow(QWidget):
    def __init__(self, font_pt: float, max_width_ratio: float) -> None:
        super().__init__(None, OVERLAY_FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._max_width_ratio = max_width_ratio
        self.label = QLabel(self)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setWordWrap(True)
        self.label.setStyleSheet(f"{_PANEL_STYLE} font-size: {font_pt}pt;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

    def set_html(self, text: str) -> None:
        screen = QGuiApplication.primaryScreen()
        max_w = int(screen.availableGeometry().width() * self._max_width_ratio) if screen else 900
        self.label.setText(text)
        self.label.setMaximumWidth(max_w)
        self.label.adjustSize()
        self.adjustSize()
        self.place()

    def place(self) -> None:  # overridden
        pass

    def _area(self) -> Any:
        screen = QGuiApplication.primaryScreen()
        return screen.availableGeometry() if screen else None


class SubtitleBar(OverlayWindow):
    def __init__(self, fade_after_s: float = 4.0) -> None:
        super().__init__(font_pt=15, max_width_ratio=0.6)
        self.fade_after_s = fade_after_s
        self.heard: str | None = None
        self.reply: str | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade_out)
        self._anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._anim.setDuration(FADE_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.finished.connect(self._faded)

    def place(self) -> None:
        area = self._area()
        if area is None:
            return
        x = area.x() + (area.width() - self.width()) // 2
        y = area.y() + area.height() - self.height() - MARGIN
        self.move(x, y)

    def show_heard(self, text: str) -> None:
        self.heard, self.reply = text, None  # a new utterance starts a new subtitle
        self._refresh()

    def show_reply(self, text: str) -> None:
        self.reply = text
        self._refresh()

    def _refresh(self) -> None:
        self._anim.stop()
        self.setWindowOpacity(1.0)
        self.set_html(subtitle_html(self.heard, self.reply))
        self.show()
        self.raise_()
        self._timer.start(int(self.fade_after_s * 1000))

    def _fade_out(self) -> None:
        self._anim.setStartValue(self.windowOpacity())
        self._anim.setEndValue(0.0)
        self._anim.start()

    def _faded(self) -> None:
        if self.windowOpacity() <= 0.01:
            self.hide()
            self.heard = self.reply = None

    @property
    def fading(self) -> bool:
        return self._anim.state() == QPropertyAnimation.State.Running


class StatusChip(OverlayWindow):
    def __init__(self) -> None:
        super().__init__(font_pt=10.5, max_width_ratio=0.45)

    def place(self) -> None:
        area = self._area()
        if area is None:
            return
        x = area.x() + area.width() - self.width() - MARGIN
        y = area.y() + area.height() - self.height() - MARGIN
        self.move(x, y)


class HistoryPanel(OverlayWindow):
    def __init__(self, chip: StatusChip) -> None:
        super().__init__(font_pt=11, max_width_ratio=0.32)
        self._chip = chip

    def place(self) -> None:
        area = self._area()
        if area is None:
            return
        x = area.x() + area.width() - self.width() - MARGIN
        y = self._chip.y() - self.height() - 8
        self.move(x, max(area.y(), y))


# --------------------------------------------------------------------------- controller
class Overlay(QObject):
    def __init__(
        self,
        bridge: Any,
        initial: StateSnapshot,
        *,
        fade_after_s: float = 4.0,
        on_toggle_shown: Callable[[frozenset[str], float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.subtitle = SubtitleBar(fade_after_s)
        self.chip = StatusChip()
        self.history = HistoryPanel(self.chip)
        self._on_toggle_shown = on_toggle_shown
        self._clock = clock
        self.snapshot = initial
        bridge.event.connect(self.on_event)
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            screen.availableGeometryChanged.connect(self._replace)

    def show(self) -> None:
        self._apply_state(self.snapshot)
        self.chip.show()

    def close(self) -> None:
        for w in (self.subtitle, self.chip, self.history):
            w.close()

    @Slot(object)
    def on_event(self, event: Event) -> None:
        try:
            if event.topic == STATE:
                self._on_state(event)
            elif event.topic == HEARD:
                self.subtitle.show_heard(event.data.get("text", ""))
            elif event.topic == REPLY:
                self.subtitle.show_reply(event.data.get("text", ""))
        except Exception:  # noqa: BLE001 - a UI glitch must never take the app down
            log.exception("overlay: event %s failed", event.topic)

    def _on_state(self, event: Event) -> None:
        snap: StateSnapshot = event.data["snapshot"]
        self.snapshot = snap
        self._apply_state(snap)
        changed: frozenset[str] = event.data.get("changed", frozenset())
        cause_t = event.data.get("cause_t")
        if cause_t is not None and changed & SHOWN_CHANGES:
            self.chip.repaint()  # paint now, so the measurement includes reaching the screen
            latency_ms = (self._clock() - cause_t) * 1000
            if self._on_toggle_shown is not None:
                self._on_toggle_shown(changed, latency_ms)

    def _apply_state(self, snap: StateSnapshot) -> None:
        self.chip.set_html(chip_html(snap))
        if snap.history_visible:
            self.history.set_html(history_html(snap.turns))
            self.history.show()
        else:
            self.history.hide()

    def _replace(self, *_: Any) -> None:
        for w in (self.subtitle, self.chip, self.history):
            w.place()


def create_app(argv: list[str] | None = None) -> QApplication:
    """The QApplication. Create it before pyautogui is imported: Qt then sets per-monitor DPI
    awareness (v2) and pyautogui's later SetProcessDPIAware is a harmless no-op."""
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(argv or ["grace"])
    app.setQuitOnLastWindowClosed(False)  # overlay windows hide and show; the app stays
    app.setApplicationName("GRACE")
    return app
