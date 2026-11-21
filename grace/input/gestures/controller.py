"""Gesture state machine: per-frame poses -> Intents for the action registry. Pure, no OS calls.

Time comes from frame timestamps (milliseconds), never from the clock, and nothing sleeps, so every
behaviour is reproducible in tests.

Flow per frame::

    raw pose --debounce (hold_frames)--> active pose --gesture map--> behaviour --> Intents

- **Debounce:** a new pose becomes active only after ``hold_frames`` consecutive frames. Frames
  whose raw pose differs from the active one (transitions, glitches) do nothing.
- **Entry points:** every activation starts fresh (scroll/volume/brightness entry point, pinch
  timer, cursor filter). Switching poses, including to the open-palm neutral pose, resets them.
- **cursor:** index fingertip, inner active box (``active_box_margin``) -> full screen, One Euro
  filter. The frame is already mirrored by the tracker.
- **left_click:** the cursor freezes while pinching. Released before ``drag_hold_ms`` (and without
  moving) = exactly one left click, sent on release. Held longer, or moved, = drag: mouse down,
  the cursor follows the pinch, mouse up on release. Losing the hand never clicks.
- **right_click:** one right click when the pose activates.
- **scroll / volume / brightness:** rate proportional to the vertical offset of the palm from its
  entry point beyond ``deadzone`` (up = more); nothing inside the deadzone. Fractions accumulate;
  at most ``max_os_calls_per_s`` intents per second.
- **pause:** held ``pause_hold_ms`` toggles pause (once per hold). While paused only this works.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from grace.input.gestures.classifier import (
    ClassifierConfig,
    HandObservation,
    Point,
    Pose,
    Reading,
    read_hand,
)
from grace.input.gestures.filters import OneEuroFilter2D
from grace.intent.types import Intent

if TYPE_CHECKING:
    from grace.config import GestureSettings


class Behaviour(StrEnum):
    """What a pose does. Values are the settings.yaml ``gestures.map`` values."""

    CURSOR = "cursor"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    SCROLL = "scroll"
    VOLUME = "volume"
    BRIGHTNESS = "brightness"
    PAUSE = "pause"
    NEUTRAL = "neutral"
    NONE = "none"


DEFAULT_MAP: dict[Pose, Behaviour] = {
    Pose.INDEX_UP: Behaviour.CURSOR,
    Pose.PINCH_INDEX: Behaviour.LEFT_CLICK,
    Pose.PINCH_MIDDLE: Behaviour.RIGHT_CLICK,
    Pose.TWO_UP: Behaviour.SCROLL,
    Pose.THREE_UP: Behaviour.BRIGHTNESS,
    Pose.THUMB_INDEX_MIDDLE: Behaviour.VOLUME,
    Pose.FIST: Behaviour.PAUSE,
    Pose.OPEN_PALM: Behaviour.NEUTRAL,
}

# Names for the overlay status chip ("Gesture: SCROLL").
DISPLAY: dict[Behaviour, str] = {
    Behaviour.CURSOR: "MOVE",
    Behaviour.LEFT_CLICK: "CLICK",
    Behaviour.RIGHT_CLICK: "RIGHT CLICK",
    Behaviour.SCROLL: "SCROLL",
    Behaviour.VOLUME: "VOLUME",
    Behaviour.BRIGHTNESS: "BRIGHTNESS",
    Behaviour.PAUSE: "FIST",
    Behaviour.NEUTRAL: "READY",
    Behaviour.NONE: "NONE",
}
PAUSED = "PAUSED"
DRAG = "DRAG"

MAX_FRAME_GAP_MS = 100.0  # longer gaps (dropped frames) count as this, so rates don't jump


@dataclass(frozen=True)
class ControllerConfig:
    screen_size: tuple[int, int] = (1920, 1080)
    gesture_map: Mapping[Pose, Behaviour] = field(default_factory=lambda: dict(DEFAULT_MAP))
    hold_frames: int = 4
    drag_hold_ms: float = 400.0
    pause_hold_ms: float = 1000.0
    deadzone: float = 0.05
    active_box_margin: float = 0.15
    drag_move_threshold: float = 0.04  # pinch point moved this far (frame units) -> drag
    scroll_speed: float = 60.0
    volume_speed: float = 200.0
    brightness_speed: float = 200.0
    max_os_calls_per_s: float = 10.0
    cursor_min_cutoff: float = 1.0
    cursor_beta: float = 10.0
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    def __post_init__(self) -> None:
        if self.hold_frames < 1:
            raise ValueError("hold_frames must be >= 1")
        if not 0 <= self.active_box_margin < 0.5:
            raise ValueError("active_box_margin must be in [0, 0.5)")
        if self.screen_size[0] < 3 or self.screen_size[1] < 3:
            raise ValueError("screen_size too small")
        if not 0 < self.max_os_calls_per_s <= 10:
            raise ValueError("max_os_calls_per_s must be in (0, 10]")

    @classmethod
    def from_settings(cls, gs: GestureSettings, screen_size: tuple[int, int]) -> ControllerConfig:
        return cls(
            screen_size=screen_size,
            gesture_map={Pose(p): Behaviour(b) for p, b in gs.map.items()},
            hold_frames=gs.hold_frames,
            drag_hold_ms=gs.drag_hold_ms,
            pause_hold_ms=gs.pause_hold_ms,
            deadzone=gs.deadzone,
            active_box_margin=gs.active_box_margin,
            scroll_speed=gs.scroll_speed,
            volume_speed=gs.volume_speed,
            brightness_speed=gs.brightness_speed,
            max_os_calls_per_s=gs.max_os_calls_per_s,
            cursor_min_cutoff=gs.cursor_min_cutoff,
            cursor_beta=gs.cursor_beta,
            classifier=ClassifierConfig(pinch_on=gs.pinch_on, pinch_off=gs.pinch_off),
        )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _intent(name: str, **args: Any) -> Intent:
    return Intent(name=name, args=args, source="gesture")


def release_intents() -> list[Intent]:
    """Release every mouse button (panic stop, gestures off)."""
    return [_intent("input.mouse_up", button="left")]


@dataclass
class _Pinch:
    start_t: float
    anchor: Point  # filtered pinch point when the pinch started
    start_px: tuple[int, int]
    dragging: bool = False


@dataclass
class _Level:
    entry_y: float
    last_t: float
    last_emit_t: float
    accum: float = 0.0


class GestureController:
    def __init__(self, cfg: ControllerConfig | None = None) -> None:
        self.cfg = cfg or ControllerConfig()
        self._filter = OneEuroFilter2D(self.cfg.cursor_min_cutoff, self.cfg.cursor_beta)
        self.paused = False
        self._cursor: tuple[int, int] | None = None  # last position we moved the mouse to
        self._raw_prev: Pose | None = None
        self._clear()

    def _clear(self) -> None:
        self._candidate: Pose | None = None
        self._count = 0
        self.active: Pose = Pose.NONE
        self._behaviour = Behaviour.NONE
        self._pinch: _Pinch | None = None
        self._level: _Level | None = None
        self._pause_start: float | None = None
        self._pause_toggled = False

    # ------------------------------------------------------------------ public
    @property
    def behaviour(self) -> Behaviour:
        return self._behaviour

    @property
    def gesture(self) -> str:
        """Name for the overlay status chip."""
        if self.paused and self._behaviour is not Behaviour.PAUSE:
            return PAUSED
        if self._behaviour is Behaviour.LEFT_CLICK and self._pinch and self._pinch.dragging:
            return DRAG
        return DISPLAY[self._behaviour]

    def update(self, hand: HandObservation | None, t_ms: float) -> list[Intent]:
        """Classify one frame (with pinch hysteresis) and advance the state machine."""
        reading = read_hand(hand, self._raw_prev, self.cfg.classifier)
        self._raw_prev = reading.pose if reading else None
        return self.update_reading(reading, t_ms)

    def update_reading(self, reading: Reading | None, t_ms: float) -> list[Intent]:
        out: list[Intent] = []
        raw = reading.pose if reading else Pose.NONE
        if raw == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = raw, 1
        if raw != self.active and self._count >= self.cfg.hold_frames:
            out += self._exit(next_pose=raw)
            out += self._enter(raw, reading, t_ms)
        elif reading is not None and raw == self.active:
            out += self._frame(reading, t_ms)
        return out

    def reset(self) -> list[Intent]:
        """Stop whatever is in progress without clicking (gestures off, panic stop)."""
        out = release_intents() if self._pinch and self._pinch.dragging else []
        self._clear()
        self._raw_prev = None
        self._filter.reset()
        return out

    # ------------------------------------------------------------------ transitions
    def _behaviour_for(self, pose: Pose) -> Behaviour:
        b = (
            self.cfg.gesture_map.get(pose, Behaviour.NONE)
            if pose is not Pose.NONE
            else Behaviour.NONE
        )
        if self.paused and b is not Behaviour.PAUSE:
            return Behaviour.NONE
        return b

    def _exit(self, next_pose: Pose) -> list[Intent]:
        out: list[Intent] = []
        if self._behaviour is Behaviour.LEFT_CLICK and self._pinch:
            if self._pinch.dragging:
                out.append(_intent("input.mouse_up", button="left"))
            elif next_pose is not Pose.NONE:  # released, not lost: one click
                out.append(_intent("input.click", button="left", count=1))
        self._pinch = None
        self._level = None
        self._pause_start = None
        return out

    def _enter(self, pose: Pose, reading: Reading | None, t: float) -> list[Intent]:
        self.active = pose
        self._behaviour = b = self._behaviour_for(pose)
        if reading is None:
            return []
        if b is Behaviour.CURSOR:
            self._filter.reset()
            return self._move_cursor(reading.pointer, t)
        if b is Behaviour.LEFT_CLICK:
            self._filter.reset()
            anchor = self._filter(reading.pinch_point, t / 1000)
            start = self._cursor or self._to_screen(reading.pinch_point)
            self._pinch = _Pinch(start_t=t, anchor=anchor, start_px=start)
            return []
        if b is Behaviour.RIGHT_CLICK:
            return [_intent("input.click", button="right", count=1)]
        if b in (Behaviour.SCROLL, Behaviour.VOLUME, Behaviour.BRIGHTNESS):
            self._level = _Level(entry_y=reading.palm[1], last_t=t, last_emit_t=t)
            return []
        if b is Behaviour.PAUSE:
            self._pause_start, self._pause_toggled = t, False
        return []

    # ------------------------------------------------------------------ per frame
    def _frame(self, r: Reading, t: float) -> list[Intent]:
        b = self._behaviour
        if b is Behaviour.CURSOR:
            return self._move_cursor(r.pointer, t)
        if b is Behaviour.LEFT_CLICK and self._pinch:
            return self._pinch_frame(r, t)
        if self._level is not None and b in (
            Behaviour.SCROLL,
            Behaviour.VOLUME,
            Behaviour.BRIGHTNESS,
        ):
            return self._level_frame(b, r, t)
        if b is Behaviour.PAUSE and self._pause_start is not None and not self._pause_toggled:
            if t - self._pause_start >= self.cfg.pause_hold_ms:
                self.paused = not self.paused
                self._pause_toggled = True
        return []

    def _to_screen(self, pt: Point) -> tuple[int, int]:
        m = self.cfg.active_box_margin
        span = 1 - 2 * m
        w, h = self.cfg.screen_size
        u = _clamp((pt[0] - m) / span, 0.0, 1.0)
        v = _clamp((pt[1] - m) / span, 0.0, 1.0)
        return self._clamp_px(round(u * (w - 1)), round(v * (h - 1)))

    def _clamp_px(self, x: float, y: float) -> tuple[int, int]:
        # Stay one pixel off the edges: pyautogui's fail-safe fires in the screen corners.
        w, h = self.cfg.screen_size
        return int(_clamp(x, 1, w - 2)), int(_clamp(y, 1, h - 2))

    def _emit_move(self, px: tuple[int, int]) -> list[Intent]:
        if px == self._cursor:
            return []
        self._cursor = px
        return [_intent("input.move_to", x=px[0], y=px[1])]

    def _move_cursor(self, pointer: Point, t: float) -> list[Intent]:
        return self._emit_move(self._to_screen(self._filter(pointer, t / 1000)))

    def _pinch_frame(self, r: Reading, t: float) -> list[Intent]:
        p = self._pinch
        assert p is not None
        pt = self._filter(r.pinch_point, t / 1000)
        out: list[Intent] = []
        if not p.dragging:
            moved = math.dist(pt, p.anchor) > self.cfg.drag_move_threshold
            if t - p.start_t >= self.cfg.drag_hold_ms or moved:
                p.dragging = True
                out.append(_intent("input.mouse_down", button="left"))
        if p.dragging:
            span = 1 - 2 * self.cfg.active_box_margin
            w, h = self.cfg.screen_size
            dx = (pt[0] - p.anchor[0]) / span * (w - 1)
            dy = (pt[1] - p.anchor[1]) / span * (h - 1)
            out += self._emit_move(self._clamp_px(p.start_px[0] + dx, p.start_px[1] + dy))
        return out

    def _level_frame(self, b: Behaviour, r: Reading, t: float) -> list[Intent]:
        lv = self._level
        assert lv is not None
        dt = min(t - lv.last_t, MAX_FRAME_GAP_MS) / 1000
        lv.last_t = t
        offset = lv.entry_y - r.palm[1]  # image y points down: moving up is positive
        beyond = abs(offset) - self.cfg.deadzone
        if beyond <= 0:
            lv.accum = 0.0  # inside the deadzone: no change, and no leftover fraction
            return []
        speed = {
            Behaviour.SCROLL: self.cfg.scroll_speed,
            Behaviour.VOLUME: self.cfg.volume_speed,
            Behaviour.BRIGHTNESS: self.cfg.brightness_speed,
        }[b]
        lv.accum += math.copysign(beyond * speed * dt, offset)
        if t - lv.last_emit_t < 1000 / self.cfg.max_os_calls_per_s:
            return []
        steps = int(lv.accum)  # truncate toward zero; the fraction carries over
        if steps == 0:
            return []
        lv.accum -= steps
        lv.last_emit_t = t
        if b is Behaviour.SCROLL:
            direction = "up" if steps > 0 else "down"
            return [_intent("input.scroll", direction=direction, notches=abs(steps))]
        name = "volume.step" if b is Behaviour.VOLUME else "brightness.step"
        return [_intent(name, delta=steps)]
