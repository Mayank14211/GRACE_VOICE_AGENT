"""Gesture state machine: clicks, drags, scroll/volume/brightness, deadzone, pause, debounce."""

from __future__ import annotations

import ast
import random
from collections.abc import Iterable
from pathlib import Path

import pytest
from gesture_synth import make_hand

from grace.config import GestureSettings
from grace.input.gestures.classifier import Point, Pose, Reading
from grace.input.gestures.controller import (
    DRAG,
    PAUSED,
    Behaviour,
    ControllerConfig,
    GestureController,
)
from grace.intent.types import Intent

FRAME_MS = 1000 / 30
SCREEN = (1920, 1080)


class Driver:
    """Feeds readings at 30 fps and collects every intent."""

    def __init__(self, **cfg: object) -> None:
        cfg.setdefault("screen_size", SCREEN)
        self.c = GestureController(ControllerConfig(**cfg))  # type: ignore[arg-type]
        self.t = 0.0
        self.intents: list[Intent] = []

    def feed(
        self,
        pose: Pose | None,
        n: int = 1,
        *,
        pointer: Point = (0.5, 0.5),
        palm: Point = (0.5, 0.6),
        pinch: Point | None = None,
    ) -> list[Intent]:
        out: list[Intent] = []
        for _ in range(n):
            r = None if pose is None else Reading(pose, pointer, pinch or pointer, palm)
            out += self.c.update_reading(r, self.t)
            self.t += FRAME_MS
        self.intents += out
        return out

    def path(self, pose: Pose, points: Iterable[Point], field: str = "pointer") -> list[Intent]:
        out: list[Intent] = []
        for p in points:
            out += self.feed(pose, **{field: p})
        return out


def names(intents: Iterable[Intent]) -> list[str]:
    return [i.name for i in intents]


def clicks(intents: Iterable[Intent], button: str = "left") -> int:
    return sum(i.name == "input.click" and i.args["button"] == button for i in intents)


def frames_for(ms: float) -> int:
    return round(ms / FRAME_MS)


# --------------------------------------------------------------------------- debounce
def test_pose_needs_hold_frames_to_activate() -> None:
    d = Driver(hold_frames=4)
    d.feed(Pose.INDEX_UP, 3)
    assert d.c.active is Pose.NONE and d.intents == []
    d.feed(Pose.INDEX_UP, 1)
    assert d.c.active is Pose.INDEX_UP and d.c.gesture == "MOVE"
    assert names(d.intents) == ["input.move_to"]


def test_glitch_frames_never_switch_pose() -> None:
    d = Driver()
    d.feed(Pose.INDEX_UP, 10)
    for _ in range(20):  # 3-frame glitches of other poses in between
        d.feed(Pose.PINCH_INDEX, 3)
        d.feed(Pose.INDEX_UP, 2)
    assert clicks(d.intents) == 0 and d.c.active is Pose.INDEX_UP


# --------------------------------------------------------------------------- left click
def test_exactly_one_click_per_pinch() -> None:
    d = Driver()
    d.feed(Pose.INDEX_UP, 10)
    for _ in range(5):
        d.feed(Pose.PINCH_INDEX, frames_for(200))  # well under drag_hold_ms
        d.feed(Pose.INDEX_UP, 10)
    assert clicks(d.intents) == 5
    assert "input.mouse_down" not in names(d.intents)


def test_flickering_pinch_is_still_one_click() -> None:
    d = Driver()
    d.feed(Pose.INDEX_UP, 10)
    d.feed(Pose.PINCH_INDEX, 4)
    d.feed(Pose.INDEX_UP, 2)  # tracker dropout mid-pinch, shorter than hold_frames
    d.feed(Pose.PINCH_INDEX, 3)
    d.feed(None, 1)
    d.feed(Pose.PINCH_INDEX, 2)
    d.feed(Pose.INDEX_UP, 8)
    assert clicks(d.intents) == 1


def test_noisy_real_hands_near_the_threshold_click_once() -> None:
    """End to end through the classifier: hysteresis + debounce absorb a wobbling pinch."""
    from test_gesture_classifier import _with_pinch_ratio

    c = GestureController(ControllerConfig(screen_size=SCREEN))
    rng = random.Random(1)
    t, out = 0.0, []
    seq = [make_hand(Pose.INDEX_UP, "Right")] * 10
    seq += [_with_pinch_ratio(0.12)] * 3
    seq += [_with_pinch_ratio(rng.uniform(0.18, 0.31)) for _ in range(9)]  # wobbles across on
    seq += [make_hand(Pose.INDEX_UP, "Right")] * 10
    for hand in seq:
        out += c.update(hand, t)
        t += FRAME_MS
    assert clicks(out) == 1


def test_no_click_while_moving_the_cursor() -> None:
    d = Driver()
    rng = random.Random(3)
    pts = [(0.2 + i * 0.01 + rng.gauss(0, 0.004), 0.5 + rng.gauss(0, 0.004)) for i in range(60)]
    out = d.path(Pose.INDEX_UP, pts)
    assert clicks(out) == 0 and "input.mouse_down" not in names(out)
    assert names(out).count("input.move_to") > 30


def test_cursor_freezes_during_a_click_pinch() -> None:
    d = Driver()
    d.feed(Pose.INDEX_UP, 10, pointer=(0.5, 0.5))
    before = d.c._cursor
    out = d.feed(Pose.PINCH_INDEX, 6, pointer=(0.52, 0.53), pinch=(0.5, 0.5))
    assert "input.move_to" not in names(out) and d.c._cursor == before


def test_hold_becomes_drag() -> None:
    d = Driver(drag_hold_ms=400)
    d.feed(Pose.INDEX_UP, 10)
    d.feed(Pose.PINCH_INDEX, 4, pinch=(0.5, 0.5))  # activation
    held = d.feed(Pose.PINCH_INDEX, frames_for(450), pinch=(0.5, 0.5))
    assert names(held).count("input.mouse_down") == 1 and d.c.gesture == DRAG
    moved = d.path(Pose.PINCH_INDEX, [(0.5 + i * 0.01, 0.5) for i in range(1, 16)], "pinch")
    xs = [i.args["x"] for i in moved if i.name == "input.move_to"]
    assert xs and xs == sorted(xs) and xs[-1] > xs[0]
    released = d.feed(Pose.OPEN_PALM, 6)
    assert names(released) == ["input.mouse_up"]
    assert clicks(d.intents) == 0


def test_pinch_and_move_is_a_drag_not_a_click() -> None:
    d = Driver(drag_hold_ms=400, drag_move_threshold=0.04)
    d.feed(Pose.INDEX_UP, 10)
    d.feed(Pose.PINCH_INDEX, 4, pinch=(0.5, 0.5))
    d.path(Pose.PINCH_INDEX, [(0.5 + i * 0.02, 0.5) for i in range(1, 6)], "pinch")  # ~170 ms
    d.feed(Pose.INDEX_UP, 6)
    assert clicks(d.intents) == 0
    assert names(d.intents).count("input.mouse_down") == 1
    assert names(d.intents).count("input.mouse_up") == 1


def test_losing_the_hand_never_clicks_and_ends_a_drag() -> None:
    d = Driver()
    d.feed(Pose.PINCH_INDEX, 6)
    d.feed(None, 6)
    assert clicks(d.intents) == 0
    d.feed(Pose.PINCH_INDEX, 4 + frames_for(500))
    assert names(d.intents).count("input.mouse_down") == 1
    d.feed(None, 6)
    assert names(d.intents)[-1] == "input.mouse_up" and clicks(d.intents) == 0


def test_reset_releases_a_drag_without_clicking() -> None:
    d = Driver()
    d.feed(Pose.PINCH_INDEX, 4 + frames_for(500))
    assert d.c.reset() == [Intent("input.mouse_up", {"button": "left"}, source="gesture")]
    assert d.c.active is Pose.NONE
    d.feed(Pose.PINCH_INDEX, 6)
    assert d.c.reset() == []  # not dragging: nothing to release, and no click


def test_right_click_once_per_activation() -> None:
    d = Driver()
    d.feed(Pose.PINCH_MIDDLE, 4 + frames_for(1500))
    d.feed(Pose.OPEN_PALM, 5)
    d.feed(Pose.PINCH_MIDDLE, 6)
    assert clicks(d.intents, "right") == 2 and clicks(d.intents, "left") == 0


# --------------------------------------------------------------------------- cursor mapping
def test_inner_box_maps_to_full_screen() -> None:
    w, h = SCREEN
    d = Driver(active_box_margin=0.15, cursor_beta=0.0)

    def at(p: Point) -> tuple[int, int]:
        d.c.reset()
        d.feed(Pose.INDEX_UP, 4, pointer=p)
        return d.c._cursor  # type: ignore[return-value]

    assert at((0.5, 0.5)) == (round(0.5 * (w - 1)), round(0.5 * (h - 1)))
    assert at((0.15, 0.15)) == (1, 1)  # one pixel in: never the fail-safe corner
    assert at((0.85, 0.85)) == (w - 2, h - 2)
    assert at((0.0, 1.0)) == (1, h - 2)  # outside the box clamps
    assert at((0.3, 0.5))[0] < at((0.7, 0.5))[0]  # frame is pre-mirrored: right is right


def test_one_euro_steadies_a_still_hand() -> None:
    rng = random.Random(7)
    noisy = [(0.5 + rng.gauss(0, 0.006), 0.5 + rng.gauss(0, 0.006)) for _ in range(90)]
    smooth = Driver()
    smooth.path(Pose.INDEX_UP, noisy)
    raw = Driver(cursor_min_cutoff=30.0, cursor_beta=0.0)  # effectively unfiltered
    raw.path(Pose.INDEX_UP, noisy)

    def spread(d: Driver) -> int:
        xs = [i.args["x"] for i in d.intents[-60:] if i.name == "input.move_to"] or [0]
        return max(xs) - min(xs)

    assert spread(smooth) < spread(raw) / 3


# --------------------------------------------------------------------------- level controls
LEVELS = [
    (Pose.TWO_UP, "input.scroll"),
    (Pose.THUMB_INDEX_MIDDLE, "volume.step"),
    (Pose.THREE_UP, "brightness.step"),
]


def signed(i: Intent) -> int:
    if i.name == "input.scroll":
        return i.args["notches"] * (1 if i.args["direction"] == "up" else -1)
    return i.args["delta"]


@pytest.mark.parametrize(("pose", "action"), LEVELS)
def test_deadzone_yields_zero_change(pose: Pose, action: str) -> None:
    d = Driver(deadzone=0.05)
    d.feed(pose, 4, palm=(0.5, 0.6))
    out = d.feed(pose, frames_for(3000), palm=(0.5, 0.6 - 0.045))  # inside the deadzone
    out += d.feed(pose, frames_for(3000), palm=(0.5, 0.6 + 0.045))
    assert out == []


@pytest.mark.parametrize(("pose", "action"), LEVELS)
def test_change_is_proportional_to_offset(pose: Pose, action: str) -> None:
    totals = {}
    for offset in (0.1, 0.2):
        d = Driver(deadzone=0.05)
        d.feed(pose, 4, palm=(0.5, 0.6))
        up = d.feed(pose, frames_for(2000), palm=(0.5, 0.6 - offset))
        assert up and {i.name for i in up} == {action}
        totals[offset] = sum(signed(i) for i in up)
    assert totals[0.1] > 0
    # beyond the deadzone: 0.05 vs 0.15 -> about 3x
    assert 2.4 < totals[0.2] / totals[0.1] < 3.6
    d = Driver()
    d.feed(pose, 4, palm=(0.5, 0.6))
    down = d.feed(pose, frames_for(1000), palm=(0.5, 0.8))
    assert down and all(signed(i) < 0 for i in down)


@pytest.mark.parametrize(("pose", "action"), LEVELS)
def test_os_calls_are_rate_limited(pose: Pose, action: str) -> None:
    d = Driver(max_os_calls_per_s=10, scroll_speed=1000, volume_speed=2000, brightness_speed=2000)
    d.feed(pose, 4, palm=(0.5, 0.6))
    stamps = []
    for _ in range(frames_for(3000)):
        if d.feed(pose, palm=(0.5, 0.2)):
            stamps.append(d.t - FRAME_MS)
    assert len(stamps) <= 31
    assert min(b - a for a, b in zip(stamps, stamps[1:], strict=False)) >= 100 - 1e-6


def test_scroll_entry_point_resets_on_pose_switch() -> None:
    d = Driver()
    d.feed(Pose.TWO_UP, 4, palm=(0.5, 0.6))
    moved = d.feed(Pose.TWO_UP, frames_for(1000), palm=(0.5, 0.4))
    assert moved and all(i.args["direction"] == "up" for i in moved)
    d.feed(Pose.OPEN_PALM, 5, palm=(0.5, 0.4))  # neutral: entry point forgotten
    again = d.feed(Pose.TWO_UP, 4 + frames_for(2000), palm=(0.5, 0.4))
    assert again == []  # the new entry point is where the hand is now
    # same without going through neutral: any pose switch resets
    d.feed(Pose.INDEX_UP, 5)
    assert d.feed(Pose.TWO_UP, 4 + frames_for(1000), palm=(0.5, 0.3)) == []


def test_level_ignores_glitch_frames() -> None:
    d = Driver()
    d.feed(Pose.THUMB_INDEX_MIDDLE, 4, palm=(0.5, 0.6))
    for _ in range(30):  # palm jumps on 2-frame glitches only
        d.feed(Pose.OPEN_PALM, 2, palm=(0.5, 0.1))
        d.feed(Pose.THUMB_INDEX_MIDDLE, 2, palm=(0.5, 0.6))
    assert d.intents == []


# --------------------------------------------------------------------------- pause
def test_fist_pause_and_resume() -> None:
    d = Driver(pause_hold_ms=1000)
    d.feed(Pose.FIST, 4 + frames_for(900))
    assert not d.c.paused
    d.feed(Pose.FIST, frames_for(200))
    assert d.c.paused
    d.feed(Pose.FIST, frames_for(3000))  # still the same hold: toggles only once
    assert d.c.paused
    d.feed(Pose.INDEX_UP, 10)
    d.feed(Pose.PINCH_INDEX, 6)
    d.feed(Pose.TWO_UP, 4 + frames_for(1000), palm=(0.5, 0.1))
    d.feed(Pose.INDEX_UP, 6)
    assert d.intents == [] and d.c.gesture == PAUSED
    d.feed(Pose.FIST, 4 + frames_for(1100))
    assert not d.c.paused
    d.feed(Pose.INDEX_UP, 5)
    assert names(d.intents) == ["input.move_to"]


def test_short_fist_does_not_pause() -> None:
    d = Driver()
    d.feed(Pose.FIST, 4 + frames_for(600))
    d.feed(Pose.OPEN_PALM, 5)
    assert not d.c.paused


# --------------------------------------------------------------------------- map + names
def test_gesture_names_for_the_chip() -> None:
    d = Driver()
    seen = {}
    for pose in (
        Pose.INDEX_UP,
        Pose.TWO_UP,
        Pose.THREE_UP,
        Pose.THUMB_INDEX_MIDDLE,
        Pose.OPEN_PALM,
        Pose.FIST,
    ):
        d.feed(pose, 5)
        seen[pose] = d.c.gesture
    d.feed(None, 5)
    assert seen == {
        Pose.INDEX_UP: "MOVE",
        Pose.TWO_UP: "SCROLL",
        Pose.THREE_UP: "BRIGHTNESS",
        Pose.THUMB_INDEX_MIDDLE: "VOLUME",
        Pose.OPEN_PALM: "READY",
        Pose.FIST: "FIST",
    }
    assert d.c.gesture == "NONE"


def test_map_from_settings() -> None:
    gs = GestureSettings.model_validate(
        {"map": {"three_up": "none", "fist": "volume"}, "hold_frames": 2}
    )
    cfg = ControllerConfig.from_settings(gs, SCREEN)
    assert cfg.gesture_map[Pose.THREE_UP] is Behaviour.NONE
    assert cfg.gesture_map[Pose.FIST] is Behaviour.VOLUME
    assert cfg.gesture_map[Pose.INDEX_UP] is Behaviour.CURSOR  # untouched default
    assert cfg.hold_frames == 2 and cfg.classifier.pinch_on == gs.pinch_on
    d = Driver(gesture_map=cfg.gesture_map)
    d.feed(Pose.THREE_UP, 4, palm=(0.5, 0.6))
    assert d.feed(Pose.THREE_UP, 30, palm=(0.5, 0.2)) == []


def test_every_intent_is_a_gesture_intent_for_a_registered_action() -> None:
    from grace.actions import create_fake_registry

    registry, _ = create_fake_registry()
    d = Driver()
    d.feed(Pose.INDEX_UP, 10)
    d.feed(Pose.PINCH_INDEX, 5)
    d.feed(Pose.INDEX_UP, 5)
    d.feed(Pose.PINCH_INDEX, 4 + frames_for(600))
    d.feed(Pose.OPEN_PALM, 5)
    d.feed(Pose.PINCH_MIDDLE, 5)
    for pose, _ in LEVELS:
        d.feed(pose, 4, palm=(0.5, 0.6))
        d.feed(pose, 20, palm=(0.5, 0.3))
    assert {i.source for i in d.intents} == {"gesture"}
    assert {i.name for i in d.intents} >= {
        "input.move_to", "input.click", "input.mouse_down", "input.mouse_up",
        "input.scroll", "volume.step", "brightness.step",
    }  # fmt: skip
    for i in d.intents:
        assert i.name in registry and not registry.is_risky(i.name)
        assert registry.execute(i.name, i.args).ok, i


def test_gesture_modules_never_sleep() -> None:
    pkg = Path(__file__).parent.parent / "grace" / "input" / "gestures"
    for path in pkg.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr == "sleep":
                pytest.fail(f"{path.name}:{node.lineno} calls sleep")
            if isinstance(node, ast.ImportFrom) and any(a.name == "sleep" for a in node.names):
                pytest.fail(f"{path.name}:{node.lineno} imports sleep")
