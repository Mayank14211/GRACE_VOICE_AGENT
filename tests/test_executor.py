"""Action executor: order, merging, risky refusal, drain on stop, thread survives failures."""

from __future__ import annotations

import threading

import pytest

from grace.actions import create_fake_registry
from grace.actions.executor import ActionExecutor, merge
from grace.actions.registry import ActionResult
from grace.intent.types import Intent


def g(name: str, **args: object) -> Intent:
    return Intent(name, dict(args), source="gesture")


def test_runs_in_order_and_reports_results() -> None:
    registry, b = create_fake_registry()
    results: list[tuple[str, bool]] = []
    ex = ActionExecutor(registry, lambda i, r: results.append((i.name, r.ok)))
    ex.submit(g("input.mouse_down", button="left"))
    ex.submit(g("input.move_to", x=10, y=20))
    ex.submit(g("input.mouse_up", button="left"))
    assert ex.run_pending() == 3
    assert b.input.calls == [("mouse_down", "left"), ("move_to", 10, 20), ("mouse_up", "left")]
    assert results == [
        ("input.mouse_down", True),
        ("input.move_to", True),
        ("input.mouse_up", True),
    ]


def test_consecutive_moves_keep_only_the_latest() -> None:
    registry, b = create_fake_registry()
    ex = ActionExecutor(registry)
    for x in range(50):
        ex.submit(g("input.move_to", x=x, y=0))
    assert ex.pending() == 1
    ex.run_pending()
    assert b.input.calls == [("move_to", 49, 0)]


def test_merge_never_reorders_across_other_actions() -> None:
    registry, b = create_fake_registry()
    ex = ActionExecutor(registry)
    ex.submit(g("input.move_to", x=1, y=1))
    ex.submit(g("input.mouse_down", button="left"))
    ex.submit(g("input.move_to", x=2, y=2))
    ex.submit(g("input.move_to", x=3, y=3))
    ex.run_pending()
    assert b.input.calls == [("move_to", 1, 1), ("mouse_down", "left"), ("move_to", 3, 3)]


def test_steps_and_scrolls_add_up() -> None:
    registry, b = create_fake_registry()
    ex = ActionExecutor(registry)
    for d in (3, 4, -2):
        ex.submit(g("volume.step", delta=d))
    ex.submit(g("brightness.step", delta=5))
    ex.submit(g("brightness.step", delta=-5))  # cancels out
    ex.submit(g("input.scroll", direction="up", notches=2))
    ex.submit(g("input.scroll", direction="down", notches=5))
    ex.run_pending()
    assert b.volume.level == 55
    assert b.brightness.calls == []
    assert b.input.calls == [("scroll", -3 * 120)]


def test_merge_rules() -> None:
    assert merge(g("input.click", button="left"), g("input.click", button="left")) is None
    assert merge(g("volume.step", delta=1), Intent("volume.step", {"delta": 1})) is None  # source
    assert merge(g("volume.step", delta=1), g("volume.step", delta=-1)).name == ""  # type: ignore[union-attr]


def test_risky_actions_are_refused() -> None:
    registry, b = create_fake_registry()
    results: list[ActionResult] = []
    ex = ActionExecutor(registry, lambda i, r: results.append(r))
    ex.submit(Intent("system.shutdown", source="gesture"))
    ex.run_pending()
    assert b.power.calls == [] and results[0].ok is False


def test_thread_survives_callback_errors_and_drains_on_stop() -> None:
    registry, b = create_fake_registry()
    done = threading.Event()

    def on_result(intent: Intent, result: ActionResult) -> None:
        if intent.name == "input.mouse_up":
            done.set()
        raise RuntimeError("broken UI")

    ex = ActionExecutor(registry, on_result).start()
    ex.submit(g("input.mouse_down", button="left"))
    ex.submit(g("input.mouse_up", button="left"))
    assert done.wait(2.0)
    ex.submit(g("input.click", button="left", count=1))
    ex.stop()
    assert ("click", "left", 1) in b.input.calls
    assert ex.executed == 3


@pytest.mark.timeout(5)
def test_stop_without_start_and_double_stop() -> None:
    registry, _ = create_fake_registry()
    ex = ActionExecutor(registry)
    ex.stop()
    ex.start()
    ex.stop()
    ex.stop()
