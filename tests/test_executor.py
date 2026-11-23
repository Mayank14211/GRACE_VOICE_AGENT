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


# --------------------------------------------------------------------------- S05: call / facade
def test_call_waits_behind_submitted_intents_on_the_executor_thread() -> None:
    registry, b = create_fake_registry()
    threads: list[str] = []
    ex = ActionExecutor(registry, lambda i, r: threads.append(threading.current_thread().name))
    ex.start()
    try:
        ex.submit(g("input.mouse_down", button="left"))
        ex.submit(g("input.mouse_up", button="left"))
        result = ex.call(Intent("volume.set", {"level": 30}))
    finally:
        ex.stop()
    assert result.ok and b.volume.level == 30
    # Voice and gesture ran on the same thread, in submission order.
    assert b.input.calls == [("mouse_down", "left"), ("mouse_up", "left")]
    assert set(threads) == {"grace-actions"} and len(threads) == 3


def test_call_runs_inline_when_not_started() -> None:
    registry, b = create_fake_registry()
    assert ActionExecutor(registry).call(Intent("volume.set", {"level": 10})).ok
    assert b.volume.level == 10


def test_call_refuses_risky_unless_confirmed() -> None:
    registry, b = create_fake_registry()
    ex = ActionExecutor(registry).start()
    try:
        assert not ex.call(Intent("system.shutdown")).ok
        assert b.power.calls == []
        assert ex.call(Intent("system.shutdown"), confirmed=True).ok
    finally:
        ex.stop()
    assert b.power.calls == ["shutdown"]


@pytest.mark.timeout(5)
def test_call_from_a_handler_on_the_executor_thread_does_not_deadlock() -> None:
    registry, _ = create_fake_registry()
    ex = ActionExecutor(registry)
    inner: list[bool] = []

    def on_result(intent: Intent, result: ActionResult) -> None:
        if intent.name == "input.click":
            inner.append(ex.call(Intent("volume.set", {"level": 5})).ok)

    ex._on_result = on_result
    ex.start()
    try:
        assert ex.call(g("input.click", button="left", count=1)).ok
    finally:
        ex.stop()
    assert inner == [True]


def test_clear_pending_drops_fire_and_forget_intents_only() -> None:
    registry, b = create_fake_registry()
    ex = ActionExecutor(registry)
    for x in range(3):
        ex.submit(g("input.move_to", x=x, y=0))
        ex.submit(g("input.click", button="left", count=1))
    assert ex.clear_pending() == 6
    assert ex.run_pending() == 0 and b.input.calls == []


def test_executor_registry_facade() -> None:
    from grace.actions.executor import ExecutorRegistry

    registry, b = create_fake_registry()
    sources: list[str] = []
    ex = ActionExecutor(registry, lambda i, r: sources.append(i.source)).start()
    try:
        facade = ExecutorRegistry(registry, ex, source="router")
        assert "volume.set" in facade and facade.is_risky("system.shutdown")
        assert facade.spec("volume.set").name == "volume.set"
        assert {s.name for s in facade} == set(registry.names())
        assert facade.execute("volume.set", {"level": 70}).ok
    finally:
        ex.stop()
    assert b.volume.level == 70 and sources == ["router"]


def test_release_all_releases_only_what_grace_holds() -> None:
    registry, b = create_fake_registry()
    first = registry.execute("input.release_all")
    assert first.ok and first.data["released"] == []
    registry.execute("input.mouse_down", {"button": "left"})
    registry.execute("input.mouse_down", {"button": "right"})
    registry.execute("input.mouse_up", {"button": "right"})
    b.input.calls.clear()
    result = registry.execute("input.release_all")
    assert result.data["released"] == ["left"] and b.input.calls == [("mouse_up", "left")]
    assert registry.execute("input.release_all").data["released"] == []
