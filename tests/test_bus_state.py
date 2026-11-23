"""core/bus.py and core/state.py: ordering, isolation, threads, snapshots, change events."""

from __future__ import annotations

import threading

import pytest

from grace.core.bus import ALL, HEARD, REPLY, STATE, Event, EventBus
from grace.core.state import AppState, StateSnapshot


# --------------------------------------------------------------------------- bus
def test_publish_then_pump_delivers_in_order() -> None:
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(HEARD, lambda e: seen.append(e.data["text"]))
    for word in ("one", "two", "three"):
        bus.publish(HEARD, text=word)
    assert seen == []  # publish never runs handlers itself
    assert bus.pump() == 3
    assert seen == ["one", "two", "three"]


def test_topics_are_separate_and_all_sees_everything() -> None:
    bus = EventBus()
    heard: list[Event] = []
    everything: list[str] = []
    bus.subscribe(HEARD, heard.append)
    bus.subscribe(ALL, lambda e: everything.append(e.topic))
    bus.publish(HEARD, text="a")
    bus.publish(REPLY, text="b")
    bus.pump()
    assert [e.topic for e in heard] == [HEARD]
    assert everything == [HEARD, REPLY]


def test_a_failing_handler_does_not_stop_the_others(caplog: pytest.LogCaptureFixture) -> None:
    bus = EventBus()
    seen: list[str] = []

    def broken(_: Event) -> None:
        raise RuntimeError("boom")

    bus.subscribe(HEARD, broken)
    bus.subscribe(HEARD, lambda e: seen.append(e.data["text"]))
    bus.publish(HEARD, text="still delivered")
    bus.pump()
    assert seen == ["still delivered"]
    assert "handler for 'heard' failed" in caplog.text


def test_unsubscribe() -> None:
    bus = EventBus()
    seen: list[Event] = []
    off = bus.subscribe(HEARD, seen.append)
    off()
    off()  # twice is fine
    bus.publish(HEARD, text="x")
    bus.pump()
    assert seen == []


def test_event_carries_publish_time() -> None:
    bus = EventBus(clock=lambda: 12.5)
    assert bus.publish(HEARD, text="x").t == 12.5


def test_bus_thread_delivers_from_many_publishers() -> None:
    bus = EventBus().start()
    got: list[int] = []
    done = threading.Event()
    lock = threading.Lock()

    def handler(e: Event) -> None:
        with lock:
            got.append(e.data["n"])
            if len(got) == 400:
                done.set()

    bus.subscribe(HEARD, handler)
    threads = [
        threading.Thread(target=lambda k=k: [bus.publish(HEARD, n=k * 100 + i) for i in range(100)])
        for k in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert done.wait(5)
    bus.stop()
    assert sorted(got) == list(range(400))
    # Per publisher, order is preserved.
    for k in range(4):
        mine = [n for n in got if n // 100 == k]
        assert mine == sorted(mine)


def test_stop_drains_queued_events() -> None:
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(HEARD, seen.append)
    bus.start()
    for _ in range(50):
        bus.publish(HEARD, text="x")
    bus.stop()
    assert len(seen) == 50
    bus.stop()  # idempotent


# --------------------------------------------------------------------------- state
def test_defaults_and_initial_values() -> None:
    s = AppState(mic_on=False, provider="gemini")
    snap = s.snapshot()
    assert snap == StateSnapshot(mic_on=False, provider="gemini")
    assert not s.gesture_on and s.mode == "command"
    with pytest.raises(TypeError):
        AppState(volume=3)


def test_update_publishes_only_real_changes() -> None:
    bus = EventBus()
    s = AppState(bus)
    events: list[Event] = []
    bus.subscribe(STATE, events.append)
    assert s.update(mic_on=True) == frozenset()  # already on: no event
    assert s.update(cause_t=1.5, mic_on=False, gesture="READY") == {"mic_on", "gesture"}
    bus.pump()
    [e] = events
    assert e.data["changed"] == {"mic_on", "gesture"}
    assert e.data["cause_t"] == 1.5
    assert e.data["snapshot"].mic_on is False and e.data["snapshot"].gesture == "READY"
    with pytest.raises(TypeError):
        s.update(colour="red")


def test_snapshots_are_immutable_values() -> None:
    s = AppState()
    before = s.snapshot()
    s.update(gesture_on=True)
    assert before.gesture_on is False and s.snapshot().gesture_on is True
    with pytest.raises(AttributeError):
        before.mic_on = False  # type: ignore[misc]


def test_toggle_is_atomic_under_threads() -> None:
    s = AppState()
    barrier = threading.Barrier(8)

    def flip() -> None:
        barrier.wait()
        for _ in range(125):
            s.toggle("gesture_on")

    threads = [threading.Thread(target=flip) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert s.gesture_on is False  # 1000 flips: back where it started
    with pytest.raises(TypeError):
        s.toggle("gesture")  # not a boolean


def test_turns_keep_the_last_n() -> None:
    bus = EventBus()
    s = AppState(bus, max_turns=3, clock=lambda: 7.0)
    for i in range(5):
        s.add_turn("user", f"command {i}")
    s.add_turn("grace", "   ")  # blank: ignored
    turns = s.snapshot().turns
    assert [t.text for t in turns] == ["command 2", "command 3", "command 4"]
    assert turns[0].t == 7.0
    assert bus.pump() == 5
