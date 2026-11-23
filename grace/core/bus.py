"""Thread-safe event bus: workers publish, subscribers run on one bus thread, Qt gets a signal.

    bus = EventBus().start()
    bus.subscribe(HEARD, lambda e: print(e.data["text"]))
    bus.publish(HEARD, text="open notepad")          # never blocks, callable from any thread
    bridge = qt_bridge(bus, (STATE, HEARD, REPLY))   # main thread only; bridge.event -> Qt slots

``publish`` only puts the event on a ``queue.Queue``; the bus thread hands it to every subscriber
of that topic (and of ``ALL``) in publish order. A subscriber that raises is logged and skipped,
so one broken handler never stops the others. Subscribers must be quick: anything slow belongs in
its own worker (the action executor, the voice worker).

The Qt bridge is a QObject created on the main thread whose signal is emitted from the bus
thread; Qt delivers it as a queued call on the main thread, so overlay widgets are only ever
touched there. PySide6 is imported only when the bridge is created.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Topics. data keys in brackets.
STATE = "state"  # AppState changed [snapshot: StateSnapshot, changed: frozenset[str], cause_t]
HEARD = "heard"  # what GRACE heard [text]
PARTIAL = "partial"  # what GRACE is hearing, still changing [text]
REPLY = "reply"  # what GRACE answered [text]
ACTION = "action"  # an action finished [intent: Intent, result: ActionResult]
QUIT = "quit"  # the app should exit [reason]
ALL = "*"  # subscribe to every topic

Handler = Callable[["Event"], None]


@dataclass(frozen=True)
class Event:
    topic: str
    data: Mapping[str, Any] = field(default_factory=dict)
    t: float = 0.0  # time.monotonic() when published


_STOP = object()


class EventBus:
    def __init__(self, *, name: str = "grace-bus", clock: Callable[[], float] = time.monotonic):
        self._queue: queue.Queue[Any] = queue.Queue()
        self._subs: dict[str, list[Handler]] = {}
        self._lock = threading.Lock()
        self._name = name
        self._clock = clock
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ any thread
    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        """Call ``handler(event)`` for each event on ``topic``. Returns an unsubscribe function."""
        with self._lock:
            self._subs.setdefault(topic, []).append(handler)

        def unsubscribe() -> None:
            with self._lock:
                handlers = self._subs.get(topic, [])
                if handler in handlers:
                    handlers.remove(handler)

        return unsubscribe

    def publish(self, topic: str, **data: Any) -> Event:
        event = Event(topic, data, self._clock())
        self._queue.put(event)
        return event

    def pending(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------ dispatch
    def _handlers(self, topic: str) -> list[Handler]:
        with self._lock:
            return [*self._subs.get(topic, ()), *self._subs.get(ALL, ())]

    def dispatch(self, event: Event) -> None:
        for handler in self._handlers(event.topic):
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - one broken subscriber must not stop the rest
                log.exception("bus: handler for %r failed", event.topic)

    def pump(self, limit: int | None = None) -> int:
        """Dispatch queued events on the calling thread (tests, or a bus that isn't started)."""
        count = 0
        while limit is None or count < limit:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                continue
            self.dispatch(item)
            count += 1
        return count

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            self.dispatch(item)

    def start(self) -> EventBus:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Dispatch what is already queued, then stop the bus thread."""
        if self._thread is None:
            return
        self._queue.put(_STOP)
        self._thread.join(timeout)
        if self._thread.is_alive():
            log.warning("event bus did not stop within %.1f s", timeout)
        self._thread = None


def qt_bridge(bus: EventBus, topics: Iterable[str]) -> Any:
    """QObject with ``event = Signal(object)`` fed by ``bus`` for ``topics``.

    Create it on the Qt main thread; connect ``bridge.event`` to slots of main-thread objects and
    Qt queues every emit from the bus thread onto the main thread.
    """
    from PySide6.QtCore import QObject, Signal

    class _Bridge(QObject):
        event = Signal(object)

    bridge = _Bridge()
    unsubscribers = [bus.subscribe(topic, bridge.event.emit) for topic in topics]
    bridge.destroyed.connect(lambda *_: [u() for u in unsubscribers])
    return bridge
