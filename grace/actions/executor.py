"""Action executor: one worker thread runs every OS action, in order. ``submit`` never blocks.

Input threads (gesture camera loop) hand Intents to ``submit`` and go straight back to work, so a
slow OS call (brightness over WMI can take ~100 ms) never stalls them. When the newest pending
intent has the same action as the one being submitted, the two are merged instead of queued: cursor
moves keep the latest position, volume / brightness steps and scrolls add up. Only the tail is
merged, so order is preserved (a move is never reordered across a mouse_down).

The voice path needs each result to speak it, so it uses ``call`` (or the ``ExecutorRegistry``
facade, which looks like an ActionRegistry): the intent is queued like any other and the caller
waits for the result. Voice and gesture actions therefore run on the same thread, in order.

Risky actions are refused unless ``call(..., confirmed=True)`` is used, which only the
confirmation gate does (through the facade).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from grace.actions.registry import ActionRegistry, ActionResult, ConfirmationRequiredError
from grace.intent.types import Intent

log = logging.getLogger(__name__)

ResultCallback = Callable[[Intent, ActionResult], None]
BACKLOG_WARNING = 100


def _scroll_value(intent: Intent) -> int:
    n = int(intent.args.get("notches", 3))
    return n if intent.args.get("direction", "down") == "up" else -n


def merge(pending: Intent, new: Intent) -> Intent | None:
    """Combine two consecutive intents for the same action, or None if they can't merge.

    Returns an Intent with ``name == ""`` when the two cancel out (e.g. scroll up 2 + down 2).
    """
    if pending.name != new.name or pending.source != new.source:
        return None
    if new.name == "input.move_to":
        return new
    if new.name in ("volume.step", "brightness.step"):
        delta = int(pending.args.get("delta", 0)) + int(new.args.get("delta", 0))
        return Intent(new.name, {"delta": delta}, new.text, new.source) if delta else _NOTHING
    if new.name == "input.scroll":
        total = _scroll_value(pending) + _scroll_value(new)
        if not total:
            return _NOTHING
        args = {"direction": "up" if total > 0 else "down", "notches": abs(total)}
        return Intent(new.name, args, new.text, new.source)
    return None


_NOTHING = Intent("")

DEFAULT_CALL_TIMEOUT_S = 30.0


@dataclass
class _Call:
    """A queued intent whose caller is waiting for the result."""

    intent: Intent
    confirmed: bool
    done: threading.Event = field(default_factory=threading.Event)
    result: ActionResult | None = None


class ActionExecutor:
    def __init__(
        self,
        registry: ActionRegistry,
        on_result: ResultCallback | None = None,
        *,
        name: str = "grace-actions",
    ) -> None:
        self._registry = registry
        self._on_result = on_result
        self._name = name
        self._queue: deque[Intent | _Call] = deque()
        self._cond = threading.Condition()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self.executed = 0

    # ------------------------------------------------------------------ producer side
    def submit(self, intent: Intent) -> None:
        with self._cond:
            if self._queue and isinstance(self._queue[-1], Intent):
                merged = merge(self._queue[-1], intent)
                if merged is not None:
                    self._queue.pop()
                    if merged.name:
                        self._queue.append(merged)
                    self._cond.notify()
                    return
            self._queue.append(intent)
            if len(self._queue) == BACKLOG_WARNING:
                log.warning("action executor backlog: %d intents pending", len(self._queue))
            self._cond.notify()

    def call(
        self,
        intent: Intent,
        *,
        confirmed: bool = False,
        timeout: float = DEFAULT_CALL_TIMEOUT_S,
    ) -> ActionResult:
        """Queue ``intent`` behind everything already submitted and wait for its result.

        Runs inline when the executor thread isn't running (tests, shutdown) or when called from
        the executor thread itself (a handler can't wait for itself).
        """
        if self._thread is None or threading.current_thread() is self._thread:
            return self._execute(intent, confirmed)
        job = _Call(intent, confirmed)
        with self._cond:
            self._queue.append(job)
            self._cond.notify()
        if not job.done.wait(timeout):
            log.warning("executor: %s did not finish within %.0f s", intent.name, timeout)
            return ActionResult(False, "That is taking too long.")
        assert job.result is not None
        return job.result

    def pending(self) -> int:
        with self._cond:
            return len(self._queue)

    def clear_pending(self) -> int:
        """Drop queued fire-and-forget intents (panic stop). Waiting ``call``s still run."""
        with self._cond:
            kept = [item for item in self._queue if isinstance(item, _Call)]
            dropped = len(self._queue) - len(kept)
            self._queue = deque(kept)
        if dropped:
            log.info("executor: dropped %d pending intents", dropped)
        return dropped

    # ------------------------------------------------------------------ worker side
    def run_pending(self) -> int:
        """Run everything queued now, on the calling thread (tests, shutdown). Returns the count."""
        count = 0
        while True:
            with self._cond:
                if not self._queue:
                    return count
                item = self._queue.popleft()
            self._run(item)
            count += 1

    def _execute(self, intent: Intent, confirmed: bool = False) -> ActionResult:
        try:
            result = self._registry.execute(intent.name, intent.args, confirmed=confirmed)
        except ConfirmationRequiredError:
            log.error(
                "refused risky action %s from %s: needs confirmation", intent.name, intent.source
            )
            result = ActionResult(False, "That needs confirmation.")
        except Exception:  # noqa: BLE001 - the executor thread must survive anything
            log.exception("executor: %s crashed", intent.name)
            result = ActionResult(False, "Sorry, that didn't work.")
        self.executed += 1
        self._report(intent, result)
        return result

    def _report(self, intent: Intent, result: ActionResult) -> None:
        if self._on_result is not None:
            try:
                self._on_result(intent, result)
            except Exception:  # noqa: BLE001
                log.exception("executor: result callback failed")

    def _run(self, item: Intent | _Call) -> None:
        if isinstance(item, _Call):
            try:
                item.result = self._execute(item.intent, item.confirmed)
            finally:
                item.done.set()
        else:
            self._execute(item)

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stopping:
                    self._cond.wait()
                if not self._queue:  # stopping and drained
                    return
                item = self._queue.popleft()
            self._run(item)

    def start(self) -> ActionExecutor:
        if self._thread is None:
            self._stopping = False
            self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Finish what is queued (so a pending mouse_up still runs), then stop the thread."""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("action executor did not stop within %.1f s", timeout)
            self._thread = None


class ExecutorRegistry:
    """An ActionRegistry look-alike whose ``execute`` runs on the executor thread.

    Hand it to code written against the registry (voice dispatcher, confirmation gate, Gemini
    assistant): lookups go straight to the registry, execution is serialised with gestures.
    """

    def __init__(
        self, registry: ActionRegistry, executor: ActionExecutor, source: str = "grammar"
    ) -> None:
        self._registry = registry
        self._executor = executor
        self._source = source

    def execute(
        self, name: str, args: Mapping[str, Any] | None = None, *, confirmed: bool = False
    ) -> ActionResult:
        intent = Intent(name, dict(args or {}), source=self._source)  # type: ignore[arg-type]
        return self._executor.call(intent, confirmed=confirmed)

    def for_source(self, source: str) -> ExecutorRegistry:
        """The same facade, tagging its actions with another source (router, followup)."""
        if source == self._source:
            return self
        return ExecutorRegistry(self._registry, self._executor, source)

    def __contains__(self, name: object) -> bool:
        return name in self._registry

    def __iter__(self) -> Iterator[Any]:
        return iter(self._registry)

    def __getattr__(self, attr: str) -> Any:  # spec, names, is_risky, validate, ...
        return getattr(self._registry, attr)
