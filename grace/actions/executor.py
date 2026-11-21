"""Action executor: one worker thread runs every OS action, in order. ``submit`` never blocks.

Input threads (gesture camera loop now, voice and hotkeys from S05) hand Intents to ``submit`` and
go straight back to work, so a slow OS call (brightness over WMI can take ~100 ms) never stalls
them. When the newest pending intent has the same action as the one being submitted, the two are
merged instead of queued: cursor moves keep the latest position, volume / brightness steps and
scrolls add up. Only the tail is merged, so order is preserved (a move is never reordered across
a mouse_down).

Risky actions are refused here: they must go through the confirmation gate, never an executor.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable

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
        self._queue: deque[Intent] = deque()
        self._cond = threading.Condition()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self.executed = 0

    # ------------------------------------------------------------------ producer side
    def submit(self, intent: Intent) -> None:
        with self._cond:
            if self._queue:
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

    def pending(self) -> int:
        with self._cond:
            return len(self._queue)

    # ------------------------------------------------------------------ worker side
    def run_pending(self) -> int:
        """Run everything queued now, on the calling thread (tests, shutdown). Returns the count."""
        count = 0
        while True:
            with self._cond:
                if not self._queue:
                    return count
                intent = self._queue.popleft()
            self._run(intent)
            count += 1

    def _run(self, intent: Intent) -> None:
        try:
            result = self._registry.execute(intent.name, intent.args)
        except ConfirmationRequiredError:
            log.error(
                "refused risky action %s from %s: needs confirmation", intent.name, intent.source
            )
            result = ActionResult(False, "That needs confirmation.")
        except Exception:  # noqa: BLE001 - the executor thread must survive anything
            log.exception("executor: %s crashed", intent.name)
            result = ActionResult(False, "Sorry, that didn't work.")
        self.executed += 1
        if self._on_result is not None:
            try:
                self._on_result(intent, result)
            except Exception:  # noqa: BLE001
                log.exception("executor: result callback failed")

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stopping:
                    self._cond.wait()
                if not self._queue:  # stopping and drained
                    return
                intent = self._queue.popleft()
            self._run(intent)

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
