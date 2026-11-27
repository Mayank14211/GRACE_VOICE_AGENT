"""Confirmation flow for risky actions, and the intent dispatcher in front of the registry.

risky intent -> "Confirm shutdown? Say yes or press Ctrl+Alt+Y." -> yes / hotkey within 8 s runs it;
"no", any other command, or the timeout cancels it. Only this module calls
``registry.execute(..., confirmed=True)``.

`say` is the feedback sink (console print + TTS for now; the overlay subtitle bar in S05).
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from grace.actions.registry import ActionRegistry, ActionResult
from grace.intent.types import Intent

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 8.0


class Cancellable(Protocol):
    def cancel(self) -> None: ...


TimerFactory = Callable[[float, Callable[[], None]], Cancellable]


def start_thread_timer(delay_s: float, callback: Callable[[], None]) -> Cancellable:
    timer = threading.Timer(delay_s, callback)
    timer.daemon = True
    timer.start()
    return timer


def for_source(registry: Any, source: str) -> Any:
    """``registry`` tagging its actions with ``source`` (ExecutorRegistry), or itself."""
    tag = getattr(registry, "for_source", None)
    return tag(source) if callable(tag) else registry


def format_hotkey(hotkey: str) -> str:
    """ "ctrl+alt+y" -> "Ctrl+Alt+Y"."""
    return "+".join(part.capitalize() for part in hotkey.split("+"))


@dataclass
class _Pending:
    intent: Intent
    label: str
    deadline: float
    token: int
    timer: Cancellable | None


class ConfirmationGate:
    """Holds at most one risky intent until it is confirmed, cancelled or times out."""

    def __init__(
        self,
        registry: ActionRegistry,
        *,
        say: Callable[[str], None] = print,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        confirm_hotkey: str = "ctrl+alt+y",
        clock: Callable[[], float] = time.monotonic,
        start_timer: TimerFactory | None = start_thread_timer,
    ) -> None:
        self._registry = registry
        self._say = say
        self.timeout_s = timeout_s
        self.confirm_hotkey = confirm_hotkey
        self._clock = clock
        self._start_timer = start_timer
        self._lock = threading.Lock()
        self._pending: _Pending | None = None
        self._tokens = itertools.count(1)

    @property
    def pending(self) -> Intent | None:
        with self._lock:
            return self._pending.intent if self._pending else None

    def request(self, intent: Intent) -> str | None:
        """Hold a risky intent and ask for confirmation. Returns the prompt (None if rejected)."""
        spec = self._registry.spec(intent.name)
        try:
            self._registry.validate(intent.name, intent.args)
        except ValidationError:
            self._say("I couldn't do that: invalid arguments.")
            return None
        label = spec.confirm_label or spec.name
        self.cancel(announce=False)  # a new request replaces an older one
        token = next(self._tokens)
        timer = (
            self._start_timer(self.timeout_s, lambda: self.expire(token))
            if self._start_timer
            else None
        )
        with self._lock:
            self._pending = _Pending(intent, label, self._clock() + self.timeout_s, token, timer)
        prompt = f"Confirm {label}? Say yes or press {format_hotkey(self.confirm_hotkey)}."
        log.info("confirmation requested for %s", intent.name)
        self._say(prompt)
        return prompt

    def _take(self) -> _Pending | None:
        """Remove and return the pending action (caller holds the lock)."""
        pending, self._pending = self._pending, None
        if pending and pending.timer:
            pending.timer.cancel()
        return pending

    def confirm(self) -> ActionResult | None:
        """Run the pending action if it hasn't expired. Safe to call from any thread."""
        with self._lock:
            if self._pending is None:
                return None
            expired = self._clock() >= self._pending.deadline
            pending = self._take()
        assert pending is not None
        if expired:
            self._say(f"Cancelled: {pending.label} was not confirmed in time.")
            return None
        log.info("confirmed %s", pending.intent.name)
        registry = for_source(self._registry, pending.intent.source)
        result = registry.execute(pending.intent.name, pending.intent.args, confirmed=True)
        self._say(result.message)
        return result

    def cancel(self, *, announce: bool = True) -> bool:
        with self._lock:
            pending = self._take()
        if pending and announce:
            self._say(f"Cancelled {pending.label}.")
        return pending is not None

    def expire(self, token: int | None = None) -> bool:
        """Cancel the pending action. With a token (timer callback) only if it is still the same
        request; without one (polling) only if the deadline has passed."""
        with self._lock:
            p = self._pending
            if p is None:
                return False
            if token is not None and p.token != token:
                return False  # stale timer from a request that was already replaced
            if token is None and self._clock() < p.deadline:
                return False
            self._take()
        self._say(f"Cancelled: {p.label} was not confirmed in time.")
        return True


class Dispatcher:
    """Routes an Intent: confirm/cancel -> gate; risky -> gate.request; other -> registry."""

    def __init__(
        self, registry: ActionRegistry, gate: ConfirmationGate, say: Callable[[str], None] = print
    ) -> None:
        self.registry = registry
        self.gate = gate
        self._say = say

    def handle(self, intent: Intent) -> ActionResult | None:
        if intent.name == "app.confirm":
            if self.gate.pending is None:
                self._say("There's nothing to confirm.")
                return None
            return self.gate.confirm()
        if intent.name == "app.cancel":
            if not self.gate.cancel():
                self._say("There's nothing to cancel.")
            return None
        if self.gate.pending is not None:
            self.gate.cancel()  # any other command abandons a pending risky action
        if intent.name not in self.registry:
            result = ActionResult(False, "I can't do that yet.")
        elif self.registry.is_risky(intent.name):
            self.gate.request(intent)
            return None
        else:
            result = for_source(self.registry, intent.source).execute(intent.name, intent.args)
        self._say(result.message)
        return result
