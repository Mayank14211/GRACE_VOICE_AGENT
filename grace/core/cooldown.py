"""Provider cooldowns: a provider that failed (HTTP 429, timeout, error) is skipped until its
cooldown ends (docs/PLAN.md: 60 s). Pure state with an injectable clock; thread-safe.

    cooldowns = Cooldowns(60.0)
    cooldowns.trip("groq")
    cooldowns.active("groq")      # True for the next 60 s
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class Cooldowns:
    def __init__(self, cooldown_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._until: dict[str, float] = {}

    def trip(self, name: str, seconds: float | None = None) -> None:
        with self._lock:
            self._until[name] = self._clock() + (self.cooldown_s if seconds is None else seconds)

    def active(self, name: str) -> bool:
        with self._lock:
            until = self._until.get(name)
            if until is None:
                return False
            if self._clock() >= until:
                del self._until[name]
                return False
            return True

    def remaining(self, name: str) -> float:
        with self._lock:
            until = self._until.get(name)
            return max(0.0, until - self._clock()) if until is not None else 0.0

    def clear(self, name: str) -> None:
        with self._lock:
            self._until.pop(name, None)
