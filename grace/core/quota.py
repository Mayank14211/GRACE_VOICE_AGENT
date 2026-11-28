"""Quota governor: counts requests, tokens, audio seconds and other units (image neurons, search
credits) per provider and model against free-tier limits, warns at 80 % and refuses at 100 %.
One store for the whole app: the LLM router, its tools and speech to text (Groq Whisper) share it.

    governor = QuotaGovernor({"groq:openai/gpt-oss-20b": Limits(rpm=30, rpd=1000, tpd=200_000)})
    if governor.allow("groq:openai/gpt-oss-20b"):
        ...call...
        governor.record("groq:openai/gpt-oss-20b", tokens=reply_tokens)
    governor.allow("groq:whisper-large-v3-turbo", audio_s=10)      # STT: audio seconds billed
    governor.record("groq:whisper-large-v3-turbo", audio_s=10)

Windows are fixed: the current minute, UTC hour, UTC day (free tiers reset at 00:00 UTC) and UTC
month.
Hour, day and month counters are saved to a small JSON file (logs/quota.json, git-ignored), so a
restart does not hand out a fresh daily allowance. Keys never contain secrets.

S06b decided against a second store (grace/input/stt/budget.py was the plan): speech to text uses
this governor, so STT and the router share one quota file and one 80 % warning path.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

WARN_AT = 0.8
DEFAULT_STATE = Path(__file__).resolve().parents[2] / "logs" / "quota.json"


@dataclass(frozen=True)
class Limits:
    """None = not limited. units_* are model-specific units (neurons, search credits);
    audio_s_* are seconds of audio as the provider bills them (Groq Whisper: at least 10 s per
    request)."""

    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    tpd: int | None = None
    units_per_day: int | None = None
    units_per_month: int | None = None
    audio_s_per_minute: int | None = None
    audio_s_per_hour: int | None = None
    audio_s_per_day: int | None = None


class Quota(Protocol):
    """What callers need. Real: QuotaGovernor; tests may pass NullQuota."""

    def allow(
        self, key: str, *, tokens: int = 0, units: float = 0.0, audio_s: float = 0.0
    ) -> bool: ...

    def record(
        self,
        key: str,
        *,
        requests: int = 1,
        tokens: int = 0,
        units: float = 0.0,
        audio_s: float = 0.0,
    ) -> None: ...


class NullQuota:
    def allow(self, key: str, *, tokens: int = 0, units: float = 0.0, audio_s: float = 0.0) -> bool:
        return True

    def record(
        self,
        key: str,
        *,
        requests: int = 1,
        tokens: int = 0,
        units: float = 0.0,
        audio_s: float = 0.0,
    ) -> None:
        pass


def _windows(now: float) -> tuple[str, str, str, str]:
    """(minute, day, month, hour) in UTC. The index is part of logs/quota.json: append only."""
    t = datetime.fromtimestamp(now, UTC)
    return (
        t.strftime("%Y-%m-%dT%H:%M"),
        t.strftime("%Y-%m-%d"),
        t.strftime("%Y-%m"),
        t.strftime("%Y-%m-%dT%H"),
    )


class QuotaGovernor:
    # counter -> (window index in _windows, limit attribute)
    _COUNTERS = {
        "rpm": (0, "rpm"),
        "rpd": (1, "rpd"),
        "tpm": (0, "tpm"),
        "tpd": (1, "tpd"),
        "upd": (1, "units_per_day"),
        "upm": (2, "units_per_month"),
        "asm": (0, "audio_s_per_minute"),
        "ash": (3, "audio_s_per_hour"),
        "asd": (1, "audio_s_per_day"),
    }

    def __init__(
        self,
        limits: Mapping[str, Limits],
        *,
        state_path: Path | None = None,
        clock: Callable[[], float] = time.time,
        on_warning: Callable[[str, str, float], None] | None = None,
    ) -> None:
        self.limits = dict(limits)
        self._path = state_path
        self._clock = clock
        self._on_warning = on_warning or (lambda key, counter, used: None)
        self._lock = threading.Lock()
        # key -> counter -> [window, value]
        self._counts: dict[str, dict[str, list]] = {}
        self._warned: set[tuple[str, str, str]] = set()
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        if self._path is None or not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("quota: %s unreadable; starting from zero", self._path.name)
            return
        if isinstance(data, dict):
            for key, counters in data.items():
                if isinstance(counters, dict):
                    self._counts[key] = {
                        c: list(v) for c, v in counters.items()
                        if c in self._COUNTERS and isinstance(v, list) and len(v) == 2
                    }  # fmt: skip

    def _save(self) -> None:
        if self._path is None:
            return
        keep = {  # minute counters don't survive a restart: not worth a write
            key: {c: v for c, v in counters.items() if self._COUNTERS[c][0] > 0}
            for key, counters in self._counts.items()
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(keep), encoding="utf-8")
        except OSError:
            log.warning("quota: could not save %s", self._path.name)

    # ------------------------------------------------------------------ counting
    def _value(self, key: str, counter: str, windows: tuple[str, ...]) -> float:
        window = windows[self._COUNTERS[counter][0]]
        entry = self._counts.get(key, {}).get(counter)
        return float(entry[1]) if entry and entry[0] == window else 0.0

    def used(self, key: str) -> dict[str, float]:
        """Current-window usage per counter (rpm, rpd, tpm, tpd, upd, upm, asm, ash, asd)."""
        with self._lock:
            windows = _windows(self._clock())
            return {c: self._value(key, c, windows) for c in self._COUNTERS}

    def allow(self, key: str, *, tokens: int = 0, units: float = 0.0, audio_s: float = 0.0) -> bool:
        """False when one more request (with ``tokens`` / ``units`` / ``audio_s``) would pass a
        limit."""
        limits = self.limits.get(key)
        if limits is None:
            return True
        with self._lock:
            windows = _windows(self._clock())
            extra = {"rpm": 1, "rpd": 1, "tpm": tokens, "tpd": tokens, "upd": units, "upm": units,
                     "asm": audio_s, "ash": audio_s, "asd": audio_s}  # fmt: skip
            for counter, (_, attr) in self._COUNTERS.items():
                limit = getattr(limits, attr)
                if limit is None:
                    continue
                value = self._value(key, counter, windows)
                # Token use is only known after a call: an exhausted token budget still blocks.
                if value + extra[counter] > limit or (not extra[counter] and value >= limit):
                    log.info("quota: %s would pass %s (%s)", key, counter, limit)
                    return False
        return True

    def record(
        self,
        key: str,
        *,
        requests: int = 1,
        tokens: int = 0,
        units: float = 0.0,
        audio_s: float = 0.0,
    ) -> None:
        add = {"rpm": requests, "rpd": requests, "tpm": tokens, "tpd": tokens, "upd": units,
               "upm": units, "asm": audio_s, "ash": audio_s, "asd": audio_s}  # fmt: skip
        warnings = []
        with self._lock:
            windows = _windows(self._clock())
            counters = self._counts.setdefault(key, {})
            limits = self.limits.get(key)
            for counter, (index, attr) in self._COUNTERS.items():
                if not add[counter]:
                    continue
                value = self._value(key, counter, windows) + add[counter]
                counters[counter] = [windows[index], value]
                limit = getattr(limits, attr) if limits else None
                token = (key, counter, windows[index])
                if limit and value >= WARN_AT * limit and token not in self._warned:
                    self._warned.add(token)
                    warnings.append((counter, value / limit))
            self._save()
        for counter, fraction in warnings:
            log.warning("quota: %s at %.0f%% of its %s limit", key, fraction * 100, counter)
            try:
                self._on_warning(key, counter, fraction)
            except Exception:  # noqa: BLE001 - a broken listener must not break counting
                log.exception("quota warning listener failed")


def build_governor(
    settings: object, *, state_path: Path | None = DEFAULT_STATE, **kw
) -> QuotaGovernor:
    """Limits from settings: router.quota per provider (keyed provider:model), images neurons
    per day, search credits per month, and Groq Whisper from stt.quota (keyed
    groq:<stt.groq_model>: requests and billed audio seconds)."""
    r = settings.router  # type: ignore[attr-defined]
    limits: dict[str, Limits] = {}
    for provider, q in r.quota.items():
        model = r.endpoint(provider).model
        limits[f"{provider}:{model}"] = Limits(rpm=q.rpm, rpd=q.rpd, tpm=q.tpm, tpd=q.tpd)
    img = settings.images  # type: ignore[attr-defined]
    limits[f"cloudflare:{img.model}"] = Limits(units_per_day=img.neurons_per_day or None)
    limits["tavily:search"] = Limits(
        units_per_month=settings.search.credits_per_month or None  # type: ignore[attr-defined]
    )
    limits[stt_quota_key(settings)] = stt_limits(settings)
    return QuotaGovernor(limits, state_path=state_path, **kw)


def stt_quota_key(settings: object) -> str:
    """The governor key of the cloud STT model (Groq counts limits per model)."""
    return f"groq:{settings.stt.groq_model}"  # type: ignore[attr-defined]


def stt_limits(settings: object) -> Limits:
    q = settings.stt.quota  # type: ignore[attr-defined]
    return Limits(
        rpm=q.rpm, rpd=q.rpd, audio_s_per_hour=q.audio_s_per_hour, audio_s_per_day=q.audio_s_per_day
    )
