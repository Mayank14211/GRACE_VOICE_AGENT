"""JSONL metrics log: one JSON object per line, appended and flushed as it happens.

    metrics = MetricsLog(Path("logs/metrics/session.jsonl"))
    metrics.write("toggle", what="mic", on=False, latency_ms=12.4)

Every row gets ``t`` (wall clock, seconds) and ``kind``. String values are passed through the
secret redaction filter, and callers must not log typed text or clipboard contents (log the
action name and ok flag, not its args). S10 builds the report on top of these rows.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Protocol

from grace.redaction import redact

log = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "logs" / "metrics"


class Metrics(Protocol):
    def write(self, kind: str, **fields: Any) -> None: ...

    def close(self) -> None: ...


class NullMetrics:
    def write(self, kind: str, **fields: Any) -> None:
        pass

    def close(self) -> None:
        pass


def _clean(value: Any, secrets: Iterable[str]) -> Any:
    if isinstance(value, str):
        return redact(value, secrets)
    if isinstance(value, dict):
        return {str(k): _clean(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_clean(v, secrets) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value), secrets)


class MetricsLog:
    def __init__(
        self,
        path: Path,
        *,
        secrets: Iterable[str] = (),
        clock: Callable[[], float] = time.time,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file: IO[str] | None = path.open("a", encoding="utf-8")
        self._secrets = tuple(secrets)
        self._clock = clock
        self._lock = threading.Lock()

    def write(self, kind: str, **fields: Any) -> None:
        row = {"t": round(self._clock(), 3), "kind": kind, **_clean(fields, self._secrets)}
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            if self._file is None:
                return
            try:
                self._file.write(line + "\n")
                self._file.flush()
            except OSError:
                log.exception("metrics: write failed")

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


def session_path(directory: Path = DEFAULT_DIR, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return directory / f"session-{stamp}.jsonl"
