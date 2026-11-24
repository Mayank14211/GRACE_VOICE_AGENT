"""Scripted STT engine for tests (and --dry-run style checks): no model, no audio decoding.

    stt = FakeStt(["open notepad", "volume 40"], partial_every=2)

Each utterance consumes all its chunks, yields a partial every ``partial_every`` chunks (the
growing prefix of the scripted text), then the next scripted text as the final. ``fail`` makes it
raise SttError (to test fallbacks). ``utterances`` records the samples of every call.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator

import numpy as np

from grace.input.stt.base import AudioClock, Partial, SttError, SttEvent


class FakeStt:
    streaming = True

    def __init__(
        self,
        texts: list[str] | Callable[[np.ndarray], str] | None = None,
        *,
        name: str = "fake",
        partial_every: int = 0,
        fail: str | None = None,
    ) -> None:
        self.name = name
        self._texts = texts if callable(texts) else list(texts or [])
        self.partial_every = partial_every
        self.fail = fail
        self.loads = 0
        self.utterances: list[np.ndarray] = []

    def load(self) -> None:
        self.loads += 1
        if self.fail:
            raise SttError(self.fail)

    def _next(self, audio: np.ndarray) -> str:
        if callable(self._texts):
            return self._texts(audio)
        return self._texts.pop(0) if self._texts else ""

    def transcribe_stream(self, chunks: Iterable[np.ndarray]) -> Iterator[SttEvent]:
        clock = AudioClock()
        parts: list[np.ndarray] = []
        partials = 0
        for i, chunk in enumerate(chunks, 1):
            clock.add(chunk)
            parts.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            if self.partial_every and i % self.partial_every == 0:
                partials += 1
                yield Partial(f"...{partials}", time.perf_counter())
        clock.end()
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        self.utterances.append(audio)
        if self.fail:
            raise SttError(self.fail)
        yield clock.final(self._next(audio), self.name, partials)
