"""Speech-to-text interface: audio chunks in, partial texts and one final text out, with timings.

    for event in engine.transcribe_stream(chunks):   # chunks: float32 16 kHz frames, any size
        if isinstance(event, Partial): overlay(event.text)
        else: grammar(event.text); event.latency_ms   # end of audio -> final text

The chunk iterable ends when the utterance ends (VAD end of speech, push-to-talk release), so
``Final.latency_ms`` is exactly "end of speech decided -> final text", the M5 acceptance number.
A streaming engine (Moonshine) yields partials while audio arrives; a batch engine (faster-whisper,
Groq) only a final. Every engine yields exactly one ``Final``, last, even for empty audio, and
raises ``SttError`` when it cannot work (missing model or package, network error): callers fall
back (``FallbackEngine``) instead of crashing.

``select_engines`` is the pure selection rule from docs/PLAN.md:
English -> moonshine; Hindi / auto-detect, or dictation with dictation_accurate -> groq if it is
usable (key set, not cooling down after an error), else whisper_local.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Partial:
    text: str
    t: float  # time.perf_counter()


@dataclass(frozen=True)
class Final:
    text: str
    engine: str
    audio_s: float
    t_first_audio: float  # perf_counter when the first chunk arrived (0 if none)
    t_end_of_audio: float  # perf_counter when the chunk stream ended
    t_final: float  # perf_counter when the text was ready
    partials: int = 0
    fallback_from: str | None = None  # engine that failed before this one answered

    @property
    def latency_ms(self) -> float:
        """End of audio (end of speech decided) -> final text."""
        return (self.t_final - self.t_end_of_audio) * 1000


SttEvent = Partial | Final


class SttError(RuntimeError):
    """The engine cannot transcribe (not installed, model missing, network). Safe to show."""


class SttEngine(Protocol):
    name: str
    streaming: bool

    def load(self) -> None:
        """Load the model now (idempotent). Raises SttError if the engine can't work."""
        ...

    def transcribe_stream(self, chunks: Iterable[np.ndarray]) -> Iterator[SttEvent]: ...


class AudioClock:
    """Counts samples and remembers when the first chunk came and the stream ended."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self.samples = 0
        self.t_first = 0.0
        self.t_end = 0.0

    def add(self, chunk: np.ndarray) -> None:
        if self.samples == 0:
            self.t_first = self._clock()
        self.samples += len(chunk)

    def end(self) -> float:
        self.t_end = self._clock()
        return self.t_end

    def final(self, text: str, engine: str, partials: int = 0) -> Final:
        return Final(
            text=" ".join(text.split()),
            engine=engine,
            audio_s=self.samples / SAMPLE_RATE,
            t_first_audio=self.t_first,
            t_end_of_audio=self.t_end or self._clock(),
            t_final=self._clock(),
            partials=partials,
        )


class BatchEngine:
    """Base for engines that need the whole utterance first (faster-whisper, Groq)."""

    name = "batch"
    streaming = False

    def load(self) -> None:  # pragma: no cover - overridden
        pass

    def _transcribe(self, audio: np.ndarray) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def transcribe_stream(self, chunks: Iterable[np.ndarray]) -> Iterator[SttEvent]:
        self.load()
        clock = AudioClock()
        parts: list[np.ndarray] = []
        for chunk in chunks:
            clock.add(chunk)
            parts.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
        clock.end()
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        text = self._transcribe(audio) if len(audio) else ""
        yield clock.final(text, self.name)


class FallbackEngine:
    """Try ``primary``; if it raises SttError, run ``fallback`` on the same audio.

    The audio is kept while the primary reads it, so nothing the user said is lost. Timings of
    the fallback's Final are taken from the real stream (when audio started and ended), so its
    latency includes the time the primary wasted. ``on_failure(name, error)`` lets the caller
    put the primary on cooldown.
    """

    def __init__(
        self,
        primary: SttEngine,
        fallback: SttEngine,
        on_failure: Callable[[str, SttError], None] | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}>{fallback.name}"
        self.streaming = primary.streaming
        self._on_failure = on_failure

    def load(self) -> None:
        try:
            self.primary.load()
        except SttError:
            self.fallback.load()

    def transcribe_stream(self, chunks: Iterable[np.ndarray]) -> Iterator[SttEvent]:
        source = iter(chunks)
        kept: list[np.ndarray] = []
        clock = AudioClock()

        def tee() -> Iterator[np.ndarray]:
            for chunk in source:
                clock.add(chunk)
                kept.append(chunk)
                yield chunk
            clock.end()

        try:
            yield from self.primary.transcribe_stream(tee())
            return
        except SttError as exc:
            log.warning("stt %s failed (%s); using %s", self.primary.name, exc, self.fallback.name)
            if self._on_failure is not None:
                self._on_failure(self.primary.name, exc)
        for chunk in source:  # the primary may have stopped reading early
            clock.add(chunk)
            kept.append(chunk)
        if not clock.t_end:
            clock.end()
        for event in self.fallback.transcribe_stream(iter(kept)):
            if isinstance(event, Final):
                event = replace(
                    event,
                    t_first_audio=clock.t_first,
                    t_end_of_audio=clock.t_end,
                    fallback_from=event.fallback_from or self.primary.name,
                )
            yield event


def chain(
    engines: list[SttEngine], on_failure: Callable[[str, SttError], None] | None = None
) -> SttEngine:
    """engines[0], falling back to engines[1], then engines[2], ..."""
    if not engines:
        raise ValueError("no engines")
    engine = engines[-1]
    for earlier in reversed(engines[:-1]):
        engine = FallbackEngine(earlier, engine, on_failure)
    return engine


def transcribe(engine: SttEngine, chunks: Iterable[np.ndarray]) -> tuple[Final, list[Partial]]:
    """Run ``engine`` to the end (eval, bench, tests)."""
    partials: list[Partial] = []
    final: Final | None = None
    for event in engine.transcribe_stream(chunks):
        if isinstance(event, Partial):
            partials.append(event)
        else:
            final = event
    if final is None:
        raise SttError(f"{engine.name} produced no final text")
    return final, partials


# --------------------------------------------------------------------------- selection (pure)
def select_engines(
    *,
    language: str,
    dictation: bool,
    engine: str = "auto",
    dictation_accurate: bool = False,
    groq_ready: bool = False,
    whisper_installed: bool = False,
) -> list[str]:
    """Engine names to try, in order. Empty = nothing can transcribe this language offline.

    ``groq_ready``: GROQ_API_KEY set and Groq not cooling down after an error (it is only known
    to be offline when a request fails; that failure falls back within the same utterance).
    """
    english = language == "en"
    if engine == "moonshine":
        return ["moonshine"]
    if engine == "whisper_local":
        names = ["whisper_local"] if whisper_installed else []
    elif engine == "groq":
        names = ["groq"] + (["whisper_local"] if whisper_installed else [])
    elif english and not (dictation and dictation_accurate):
        return ["moonshine"]
    else:
        names = (["groq"] if groq_ready else []) + (["whisper_local"] if whisper_installed else [])
    if english:
        names.append("moonshine")  # last resort: offline English always works
    return names
