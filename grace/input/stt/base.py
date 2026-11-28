"""Speech-to-text interface: audio chunks in, partial texts and one final text out, with timings.

    for event in engine.transcribe_stream(chunks):   # chunks: float32 16 kHz frames, any size
        if isinstance(event, Partial): overlay(event.text)
        else: grammar(event.text); event.latency_ms   # end of audio -> final text

The chunk iterable ends when the utterance ends (VAD end of speech, push-to-talk release), so
``Final.latency_ms`` is exactly "end of speech decided -> final text", the M5 acceptance number.
A streaming engine (Moonshine) yields partials while audio arrives; a batch engine (faster-whisper,
Groq) only a final. Every engine yields exactly one ``Final``, last, even for empty audio, and
raises ``SttError`` when it cannot work (missing model or package, network error): callers fall
back (``FallbackEngine``) instead of crashing. An engine that sends audio off the machine says so
with ``uploads = True`` (Groq), so grace/input/voice.py can apply its upload guards.

``select_engines`` is the pure selection rule (cloud-first since S06b, 2026-11-26):
commands -> groq if usable (key set, under quota, not cooling down), then moonshine (extra
[stt-local]), then whisper_local (extra [stt-whisper]); dictation and Hindi / auto-detect -> groq,
then whisper_local, then moonshine for English only; demo mode -> moonshine first, then groq.
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


class UtteranceCancelled(BaseException):  # noqa: N818 - a signal, like asyncio.CancelledError
    """The utterance was dropped before it ended (mic off, panic, too short to send). Raised by
    the chunk iterator, so an engine stops without a result and without uploading anything.

    A BaseException on purpose: engines wrap unexpected errors with ``except Exception`` into
    SttError, which would make FallbackEngine replay the audio to the next engine (and send a
    cancelled utterance to the cloud). This passes straight through them.
    """


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
    uploads = False  # True: the audio leaves the machine (cloud engines)

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
        self.uploads = uploads(primary) or uploads(fallback)
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


class ReportingEngine:
    """``engine``, but an SttError is reported to ``on_failure`` before it is raised. The last
    engine of a chain has no FallbackEngine around it to report its failure, so without this a
    lone Groq that is offline would be retried (and wait for its timeout) on every utterance."""

    def __init__(self, engine: SttEngine, on_failure: Callable[[str, SttError], None]) -> None:
        self.engine = engine
        self.name = engine.name
        self.streaming = engine.streaming
        self.uploads = uploads(engine)
        self._on_failure = on_failure

    def load(self) -> None:
        self.engine.load()

    def transcribe_stream(self, chunks: Iterable[np.ndarray]) -> Iterator[SttEvent]:
        try:
            yield from self.engine.transcribe_stream(chunks)
        except SttError as exc:
            self._on_failure(self.engine.name, exc)
            raise


def uploads(engine: object) -> bool:
    """Does this engine (or one it falls back to) send audio off the machine?"""
    return bool(getattr(engine, "uploads", False))


def chain(
    engines: list[SttEngine], on_failure: Callable[[str, SttError], None] | None = None
) -> SttEngine:
    """engines[0], falling back to engines[1], then engines[2], ... With ``on_failure``, every
    engine's failure is reported, the last one's too."""
    if not engines:
        raise ValueError("no engines")
    engine = engines[-1]
    if on_failure is not None:
        engine = ReportingEngine(engine, on_failure)
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
LOCAL_ENGINES = frozenset({"moonshine", "whisper_local"})


def select_engines(
    *,
    language: str,
    dictation: bool,
    engine: str = "auto",
    dictation_accurate: bool = False,
    groq_ready: bool = False,
    whisper_installed: bool = False,
    moonshine_installed: bool = True,
    demo_mode: bool = False,
    prefer_local: bool = False,
) -> list[str]:
    """Engine names to try, in order. Empty = nothing can transcribe right now.

    ``groq_ready``: GROQ_API_KEY set, the quota governor allows one more request, and Groq is
    not cooling down after an error (it is only known to be offline when a request fails; that
    failure falls back within the same utterance). ``moonshine_installed`` / ``whisper_installed``:
    the [stt-local] / [stt-whisper] extras. ``demo_mode``: Moonshine first, no network needed on
    stage. ``prefer_local``: local engines before the cloud (wake-word checks: what is said
    before the wake word stays on the machine when it can). ``dictation_accurate`` is kept for
    old callers; since S06b dictation always tries Groq first.
    """
    english = language == "en"
    groq = ["groq"] if groq_ready else []
    whisper = ["whisper_local"] if whisper_installed else []
    moonshine = ["moonshine"] if moonshine_installed else []
    moonshine_en = moonshine if english else []  # Moonshine models are English only
    if engine == "moonshine":
        names = moonshine
    elif engine == "whisper_local":
        names = whisper + moonshine_en
    elif engine == "groq":
        names = groq + whisper + moonshine_en
    elif demo_mode and english:
        names = moonshine + groq + whisper
    elif english and not dictation:
        names = groq + moonshine + whisper
    else:
        names = groq + whisper + moonshine_en
    if prefer_local:
        names = [n for n in names if n in LOCAL_ENGINES] + [
            n for n in names if n not in LOCAL_ENGINES
        ]
    return names
