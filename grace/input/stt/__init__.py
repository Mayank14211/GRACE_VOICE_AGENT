"""Speech to text: engines (moonshine, whisper_local, groq) and the per-utterance choice.

    router = build_stt_router(settings)
    router.preload()                         # load the default engine before the first command
    engine = router.engine_for(dictation=False)
    for event in engine.transcribe_stream(chunks): ...

Engines are created lazily and cached; one that can't load (package or model missing) is skipped
with a logged reason. After a Groq failure, Groq is skipped for ``cooldown_s``.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from grace.input.stt.base import (
    Final,
    Partial,
    SttEngine,
    SttError,
    SttEvent,
    chain,
    select_engines,
    transcribe,
)

__all__ = [
    "Final",
    "Partial",
    "SttEngine",
    "SttError",
    "SttEvent",
    "SttRouter",
    "build_stt_router",
    "engine_factories",
    "select_engines",
    "transcribe",
]

log = logging.getLogger(__name__)


class SttRouter:
    def __init__(
        self,
        factories: dict[str, Callable[[], SttEngine]],
        *,
        language: str = "en",
        engine: str = "auto",
        dictation_accurate: bool = False,
        groq_key_set: bool = False,
        whisper_installed: bool = False,
        cooldown_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._factories = factories
        self.language = language
        self.engine_setting = engine
        self.dictation_accurate = dictation_accurate
        self.groq_key_set = groq_key_set
        self.whisper_installed = whisper_installed
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._engines: dict[str, SttEngine] = {}
        self._broken: dict[str, str] = {}  # name -> why it can't load
        self._cool_until: dict[str, float] = {}

    def groq_ready(self) -> bool:
        return self.groq_key_set and self._clock() >= self._cool_until.get("groq", 0.0)

    def names_for(self, dictation: bool) -> list[str]:
        names = select_engines(
            language=self.language,
            dictation=dictation,
            engine=self.engine_setting,
            dictation_accurate=self.dictation_accurate,
            groq_ready=self.groq_ready(),
            whisper_installed=self.whisper_installed,
        )
        with self._lock:
            return [n for n in names if n not in self._broken and n in self._factories]

    def _get(self, name: str) -> SttEngine | None:
        with self._lock:
            if name in self._engines:
                return self._engines[name]
            if name in self._broken:
                return None
        try:
            engine = self._factories[name]()
            engine.load()
        except SttError as exc:
            log.warning("stt engine %s unavailable: %s", name, exc)
            with self._lock:
                self._broken[name] = str(exc)
            return None
        with self._lock:
            self._engines[name] = engine
        return engine

    def on_failure(self, name: str, error: SttError) -> None:
        """A request failed mid-utterance: cool that engine down (Groq: offline, rate limit)."""
        with self._lock:
            self._cool_until[name] = self._clock() + self.cooldown_s
        log.info("stt %s cooling down for %.0f s (%s)", name, self.cooldown_s, error)

    def engine_for(self, dictation: bool = False) -> SttEngine | None:
        """The engine (with fallbacks) for the next utterance, or None if nothing can run."""
        engines = [e for e in (self._get(n) for n in self.names_for(dictation)) if e is not None]
        return chain(engines, self.on_failure) if engines else None

    def preload(self) -> str:
        """Load the command-mode engine now; returns a description for the console."""
        engine = self.engine_for(dictation=False)
        if engine is None:
            return "no speech-to-text engine available: " + "; ".join(
                f"{n}: {why}" for n, why in self.unavailable().items()
            )
        return engine.name

    def unavailable(self) -> dict[str, str]:
        with self._lock:
            return dict(self._broken)


def whisper_installed() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def engine_factories(
    settings: Any, *, threads: int | None = None
) -> dict[str, Callable[[], SttEngine]]:
    """name -> function that creates that engine from settings (nothing is loaded yet).
    ``threads`` overrides inference.threads (scripts/bench_threads.py)."""
    from grace.models import MODELS_DIR, MOONSHINE_ARCHS, WHISPER_DIR, moonshine_dir

    stt = settings.stt
    lang = None if settings.language == "auto" else settings.language

    def moonshine() -> SttEngine:
        from grace.input.stt.moonshine import MoonshineStt

        arch = stt.moonshine_model
        return MoonshineStt(
            moonshine_dir(arch), MOONSHINE_ARCHS[arch][1], update_interval=stt.update_interval_s
        )

    def whisper_local() -> SttEngine:
        from grace.input.stt.whisper_local import WhisperLocalStt

        return WhisperLocalStt(
            stt.whisper_model,
            threads=threads or settings.inference.threads,
            language=lang,
            download_root=MODELS_DIR / WHISPER_DIR,
        )

    def groq() -> SttEngine:
        from grace.input.stt.groq import GroqStt

        return GroqStt(
            settings.api_key("groq"),
            model=stt.groq_model,
            language=lang,
            timeout_s=stt.groq_timeout_s,
            max_retries=settings.router.max_retries,
        )

    return {"moonshine": moonshine, "whisper_local": whisper_local, "groq": groq}


def build_stt_router(settings: Any) -> SttRouter:
    """Engines from settings (grace.config.Settings). Nothing is loaded until it is needed."""
    stt = settings.stt
    return SttRouter(
        engine_factories(settings),
        language=settings.language,
        engine=stt.engine,
        dictation_accurate=stt.dictation_accurate,
        groq_key_set=bool(settings.api_key("groq")) and not settings.demo_mode,
        whisper_installed=whisper_installed(),
        cooldown_s=settings.router.cooldown_s,
    )
