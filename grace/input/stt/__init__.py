"""Speech to text: engines (groq, moonshine, whisper_local) and the per-utterance choice.

    router = build_stt_router(settings, quota=governor)  # the app's one QuotaGovernor
    router.preload()                         # load the default engine before the first command
    engine = router.engine_for(dictation=False)
    for event in engine.transcribe_stream(chunks): ...
    router.report(final.engine)              # status line: "Voice: cloud", "Voice: local mode..."

Cloud-first since S06b (2026-11-26): Groq Whisper is the default; Moonshine is the optional
[stt-local] extra (offline fallback, demo mode) and faster-whisper the optional [stt-whisper] one.
Engines are created lazily and cached; one that can't load (package or model missing) is skipped
with a logged reason. After Groq fails, it is skipped for ``cooldown_s`` (grace/core/cooldown.py,
the helper the LLM router uses too). Groq is also skipped while the quota governor
(grace/core/quota.py) says its free tier is spent. Nothing here imports moonshine-voice unless
that engine is built.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from grace.core.cooldown import Cooldowns
from grace.core.quota import Quota
from grace.input.stt.base import (
    LOCAL_ENGINES,
    Final,
    Partial,
    SttEngine,
    SttError,
    SttEvent,
    UtteranceCancelled,
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
    "UtteranceCancelled",
    "VOICE_CLOUD",
    "VOICE_LOCAL",
    "VOICE_OFFLINE",
    "build_stt_router",
    "engine_factories",
    "moonshine_installed",
    "select_engines",
    "transcribe",
    "voice_status",
]

log = logging.getLogger(__name__)

# Status lines (overlay subtitle + console). Shown when they change, never on every utterance.
VOICE_CLOUD = "Voice: cloud"
VOICE_LOCAL = "Voice: local mode (offline)"  # on Moonshine / faster-whisper because cloud is out
VOICE_LOCAL_DEMO = "Voice: local mode (demo)"
VOICE_LOCAL_CHOSEN = "Voice: local mode"  # stt.engine set to a local engine
VOICE_OFFLINE = "Voice offline: gestures and hotkeys still work"


def voice_status(engine: str | None, *, demo_mode: bool = False, setting: str = "auto") -> str:
    """The status line for the engine that answered (None = nothing could run). Pure."""
    if engine is None:
        return VOICE_OFFLINE
    if engine.split(">")[0] not in LOCAL_ENGINES:
        return VOICE_CLOUD
    if demo_mode:
        return VOICE_LOCAL_DEMO
    return VOICE_LOCAL_CHOSEN if setting in LOCAL_ENGINES else VOICE_LOCAL


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
        moonshine_installed: bool = True,
        demo_mode: bool = False,
        cooldown_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        quota: Quota | None = None,
        quota_key: str = "groq:whisper-large-v3-turbo",
        min_billed_s: float = 10.0,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self._factories = factories
        self.language = language
        self.engine_setting = engine
        self.dictation_accurate = dictation_accurate
        self.groq_key_set = groq_key_set
        self.whisper_installed = whisper_installed
        self.moonshine_installed = moonshine_installed
        self.demo_mode = demo_mode
        self.cooldown_s = cooldown_s
        self.cooldowns = Cooldowns(cooldown_s, clock)
        self.quota = quota
        self.quota_key = quota_key
        self.min_billed_s = min_billed_s
        self.on_status = on_status
        self.status: str | None = None  # last status line (None until preload / first report)
        self._lock = threading.Lock()
        self._engines: dict[str, SttEngine] = {}
        self._broken: dict[str, str] = {}  # name -> why it can't load

    # ------------------------------------------------------------------ selection
    def groq_ready(self) -> bool:
        """Key set, not cooling down, and the governor allows one more (minimum-billed) request."""
        if not self.groq_key_set or self.cooldowns.active("groq"):
            return False
        return self.quota is None or self.quota.allow(self.quota_key, audio_s=self.min_billed_s)

    def _select(self, dictation: bool, wake_check: bool, groq_ready: bool) -> list[str]:
        names = select_engines(
            language=self.language,
            dictation=dictation,
            engine=self.engine_setting,
            dictation_accurate=self.dictation_accurate,
            groq_ready=groq_ready,
            whisper_installed=self.whisper_installed,
            moonshine_installed=self.moonshine_installed,
            demo_mode=self.demo_mode,
            prefer_local=wake_check,
        )
        with self._lock:
            return [n for n in names if n not in self._broken and n in self._factories]

    def names_for(self, dictation: bool, *, wake_check: bool = False) -> list[str]:
        """Engines to try for the next utterance, in order. ``wake_check``: an utterance heard
        before the wake word (text wake mode): local engines first."""
        return self._select(dictation, wake_check, self.groq_ready())

    def can_recover(self) -> bool:
        """Nothing runs now, but something will once a cooldown or quota window passes."""
        return bool(self._select(False, False, self.groq_key_set and "groq" not in self._broken))

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
        """A request failed mid-utterance: cool Groq down (offline, rate limit). A local engine
        is not cooled down: its failures are not about the network, and it may be the only
        engine left."""
        if name in LOCAL_ENGINES:
            log.info("stt %s failed (%s)", name, error)
            return
        self.cooldowns.trip(name)
        log.info("stt %s cooling down for %.0f s (%s)", name, self.cooldown_s, error)

    def engine_for(self, dictation: bool = False, *, wake_check: bool = False) -> SttEngine | None:
        """The engine (with fallbacks) for the next utterance, or None if nothing can run."""
        names = self.names_for(dictation, wake_check=wake_check)
        engines = [e for e in (self._get(n) for n in names) if e is not None]
        return chain(engines, self.on_failure) if engines else None

    # ------------------------------------------------------------------ status
    def report(self, engine: str | None, why: str = "") -> str | None:
        """Note which engine answered (None = nothing could run). Calls ``on_status`` with the
        new line when it changed; returns that line, or None if nothing changed."""
        text = voice_status(engine, demo_mode=self.demo_mode, setting=self.engine_setting)
        if why and engine is None:
            log.info("voice offline: %s", why)
        with self._lock:
            if text == self.status:
                return None
            self.status = text
        if self.on_status is not None:
            try:
                self.on_status(text)
            except Exception:  # noqa: BLE001 - a broken listener must not stop transcription
                log.exception("stt status listener failed")
        return text

    def preload(self) -> str:
        """Load the command-mode engine now; returns a description for the console. Sets the
        first status line without announcing it (the console prints it)."""
        engine = self.engine_for(dictation=False)
        if engine is None:
            self.status = VOICE_OFFLINE
            return f"{VOICE_OFFLINE} ({'; '.join(self.reasons()) or 'no engine'})"
        self.status = voice_status(
            engine.name, demo_mode=self.demo_mode, setting=self.engine_setting
        )
        return engine.name

    def reasons(self) -> list[str]:
        """Why engines are not in use (for the console and the offline message)."""
        out = [f"{n}: {why}" for n, why in self.unavailable().items()]
        if not self.groq_key_set:
            out.append("groq: GROQ_API_KEY not set in .env")
        elif self.cooldowns.active("groq"):
            out.append(f"groq: cooling down {self.cooldowns.remaining('groq'):.0f} s")
        elif not self.groq_ready():
            out.append("groq: free speech-to-text limit reached")
        if not self.moonshine_installed:
            out.append("moonshine: not installed (optional: uv sync --extra stt-local)")
        return out

    def unavailable(self) -> dict[str, str]:
        with self._lock:
            return dict(self._broken)


def whisper_installed() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def moonshine_installed() -> bool:
    """The [stt-local] extra (moonshine-voice). Checked without importing it."""
    return importlib.util.find_spec("moonshine_voice") is not None


def engine_factories(
    settings: Any, *, threads: int | None = None, quota: Quota | None = None
) -> dict[str, Callable[[], SttEngine]]:
    """name -> function that creates that engine from settings (nothing is loaded yet).
    ``threads`` overrides inference.threads (scripts/bench_threads.py). ``quota``: the app's
    governor; Groq checks and counts each request there."""
    from grace.core.quota import stt_quota_key
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
            quota=quota,
            quota_key=stt_quota_key(settings),
            min_billed_s=stt.quota.min_billed_s,
        )

    return {"moonshine": moonshine, "whisper_local": whisper_local, "groq": groq}


def build_stt_router(
    settings: Any,
    *,
    quota: Quota | None = None,
    on_status: Callable[[str], None] | None = None,
) -> SttRouter:
    """Engines from settings (grace.config.Settings). Nothing is loaded until it is needed.

    ``quota``: the app's one QuotaGovernor (grace/core/quota.py build_governor), shared with the
    LLM router. Without it (scripts, tests) an in-memory governor with only the Whisper limits
    is used, so the free tier is still respected but nothing is persisted.
    """
    from grace.core.quota import QuotaGovernor, stt_limits, stt_quota_key

    stt = settings.stt
    key = stt_quota_key(settings)
    if quota is None:
        quota = QuotaGovernor({key: stt_limits(settings)}, state_path=None)
    return SttRouter(
        engine_factories(settings, quota=quota),
        language=settings.language,
        engine=stt.engine,
        dictation_accurate=stt.dictation_accurate,
        groq_key_set=bool(settings.api_key("groq")),
        whisper_installed=whisper_installed(),
        moonshine_installed=moonshine_installed(),
        demo_mode=settings.demo_mode,  # S07: demo mode only makes STT prefer the local engine
        cooldown_s=settings.router.cooldown_s,
        quota=quota,
        quota_key=key,
        min_billed_s=stt.quota.min_billed_s,
        on_status=on_status,
    )
