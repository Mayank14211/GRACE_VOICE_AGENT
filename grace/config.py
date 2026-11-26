"""GRACE configuration: API keys from .env, tunables from settings.yaml, validated by pydantic.

    settings = load_settings()            # repo-root settings.yaml + .env (both optional)
    settings.inference.threads            # tunables
    settings.api_key("groq")              # str | None; never log the result
    report_providers(settings)            # logs each disabled provider once

Rules (docs/PLAN.md):
- Keys come only from .env (or the process environment). settings.yaml may not contain keys.
- Every key is optional. A missing key disables its provider; it never raises.
- A broken settings.yaml raises ConfigError with a readable message (fail fast, no silent defaults).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

from grace.redaction import find_secrets

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = ROOT / "settings.yaml"
DEFAULT_ENV_PATH = ROOT / ".env"

# LLM providers (2026-11-26: cloud only, no local LLM). Order = router.provider_order.
Provider = Literal["groq", "gemini", "openrouter"]
CLOUD_PROVIDERS: tuple[Provider, ...] = ("groq", "gemini", "openrouter")
ALL_PROVIDERS: tuple[Provider, ...] = CLOUD_PROVIDERS

# Provider -> .env variable holding its key.
ENV_KEYS: dict[Provider, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
# Keys of the router's tools (images: Cloudflare Workers AI; web search: Tavily).
TOOL_ENV_KEYS: dict[str, str] = {
    "cloudflare_account_id": "CLOUDFLARE_ACCOUNT_ID",
    "cloudflare_api_token": "CLOUDFLARE_API_TOKEN",
    "tavily": "TAVILY_API_KEY",
}
ALL_ENV_KEYS: dict[str, str] = {**ENV_KEYS, **TOOL_ENV_KEYS}


class ConfigError(ValueError):
    """settings.yaml is unreadable or invalid. The message is safe to show the user."""


class _Section(BaseModel):
    # Unknown keys are typos; reject them instead of silently ignoring a setting.
    model_config = ConfigDict(extra="forbid", validate_default=True)


def _reject_removed(data: Any, removed: Mapping[str, str], section: str) -> Any:
    """Settings that existed once and were removed fail with a message saying what replaced
    them, instead of pydantic's bare "Extra inputs are not permitted"."""
    if isinstance(data, Mapping):
        for key, why in removed.items():
            if key in data:
                raise ValueError(f"{section}.{key} was removed: {why}")
    return data


class HotkeySettings(_Section):
    mic_toggle: str = "ctrl+alt+m"
    gesture_toggle: str = "ctrl+alt+g"
    push_to_talk: str = "ctrl+alt+space"
    panic_stop: str = "ctrl+alt+q"
    confirm: str = "ctrl+alt+y"
    history: str = "ctrl+alt+h"
    mark_wrong: str = "ctrl+alt+x"

    @model_validator(mode="after")
    def _normalise_and_check_unique(self) -> HotkeySettings:
        seen: dict[str, str] = {}
        for name in type(self).model_fields:
            combo = "+".join(part.strip() for part in getattr(self, name).lower().split("+"))
            if not combo or any(not part for part in combo.split("+")):
                raise ValueError(f"hotkeys.{name}: empty key in {getattr(self, name)!r}")
            if combo in seen:
                raise ValueError(f"hotkeys.{name} and hotkeys.{seen[combo]} are both {combo!r}")
            seen[combo] = name
            setattr(self, name, combo)
        return self


class RuntimeSettings(_Section):
    """``python -m grace``: what is on at start, overlay and feedback options."""

    mic_on_at_start: bool = True
    gestures_on_at_start: bool = False  # the camera opens only while gestures are on
    mic_button: bool = True  # on-screen microphone button next to the status chip (needs overlay)
    gesture_button: bool = True  # on-screen camera / gesture on-off button, left of the mic button
    # Spoken wake word used while there is no trained wake-word model (wakeword.model_path missing):
    # every utterance is transcribed offline and must start with this word. Also stripped from
    # push-to-talk commands ("listen, open notepad" -> "open notepad").
    wake_word: str = Field("listen", min_length=2)
    overlay: bool = True
    subtitle_fade_s: float = Field(4.0, ge=0.5, le=60.0)
    history_turns: int = Field(10, ge=1, le=50)
    earcons: bool = True
    metrics: bool = True  # JSONL rows in logs/metrics/ (git-ignored)
    cpu_sample_s: float = Field(60.0, ge=5.0, le=3600.0)


class InferenceSettings(_Section):
    # 4 on the i5-12450H (P-cores only), 6 on the Ryzen 5 5500U. Never logical-thread counts.
    threads: int = Field(4, ge=1, le=8)


class AudioSettings(_Section):
    """grace/input/audio.py: one 16 kHz mono mic stream, Silero VAD marks utterances."""

    sample_rate: Literal[16000] = 16000
    input_device: int | str | None = None  # None = system default microphone
    vad_threshold: float = Field(0.5, ge=0.0, le=1.0)
    end_of_speech_ms: int = Field(500, ge=100, le=5000)  # silence that ends an utterance
    max_utterance_s: float = Field(12.0, gt=0.0, le=60.0)  # cap on one command
    dictation_max_s: float = Field(30.0, gt=0.0, le=120.0)  # cap on one dictated sentence
    pre_roll_ms: int = Field(300, ge=0, le=1000)  # audio kept from before speech started
    min_speech_ms: int = Field(200, ge=0, le=2000)  # shorter blips (clicks, coughs) are dropped


class WakeWordSettings(_Section):
    """openWakeWord with the team's custom model (docs/wakeword.md).

    enabled + model file present -> openWakeWord. enabled + no model file -> the spoken
    runtime.wake_word, transcribed offline. enabled false -> while the mic is on, every utterance
    is a command (no wake word). The mic hotkey and push-to-talk always work.
    """

    enabled: bool = True
    model_path: Path = Path("models/wakeword/grace.onnx")
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    patience: int = Field(2, ge=1, le=20)  # consecutive 80 ms frames above threshold
    # After the wake word, keep listening for follow-up commands until this many seconds pass
    # without one. 0 = say the wake word before every command.
    follow_up_s: float = Field(10.0, ge=0.0, le=120.0)


SttEngineName = Literal["moonshine", "whisper_local", "groq"]


class SttSettings(_Section):
    """grace/input/stt: which engine turns speech into text.

    engine auto: language en -> moonshine; language hi / auto, or dictation with
    dictation_accurate -> groq if GROQ_API_KEY is set and it answered recently, else whisper_local.
    """

    engine: Literal["auto", "moonshine", "whisper_local", "groq"] = "auto"
    moonshine_model: Literal["tiny-streaming", "small-streaming", "medium-streaming"] = (
        "small-streaming"
    )
    update_interval_s: float = Field(0.3, ge=0.1, le=2.0)  # how often partial text refreshes
    whisper_model: Literal["tiny", "base", "small"] = "base"  # faster-whisper, int8, beam 1
    dictation_accurate: bool = False  # dictation via groq / whisper_local instead of moonshine
    groq_model: str = "whisper-large-v3-turbo"
    groq_timeout_s: float = Field(8.0, gt=0.0, le=60.0)
    partials: bool = True  # show partial text on the overlay while the user speaks


GesturePose = Literal[
    "open_palm",
    "fist",
    "index_up",
    "two_up",
    "three_up",
    "thumb_index_middle",
    "pinch_index",
    "pinch_middle",
]
GestureBehaviour = Literal[
    "cursor",
    "left_click",
    "right_click",
    "scroll",
    "volume",
    "brightness",
    "pause",
    "neutral",
    "none",
]
# docs/PLAN.md section 7. Behaviours: see grace/input/gestures/controller.py.
DEFAULT_GESTURE_MAP: dict[str, str] = {
    "index_up": "cursor",
    "pinch_index": "left_click",
    "pinch_middle": "right_click",
    "two_up": "scroll",
    "three_up": "brightness",
    "thumb_index_middle": "volume",
    "fist": "pause",
    "open_palm": "neutral",
}


class GestureSettings(_Section):
    camera_index: int = Field(0, ge=0)
    frame_width: int = Field(640, ge=160, le=1920)
    frame_height: int = Field(480, ge=120, le=1080)
    model_complexity: Literal[0, 1] = 1  # MediaPipe Hands: 0 = faster, 1 = more accurate
    min_detection_confidence: float = Field(0.7, ge=0.1, le=1.0)
    min_tracking_confidence: float = Field(0.6, ge=0.1, le=1.0)
    # Pose -> behaviour. Poses left out keep their default; "none" disables a pose.
    map: dict[GesturePose, GestureBehaviour] = Field(
        default_factory=lambda: dict(DEFAULT_GESTURE_MAP)  # type: ignore[arg-type]
    )
    hold_frames: int = Field(4, ge=1, le=30)
    drag_hold_ms: int = Field(400, ge=50, le=3000)
    pause_hold_ms: int = Field(1000, ge=200, le=5000)
    # Pinch = thumb-tip to fingertip distance / hand size. Starts below pinch_on, ends above
    # pinch_off (hysteresis, so a pinch near the threshold does not flicker).
    pinch_on: float = Field(0.22, gt=0.0, le=1.0)
    pinch_off: float = Field(0.32, gt=0.0, le=1.5)
    deadzone: float = Field(0.05, ge=0.0, le=0.5)
    active_box_margin: float = Field(0.15, ge=0.0, lt=0.5)
    # Speeds per unit of offset beyond the deadzone (offset = fraction of the frame height).
    scroll_speed: float = Field(60.0, gt=0.0, le=1000.0)  # wheel notches per second
    volume_speed: float = Field(200.0, gt=0.0, le=2000.0)  # volume points per second
    brightness_speed: float = Field(200.0, gt=0.0, le=2000.0)  # brightness points per second
    max_os_calls_per_s: float = Field(10.0, ge=1.0, le=10.0)  # scroll / volume / brightness
    # One Euro filter for the cursor (normalised units): lower min_cutoff = steadier when still,
    # higher beta = less lag when moving fast.
    cursor_min_cutoff: float = Field(1.0, gt=0.0, le=30.0)
    cursor_beta: float = Field(10.0, ge=0.0, le=1000.0)

    @model_validator(mode="after")
    def _fill_map_and_check_pinch(self) -> GestureSettings:
        self.map = {**DEFAULT_GESTURE_MAP, **self.map}  # type: ignore[dict-item]
        if self.pinch_off <= self.pinch_on:
            raise ValueError(
                f"gestures.pinch_off ({self.pinch_off}) must be larger than "
                f"gestures.pinch_on ({self.pinch_on})"
            )
        return self


class ProviderEndpoint(_Section):
    """One OpenAI-compatible chat endpoint. Model IDs live here, never in code (they change)."""

    base_url: str
    model: str = Field(min_length=1)
    # Sent as `reasoning_effort` (Groq gpt-oss, Gemini's OpenAI endpoint). None = not sent.
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    # Per-provider request timeout; None = router.cloud_timeout_s (3 s). Gemini's free tier
    # answered in 1.4-3.5 s on the base machine (2026-11-26), so it may need more headroom.
    timeout_s: float | None = Field(None, gt=0.0, le=60.0)


def _groq_endpoint() -> ProviderEndpoint:
    # console.groq.com/docs (checked 2026-11-26): OpenAI-compatible base URL; gpt-oss-20b does
    # tool use and JSON mode, but not parallel tool calls.
    return ProviderEndpoint(
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-20b",
        reasoning_effort="low",
    )


def _gemini_endpoint() -> ProviderEndpoint:
    # ai.google.dev/gemini-api/docs/openai: chat.completions with tools on the Gemini API key.
    return ProviderEndpoint(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-3.5-flash-lite",
        reasoning_effort="low",
    )


def _openrouter_endpoint() -> ProviderEndpoint:
    # openrouter.ai/api/v1/models, filtered on ":free" + "tools" (2026-11-26). The free list
    # changes often: re-check it if this model disappears.
    return ProviderEndpoint(
        base_url="https://openrouter.ai/api/v1", model="google/gemma-4-26b-a4b-it:free"
    )


_REMOVED_ROUTER = {
    "local": "no local LLM since 2026-11-26 (cloud-first): delete router.local",
    "local_timeout_s": "no local LLM since 2026-11-26: delete it",
    "allow_risky": "risky actions are never offered to the LLM; say the exact phrase instead",
}
_REMOVED_TOP = {
    "gemini": (
        "the gemini: section is gone. Chat uses router.gemini; images use images: (Cloudflare "
        "FLUX); web search uses search: (Tavily); code analysis uses the router and never "
        "Gemini (screen text stays away from Gemini)"
    ),
}


class QuotaLimit(_Section):
    """Free-tier limits of one provider's model. None = not limited by GRACE. The governor
    warns at 80 % and skips the provider at 100 % until the window resets."""

    rpm: int | None = Field(None, ge=1)  # requests per minute
    rpd: int | None = Field(None, ge=1)  # requests per day
    tpm: int | None = Field(None, ge=1)  # tokens per minute
    tpd: int | None = Field(None, ge=1)  # tokens per day


def _default_quotas() -> dict[str, QuotaLimit]:
    # Groq: console.groq.com/docs/rate-limits (gpt-oss-20b free tier; check your own limits at
    # console.groq.com/settings/limits). OpenRouter :free: 20 req/min, 50 req/day without credit.
    # Gemini: per project, visible only in AI Studio: fill it in from there.
    return {
        "groq": QuotaLimit(rpm=30, rpd=1000, tpm=8000, tpd=200_000),
        "gemini": QuotaLimit(),
        "openrouter": QuotaLimit(rpm=20, rpd=50),
    }


class ImageSettings(_Section):
    """generate_image tool: Cloudflare Workers AI FLUX.1 [schnell] (needs CLOUDFLARE_ACCOUNT_ID
    and CLOUDFLARE_API_TOKEN with Workers AI permission). Free: 10,000 neurons/day; one
    512x512 image at 4 steps is about 43 neurons."""

    enabled: bool = True
    model: str = "@cf/black-forest-labs/flux-1-schnell"
    steps: int = Field(4, ge=1, le=8)
    timeout_s: float = Field(30.0, gt=0.0, le=120.0)
    neurons_per_day: int = Field(10_000, ge=0)  # the free allowance; 0 = no limit


class SearchSettings(_Section):
    """web_search tool: Tavily (needs TAVILY_API_KEY). Free: 1,000 credits/month, no card; a
    basic search costs 1 credit. Choice: docs/results/search.md."""

    enabled: bool = True
    max_results: int = Field(3, ge=1, le=10)
    timeout_s: float = Field(8.0, gt=0.0, le=60.0)
    credits_per_month: int = Field(1000, ge=0)  # the free allowance; 0 = no limit


class RouterSettings(_Section):
    """grace/intent/router.py: the LLM fallback for anything the fixed grammar doesn't parse."""

    provider_order: list[Provider] = Field(default_factory=lambda: list(ALL_PROVIDERS))
    cloud_timeout_s: float = Field(3.0, gt=0.0, le=30.0)
    max_retries: int = Field(1, ge=0, le=3)  # network errors / 5xx only; never 429 or timeouts
    cooldown_s: float = Field(60.0, ge=0.0, le=3600.0)
    # Dialogue memory: user + GRACE exchanges sent with each request, and how long a pause
    # makes GRACE forget them. Follow-ups ("a bit more", "undo that") refer to the last action
    # for followup_s.
    history_turns: int = Field(6, ge=0, le=20)
    memory_reset_s: float = Field(600.0, ge=0.0, le=86400.0)
    followup_s: float = Field(120.0, ge=5.0, le=3600.0)
    max_tokens: int = Field(300, ge=32, le=4096)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    analysis_timeout_s: float = Field(15.0, gt=0.0, le=60.0)  # "analyze this code" (longer reply)
    groq: ProviderEndpoint = Field(default_factory=_groq_endpoint)
    gemini: ProviderEndpoint = Field(default_factory=_gemini_endpoint)
    openrouter: ProviderEndpoint = Field(default_factory=_openrouter_endpoint)
    quota: dict[Provider, QuotaLimit] = Field(default_factory=_default_quotas)

    @model_validator(mode="before")
    @classmethod
    def _removed(cls, data: Any) -> Any:
        return _reject_removed(data, _REMOVED_ROUTER, "router")

    @model_validator(mode="after")
    def _no_duplicates(self) -> RouterSettings:
        if len(set(self.provider_order)) != len(self.provider_order):
            raise ValueError(f"router.provider_order has duplicates: {self.provider_order}")
        if not self.openrouter.model.endswith(":free") and self.openrouter.model != (
            "openrouter/free"
        ):
            raise ValueError(
                "router.openrouter.model must be a ':free' model (zero running cost): "
                f"{self.openrouter.model!r}"
            )
        self.quota = {**_default_quotas(), **self.quota}
        return self

    def endpoint(self, provider: Provider) -> ProviderEndpoint:
        endpoint: ProviderEndpoint = getattr(self, provider)
        return endpoint

    def timeout_s(self, provider: Provider) -> float:
        own: float | None = getattr(self, provider).timeout_s
        return own if own is not None else self.cloud_timeout_s


class ApiKeys(BaseModel):
    """Loaded from .env only. SecretStr keeps values out of repr(), str() and dumps."""

    model_config = ConfigDict(frozen=True)

    groq: SecretStr | None = None
    gemini: SecretStr | None = None
    openrouter: SecretStr | None = None
    cloudflare_account_id: SecretStr | None = None
    cloudflare_api_token: SecretStr | None = None
    tavily: SecretStr | None = None


@dataclass(frozen=True)
class ProviderStatus:
    provider: Provider
    enabled: bool
    reason: str


class Settings(_Section):
    language: Literal["en", "hi", "auto"] = "en"
    demo_mode: bool = False  # S10 demo mode; does not change the LLM router (cloud only)
    hotkeys: HotkeySettings = Field(default_factory=HotkeySettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    wakeword: WakeWordSettings = Field(default_factory=WakeWordSettings)
    stt: SttSettings = Field(default_factory=SttSettings)
    gestures: GestureSettings = Field(default_factory=GestureSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    images: ImageSettings = Field(default_factory=ImageSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    keys: ApiKeys = Field(default_factory=ApiKeys, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _removed(cls, data: Any) -> Any:
        return _reject_removed(data, _REMOVED_TOP, "settings")

    # ------------------------------------------------------------ providers

    def api_key(self, provider: Provider) -> str | None:
        if provider not in ENV_KEYS:
            return None
        return self.secret(provider)

    def secret(self, name: str) -> str | None:
        """A value from .env by ApiKeys field name (providers and tool keys). Never log it."""
        if name not in ALL_ENV_KEYS:
            return None
        value: SecretStr | None = getattr(self.keys, name)
        return value.get_secret_value() if value else None

    def provider_order(self) -> list[Provider]:
        return list(self.router.provider_order)

    def provider_status(self) -> list[ProviderStatus]:
        order = self.provider_order()
        statuses = []
        for provider in ALL_PROVIDERS:
            if provider not in order:
                statuses.append(ProviderStatus(provider, False, "not in router.provider_order"))
            elif self.api_key(provider):
                statuses.append(ProviderStatus(provider, True, f"{ENV_KEYS[provider]} set"))
            else:
                statuses.append(
                    ProviderStatus(provider, False, f"{ENV_KEYS[provider]} not set in .env")
                )
        return statuses

    def enabled_providers(self) -> list[Provider]:
        enabled = {s.provider for s in self.provider_status() if s.enabled}
        return [p for p in self.provider_order() if p in enabled]

    def secret_values(self) -> list[str]:
        """Every configured key value, for the log redaction filter."""
        return [k for k in (self.secret(n) for n in ALL_ENV_KEYS) if k]


# ---------------------------------------------------------------- loading


def read_api_keys(env_path: Path, environ: Mapping[str, str] | None = None) -> ApiKeys:
    """Keys from `env_path` (python-dotenv syntax); process environment variables win.

    Missing file, missing key and empty value all mean "not set". Nothing is written to
    os.environ, so keys do not leak into child processes.
    """
    file_values = dotenv_values(env_path) if env_path.is_file() else {}
    environ = os.environ if environ is None else environ
    keys: dict[str, SecretStr] = {}
    for provider, var in ALL_ENV_KEYS.items():
        value = (environ.get(var) or file_values.get(var) or "").strip()
        if value:
            keys[provider] = SecretStr(value)
    return ApiKeys(**keys)


def read_settings_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        log.info("settings: %s not found, using defaults (see settings.example.yaml)", path.name)
        return {}
    text = path.read_text(encoding="utf-8")
    if find_secrets(text):
        raise ConfigError(
            f"{path.name} contains a value that looks like an API key. Move it to .env "
            "and revoke it if this file was ever shared."
        )
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must be a mapping of settings, got {type(data).__name__}")
    _reject_keys_in_yaml(data, path.name)
    return data


def _looks_like_key_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in {"keys", "api_keys", "secrets", "token"}
        or "api_key" in lowered
        or "apikey" in lowered
        or lowered.endswith(("_key", "_token", "_secret"))
    )


def _reject_keys_in_yaml(data: Mapping[str, Any], filename: str, prefix: str = "") -> None:
    for key, value in data.items():
        name = f"{prefix}{key}"
        if _looks_like_key_name(str(key)):
            raise ConfigError(
                f"{filename}: '{name}' looks like an API key. Keys belong in .env only "
                f"({', '.join(ENV_KEYS.values())})."
            )
        if isinstance(value, Mapping):
            _reject_keys_in_yaml(value, filename, f"{name}.")


def _format_validation_error(exc: ValidationError, filename: str) -> str:
    lines = [f"{filename} has {exc.error_count()} invalid setting(s):"]
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"]) or "(top level)"
        lines.append(f"  - {where}: {err['msg']}")
    return "\n".join(lines)


def load_settings(
    settings_path: Path | None = None,
    env_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load and validate configuration. Raises ConfigError only for a broken settings.yaml."""
    settings_path = Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH
    env_path = Path(env_path) if env_path else DEFAULT_ENV_PATH
    data = read_settings_file(settings_path)
    try:
        settings = Settings.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, settings_path.name)) from None
    return settings.model_copy(update={"keys": read_api_keys(env_path, environ)})


# ---------------------------------------------------------------- reporting

_reported: set[tuple[str, bool, str]] = set()


def report_providers(settings: Settings, logger: logging.Logger = log) -> list[ProviderStatus]:
    """Log each provider's status once per process (repeat calls with the same state are silent)."""
    statuses = settings.provider_status()
    for status in statuses:
        token = (status.provider, status.enabled, status.reason)
        if token in _reported:
            continue
        _reported.add(token)
        if status.enabled:
            logger.info("provider %s: enabled (%s)", status.provider, status.reason)
        else:
            logger.warning("provider %s: disabled (%s)", status.provider, status.reason)
    if not any(s.enabled for s in statuses) and ("none", False, "") not in _reported:
        _reported.add(("none", False, ""))
        logger.warning(
            "no LLM provider enabled: fixed commands still work offline; "
            "free-form requests will not be understood"
        )
    return statuses


def format_provider_report(statuses: list[ProviderStatus]) -> str:
    width = max(len(s.provider) for s in statuses)
    return "\n".join(
        f"  {s.provider:<{width}}  {'enabled ' if s.enabled else 'disabled'}  {s.reason}"
        for s in statuses
    )
