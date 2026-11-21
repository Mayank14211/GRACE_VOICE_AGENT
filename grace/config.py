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

Provider = Literal["groq", "gemini", "openrouter", "local"]
CLOUD_PROVIDERS: tuple[Provider, ...] = ("groq", "gemini", "openrouter")
ALL_PROVIDERS: tuple[Provider, ...] = (*CLOUD_PROVIDERS, "local")

# Provider -> .env variable holding its key.
ENV_KEYS: dict[Provider, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class ConfigError(ValueError):
    """settings.yaml is unreadable or invalid. The message is safe to show the user."""


class _Section(BaseModel):
    # Unknown keys are typos; reject them instead of silently ignoring a setting.
    model_config = ConfigDict(extra="forbid", validate_default=True)


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


class InferenceSettings(_Section):
    # 4 on the i5-12450H (P-cores only), 6 on the Ryzen 5 5500U. Never logical-thread counts.
    threads: int = Field(4, ge=1, le=8)


class AudioSettings(_Section):
    sample_rate: Literal[16000] = 16000
    input_device: int | str | None = None  # None = system default microphone
    vad_threshold: float = Field(0.5, ge=0.0, le=1.0)
    end_of_speech_ms: int = Field(500, ge=100, le=5000)
    max_utterance_s: float = Field(15.0, gt=0.0, le=60.0)


class WakeWordSettings(_Section):
    enabled: bool = True
    model_path: Path = Path("models/wakeword/grace.onnx")
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    # After the wake word, keep listening for follow-up commands until this many seconds pass
    # without one. 0 = say the wake word before every command.
    follow_up_s: float = Field(10.0, ge=0.0, le=120.0)


class GeminiSettings(_Section):
    """Gemini assistant for requests the fixed grammar doesn't understand (needs GEMINI_API_KEY).

    Text answers and tool calling work on the free tier. Web search (Google Search grounding)
    and image generation need a key from a billing-enabled Google Cloud project.
    """

    model: str = "gemini-3.5-flash-lite"
    image_model: str = "gemini-3.1-flash-image"
    web_search: bool = True
    image_generation: bool = True
    thinking_level: Literal["minimal", "low", "medium", "high"] = "low"
    timeout_s: float = Field(20.0, gt=0.0, le=120.0)
    image_timeout_s: float = Field(90.0, gt=0.0, le=300.0)
    # Forget the conversation after this long without a question (0 = never forget).
    memory_reset_s: float = Field(600.0, ge=0.0, le=86400.0)


class GestureSettings(_Section):
    camera_index: int = Field(0, ge=0)
    hold_frames: int = Field(4, ge=1, le=30)
    drag_hold_ms: int = Field(400, ge=50, le=3000)
    pause_hold_ms: int = Field(1000, ge=200, le=5000)
    deadzone: float = Field(0.05, ge=0.0, le=0.5)
    active_box_margin: float = Field(0.15, ge=0.0, lt=0.5)


class LocalLLMSettings(_Section):
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8080/v1"


class RouterSettings(_Section):
    provider_order: list[Provider] = Field(default_factory=lambda: list(ALL_PROVIDERS))
    cloud_timeout_s: float = Field(3.0, gt=0.0, le=30.0)
    local_timeout_s: float = Field(15.0, gt=0.0, le=120.0)
    max_retries: int = Field(1, ge=0, le=3)
    cooldown_s: float = Field(60.0, ge=0.0, le=3600.0)
    local: LocalLLMSettings = Field(default_factory=LocalLLMSettings)

    @model_validator(mode="after")
    def _no_duplicates(self) -> RouterSettings:
        if len(set(self.provider_order)) != len(self.provider_order):
            raise ValueError(f"router.provider_order has duplicates: {self.provider_order}")
        return self


class ApiKeys(BaseModel):
    """Loaded from .env only. SecretStr keeps values out of repr(), str() and dumps."""

    model_config = ConfigDict(frozen=True)

    groq: SecretStr | None = None
    gemini: SecretStr | None = None
    openrouter: SecretStr | None = None


@dataclass(frozen=True)
class ProviderStatus:
    provider: Provider
    enabled: bool
    reason: str


class Settings(_Section):
    language: Literal["en", "hi", "auto"] = "en"
    demo_mode: bool = False  # pins the router to [local] and skips cloud calls
    hotkeys: HotkeySettings = Field(default_factory=HotkeySettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    wakeword: WakeWordSettings = Field(default_factory=WakeWordSettings)
    gestures: GestureSettings = Field(default_factory=GestureSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    keys: ApiKeys = Field(default_factory=ApiKeys, exclude=True)

    # ------------------------------------------------------------ providers

    def api_key(self, provider: Provider) -> str | None:
        if provider not in ENV_KEYS:
            return None
        secret: SecretStr | None = getattr(self.keys, provider)
        return secret.get_secret_value() if secret else None

    def provider_order(self) -> list[Provider]:
        return ["local"] if self.demo_mode else list(self.router.provider_order)

    def provider_status(self) -> list[ProviderStatus]:
        order = self.provider_order()
        statuses = []
        for provider in ALL_PROVIDERS:
            if provider not in order:
                why = "demo_mode pins [local]" if self.demo_mode else "not in router.provider_order"
                statuses.append(ProviderStatus(provider, False, why))
            elif provider == "local":
                ok = self.router.local.enabled
                why = (
                    f"llama-server at {self.router.local.base_url}"
                    if ok
                    else "router.local.enabled is false in settings.yaml"
                )
                statuses.append(ProviderStatus(provider, ok, why))
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
        return [k for k in (self.api_key(p) for p in CLOUD_PROVIDERS) if k]


# ---------------------------------------------------------------- loading


def read_api_keys(env_path: Path, environ: Mapping[str, str] | None = None) -> ApiKeys:
    """Keys from `env_path` (python-dotenv syntax); process environment variables win.

    Missing file, missing key and empty value all mean "not set". Nothing is written to
    os.environ, so keys do not leak into child processes.
    """
    file_values = dotenv_values(env_path) if env_path.is_file() else {}
    environ = os.environ if environ is None else environ
    keys: dict[str, SecretStr] = {}
    for provider, var in ENV_KEYS.items():
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
