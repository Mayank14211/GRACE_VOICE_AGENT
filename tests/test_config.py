"""grace.config: defaults, overrides, missing keys, validation. No hardware, no network."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from grace import config
from grace.config import DEFAULT_GESTURE_MAP, ConfigError, Settings, load_settings

ROOT = Path(__file__).resolve().parent.parent
# Built at runtime so the repo secret scan never sees a key-shaped literal.
FAKE_GROQ = "gsk" + "_" + "A1b2C3d4" * 7
FAKE_GEMINI = "AI" + "za" + "Sy" + "X" * 33


@pytest.fixture(autouse=True)
def _fresh_report_state() -> None:
    config._reported.clear()


def load(tmp_path: Path, yaml_text: str | None = None, env_text: str | None = None, **environ):
    settings_path = tmp_path / "settings.yaml"
    env_path = tmp_path / ".env"
    if yaml_text is not None:
        settings_path.write_text(yaml_text, encoding="utf-8")
    if env_text is not None:
        env_path.write_text(env_text, encoding="utf-8")
    return load_settings(settings_path, env_path, environ=environ)


def status(settings: Settings) -> dict[str, config.ProviderStatus]:
    return {s.provider: s for s in settings.provider_status()}


# ---------------------------------------------------------------- defaults


def test_defaults_without_any_files(tmp_path: Path) -> None:
    s = load(tmp_path)
    assert s.inference.threads == 4
    assert s.language == "en"
    assert s.demo_mode is False
    assert s.hotkeys.mic_toggle == "ctrl+alt+m"
    assert s.hotkeys.gesture_toggle == "ctrl+alt+g"
    assert s.hotkeys.push_to_talk == "ctrl+alt+space"
    assert s.hotkeys.panic_stop == "ctrl+alt+q"
    assert s.hotkeys.confirm == "ctrl+alt+y"
    assert s.hotkeys.history == "ctrl+alt+h"
    assert s.hotkeys.mark_wrong == "ctrl+alt+x"
    assert s.router.provider_order == ["groq", "gemini", "openrouter", "local"]
    assert s.router.cloud_timeout_s == 3.0
    assert s.router.local_timeout_s == 15.0
    assert s.router.cooldown_s == 60.0
    assert s.gestures.hold_frames == 4
    assert s.wakeword.model_path == Path("models/wakeword/grace.onnx")


def test_empty_settings_and_env_files_give_defaults(tmp_path: Path) -> None:
    assert load(tmp_path, yaml_text="", env_text="") == Settings()


def test_example_settings_file_matches_defaults_and_documents_every_option() -> None:
    example = ROOT / "settings.example.yaml"
    data = yaml.safe_load(example.read_text(encoding="utf-8"))

    def keys(d: dict[str, Any], prefix: str = "") -> set[str]:
        out = set()
        for k, v in d.items():
            out.add(prefix + k)
            if isinstance(v, dict):
                out |= keys(v, f"{prefix}{k}.")
        return out

    assert keys(data) == keys(Settings().model_dump()), "settings.example.yaml out of sync"
    assert load_settings(example, ROOT / "does-not-exist.env", environ={}) == Settings()


def test_env_example_lists_every_key_empty() -> None:
    from dotenv import dotenv_values

    values = dotenv_values(ROOT / ".env.example")
    assert set(values) == set(config.ENV_KEYS.values())
    assert all(v == "" for v in values.values())


# ---------------------------------------------------------------- overrides


def test_partial_yaml_overrides_merge_with_defaults(tmp_path: Path) -> None:
    s = load(
        tmp_path,
        yaml_text="""
inference: {threads: 6}
hotkeys: {mic_toggle: "Ctrl + Shift + M"}
router:
  provider_order: [gemini, local]
  local: {enabled: true}
gestures: {hold_frames: 6}
language: auto
""",
    )
    assert s.inference.threads == 6
    assert s.hotkeys.mic_toggle == "ctrl+shift+m"  # normalised
    assert s.hotkeys.gesture_toggle == "ctrl+alt+g"  # untouched default
    assert s.router.provider_order == ["gemini", "local"]
    assert s.router.local.enabled is True
    assert s.router.local.base_url == "http://127.0.0.1:8080/v1"
    assert s.gestures.hold_frames == 6
    assert s.gestures.deadzone == 0.05
    assert s.language == "auto"


def test_gesture_map_defaults_and_partial_override(tmp_path: Path) -> None:
    default = load(tmp_path, "").gestures.map
    assert default == DEFAULT_GESTURE_MAP and len(default) == 8
    s = load(tmp_path, "gestures:\n  map: {three_up: none, fist: neutral}\n")
    assert s.gestures.map["three_up"] == "none" and s.gestures.map["fist"] == "neutral"
    assert s.gestures.map["index_up"] == "cursor"  # left out -> default kept


def test_demo_mode_pins_local(tmp_path: Path) -> None:
    s = load(
        tmp_path,
        "demo_mode: true\nrouter: {local: {enabled: true}}\n",
        f"GROQ_API_KEY={FAKE_GROQ}\n",
    )
    assert s.provider_order() == ["local"]
    assert s.enabled_providers() == ["local"]
    assert status(s)["groq"].reason == "demo_mode pins [local]"


@pytest.mark.parametrize(
    ("yaml_text", "fragment"),
    [
        ("inference: {threads: 12}", "inference.threads"),
        ("inference: {threads: 0}", "inference.threads"),
        ("inference: {thread: 4}", "inference.thread"),  # typo
        ("unknown_option: 1", "unknown_option"),
        ("router: {provider_order: [groq, chatgpt]}", "router.provider_order"),
        ("router: {provider_order: [groq, groq]}", "duplicates"),
        ("hotkeys: {mic_toggle: ctrl+alt+g}", "both"),
        ("hotkeys: {mic_toggle: 'ctrl++m'}", "empty key"),
        ("gestures: {deadzone: 2}", "gestures.deadzone"),
        ("gestures: {map: {thumbs_up: cursor}}", "gestures.map"),
        ("gestures: {map: {fist: shutdown}}", "gestures.map"),
        ("gestures: {pinch_on: 0.3, pinch_off: 0.2}", "pinch_off"),
        ("gestures: {max_os_calls_per_s: 30}", "gestures.max_os_calls_per_s"),
        ("language: fr", "language"),
        ("- just\n- a list", "mapping"),
        ("inference: {threads: [", "not valid YAML"),
    ],
)
def test_invalid_settings_raise_readable_config_error(
    tmp_path: Path, yaml_text: str, fragment: str
) -> None:
    with pytest.raises(ConfigError, match=None) as info:
        load(tmp_path, yaml_text)
    assert fragment in str(info.value)


@pytest.mark.parametrize(
    "yaml_text",
    ["groq_api_key: abc", "keys: {groq: abc}", "router: {gemini_api_key: abc}", "hf_token: x"],
)
def test_keys_are_rejected_in_settings_yaml(tmp_path: Path, yaml_text: str) -> None:
    with pytest.raises(ConfigError, match=r"\.env"):
        load(tmp_path, yaml_text)


def test_key_value_pasted_into_yaml_is_rejected_without_echoing_it(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        load(tmp_path, f"router:\n  local:\n    base_url: {FAKE_GEMINI}\n")
    assert FAKE_GEMINI not in str(info.value)


# ---------------------------------------------------------------- keys / providers


def test_missing_keys_disable_cloud_providers_without_crashing(tmp_path: Path) -> None:
    s = load(tmp_path)
    st = status(s)
    for provider, var in config.ENV_KEYS.items():
        assert st[provider].enabled is False
        assert var in st[provider].reason
        assert s.api_key(provider) is None
    assert st["local"].enabled is False
    assert s.enabled_providers() == []


def test_some_keys_present(tmp_path: Path) -> None:
    env = f"# comment\nGROQ_API_KEY={FAKE_GROQ}\nGEMINI_API_KEY=\nOPENROUTER_API_KEY=   \n"
    s = load(tmp_path, env_text=env)
    assert s.api_key("groq") == FAKE_GROQ
    assert s.api_key("gemini") is None  # empty value means unset
    assert s.api_key("openrouter") is None  # whitespace means unset
    assert s.enabled_providers() == ["groq"]


def test_process_environment_overrides_env_file(tmp_path: Path) -> None:
    s = load(tmp_path, env_text="GROQ_API_KEY=from-file-value\n", GROQ_API_KEY="from-environ-value")
    assert s.api_key("groq") == "from-environ-value"


def test_provider_not_in_order_is_disabled_even_with_key(tmp_path: Path) -> None:
    s = load(tmp_path, "router: {provider_order: [gemini]}", f"GROQ_API_KEY={FAKE_GROQ}\n")
    assert status(s)["groq"].enabled is False
    assert status(s)["groq"].reason == "not in router.provider_order"


def test_enabled_providers_follow_configured_order(tmp_path: Path) -> None:
    s = load(
        tmp_path,
        "router: {provider_order: [local, gemini, groq], local: {enabled: true}}",
        f"GROQ_API_KEY={FAKE_GROQ}\nGEMINI_API_KEY={FAKE_GEMINI}\n",
    )
    assert s.enabled_providers() == ["local", "gemini", "groq"]


def test_keys_never_appear_in_repr_dump_or_json(tmp_path: Path) -> None:
    s = load(tmp_path, env_text=f"GROQ_API_KEY={FAKE_GROQ}\nGEMINI_API_KEY={FAKE_GEMINI}\n")
    for text in (repr(s), str(s), str(s.model_dump()), s.model_dump_json(), repr(s.keys)):
        assert FAKE_GROQ not in text
        assert FAKE_GEMINI not in text
    assert sorted(s.secret_values()) == sorted([FAKE_GROQ, FAKE_GEMINI])


def test_loading_does_not_write_keys_to_os_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"GROQ_API_KEY={FAKE_GROQ}\n")
    load_settings(tmp_path / "settings.yaml", env)  # real os.environ
    import os

    assert "GROQ_API_KEY" not in os.environ


# ---------------------------------------------------------------- reporting


def test_report_logs_each_disabled_provider_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    s = load(tmp_path)
    with caplog.at_level(logging.INFO, logger="grace.config"):
        config.report_providers(s)
        first = list(caplog.messages)
        config.report_providers(s)
        config.report_providers(s)
    assert caplog.messages == first, "repeat reports must be silent"
    for provider in ("groq", "gemini", "openrouter", "local"):
        assert sum(f"provider {provider}: disabled" in m for m in first) == 1
    assert any("no LLM provider enabled" in m for m in first)


def test_report_never_logs_key_values(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    s = load(tmp_path, env_text=f"GROQ_API_KEY={FAKE_GROQ}\n")
    with caplog.at_level(logging.INFO, logger="grace.config"):
        config.report_providers(s)
    assert any("provider groq: enabled (GROQ_API_KEY set)" in m for m in caplog.messages)
    assert FAKE_GROQ not in caplog.text


def test_format_provider_report(tmp_path: Path) -> None:
    text = config.format_provider_report(load(tmp_path).provider_status())
    assert "groq" in text and "disabled" in text and "GROQ_API_KEY not set" in text


def test_follow_up_and_gemini_settings(tmp_path: Path) -> None:
    good = tmp_path / "settings.yaml"
    good.write_text(
        "wakeword: {follow_up_s: 0}\n"
        "gemini: {model: gemini-3.8-flash, web_search: false, thinking_level: minimal}\n"
    )
    s = load_settings(good, tmp_path / ".env", environ={})
    assert s.wakeword.follow_up_s == 0
    assert (s.gemini.model, s.gemini.web_search, s.gemini.thinking_level) == (
        "gemini-3.8-flash",
        False,
        "minimal",
    )
    assert Settings().gemini.model == "gemini-3.5-flash-lite"

    for bad in (
        "wakeword: {follow_up_s: -1}",
        "gemini: {thinking_level: huge}",
        "gemini: {timeout_s: 0}",
        "gemini: {modle: x}",
    ):
        good.write_text(bad + "\n")
        with pytest.raises(ConfigError):
            load_settings(good, tmp_path / ".env", environ={})


def test_audio_wakeword_and_stt_settings(tmp_path: Path) -> None:
    s = Settings()
    assert (s.audio.max_utterance_s, s.audio.dictation_max_s) == (12.0, 30.0)
    assert (s.wakeword.patience, s.stt.engine, s.stt.moonshine_model) == (
        2,
        "auto",
        "small-streaming",
    )
    good = tmp_path / "settings.yaml"
    good.write_text(
        "audio: {dictation_max_s: 20, end_of_speech_ms: 350}\n"
        "stt: {engine: whisper_local, whisper_model: tiny, dictation_accurate: true}\n"
    )
    s = load_settings(good, tmp_path / ".env", environ={})
    assert (s.audio.dictation_max_s, s.audio.end_of_speech_ms) == (20, 350)
    assert (s.stt.engine, s.stt.whisper_model, s.stt.dictation_accurate) == (
        "whisper_local",
        "tiny",
        True,
    )
    for bad in (
        "stt: {engine: google}",
        "stt: {moonshine_model: base}",
        "stt: {update_interval_s: 0}",
        "wakeword: {patience: 0}",
        "audio: {dictation_max_s: 0}",
    ):
        good.write_text(bad + "\n")
        with pytest.raises(ConfigError):
            load_settings(good, tmp_path / ".env", environ={})
