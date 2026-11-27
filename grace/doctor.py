"""GRACE environment doctor.

Usage:
    uv run python -m grace doctor               # full check (opens mic + camera briefly)
    uv run python -m grace doctor --no-hardware # skip mic/camera/volume/brightness
    (scripts/doctor.py still works and calls this module)

Exit code 0 when nothing FAILs. WARN never fails the run. Features that are not built yet
report "not configured". Secret values are never printed: only key names.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import shutil
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `python grace/doctor.py` only puts grace/ on sys.path
    sys.path.insert(0, str(ROOT))
IS_WINDOWS = sys.platform == "win32"

PASS, WARN, FAIL, NOT_CONFIGURED = "PASS", "WARN", "FAIL", "not configured"

REQUIRED_PYTHON = (3, 11)

# (import name, distribution shown to the user, windows_only)
IMPORTS: tuple[tuple[str, str, bool], ...] = (
    ("pyautogui", "pyautogui", False),
    ("pyperclip", "pyperclip", False),
    ("keyboard", "keyboard", True),
    ("pyttsx3", "pyttsx3", False),
    ("sounddevice", "sounddevice", False),
    ("soundfile", "soundfile", False),
    ("numpy", "numpy", False),
    ("openai", "openai", False),
    ("httpx", "httpx", False),
    ("pytesseract", "pytesseract", False),
    ("PIL", "Pillow", False),
    ("google.generativeai", "google-generativeai", False),
    ("mediapipe", "mediapipe", False),
    ("cv2", "opencv-contrib-python (via mediapipe)", False),
    ("screen_brightness_control", "screen-brightness-control", False),
    ("comtypes", "comtypes", True),
    ("pycaw.pycaw", "pycaw", True),
    ("win32gui", "pywin32", True),
    ("pydantic", "pydantic", False),
    ("dotenv", "python-dotenv", False),
    ("yaml", "PyYAML", False),
    ("grace.config", "grace (config)", False),
    ("grace.actions", "grace (action registry)", False),
    ("grace.intent.grammar", "grace (command grammar)", False),
    ("grace.input.gestures", "grace (gesture engine)", False),
    ("grace.actions.executor", "grace (action executor)", False),
    ("PySide6.QtWidgets", "PySide6-Essentials", False),
    ("grace.core.bus", "grace (event bus)", False),
    ("grace.input.hotkeys", "grace (hotkeys)", False),
    ("grace.ui.overlay", "grace (overlay)", False),
    ("grace.runtime", "grace (runtime)", False),
    ("onnxruntime", "onnxruntime", False),
    ("openwakeword", "openwakeword", False),
    ("moonshine_voice", "moonshine-voice", False),
    ("grace.input.audio", "grace (audio + VAD)", False),
    ("grace.input.stt", "grace (speech to text)", False),
    ("grace.input.voice", "grace (voice pipeline)", False),
)


# Model files: (relative path under models/, expected sha256), from grace/models.py.
# scripts/fetch_models.py downloads them.
def _required_models() -> tuple[tuple[str, str], ...]:
    from grace.models import MODEL_FILES, REQUIRED_GROUPS

    return tuple((f.path, f.sha256) for f in MODEL_FILES if f.group in REQUIRED_GROUPS)


MODELS: tuple[tuple[str, str], ...] = _required_models()

# Keys GRACE may read from .env (all optional: cloud providers are accelerators only).
# Must match grace.config.ALL_ENV_KEYS (tests/test_doctor.py checks this).
ENV_KEYS: tuple[str, ...] = (
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_API_TOKEN",
    "TAVILY_API_KEY",
)
REACH_TIMEOUT_S = 5.0

# Features from docs/PLAN.md that do not exist yet in the codebase.
NOT_BUILT: tuple[str, ...] = ()


@dataclass(frozen=True)
class Result:
    name: str
    status: str
    detail: str = ""


# ---------------------------------------------------------------- checks


def check_python(version_info: tuple[int, ...] = tuple(sys.version_info)) -> Result:
    found = ".".join(map(str, version_info[:3]))
    ok = tuple(version_info[:2]) == REQUIRED_PYTHON
    bits = platform.architecture()[0]
    return Result("python", PASS if ok and bits == "64bit" else FAIL, f"{found} {bits}")


def check_windows_build() -> Result:
    if not IS_WINDOWS:
        return Result(
            "windows build", WARN, f"not Windows ({platform.system()}); hardware tests skipped"
        )
    build = sys.getwindowsversion().build
    name = "Windows 11" if build >= 22000 else "Windows 10"
    status = PASS if build >= 19041 else WARN  # 19041 = Windows 10 2004, oldest we test against
    return Result("windows build", status, f"{name} build {build}")


def check_uv_pip_check() -> Result:
    uv = shutil.which("uv")
    if uv is None:
        return Result("uv pip check", WARN, "uv not on PATH")
    proc = subprocess.run(
        [uv, "pip", "check", "--python", sys.executable],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
    )
    out = (proc.stdout + proc.stderr).strip().splitlines()
    return Result("uv pip check", PASS if proc.returncode == 0 else FAIL, out[-1] if out else "")


def check_imports(imports: Iterable[tuple[str, str, bool]] = IMPORTS) -> list[Result]:
    results = []
    for module, dist, windows_only in imports:
        name = f"import {module}"
        if windows_only and not IS_WINDOWS:
            results.append(Result(name, WARN, "Windows-only; skipped"))
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "")
            results.append(Result(name, PASS, f"{dist} {version}".strip()))
        except Exception as exc:  # noqa: BLE001 - report any import failure
            results.append(Result(name, FAIL, f"{dist}: {type(exc).__name__}: {exc}"))
    return results


def check_mediapipe_legacy_hands() -> Result:
    try:
        import mediapipe as mp

        mp.solutions.hands  # noqa: B018 - grace/input/gestures/tracker.py uses the legacy API
        return Result("mediapipe solutions.hands", PASS, "legacy API present")
    except Exception as exc:  # noqa: BLE001
        return Result("mediapipe solutions.hands", FAIL, f"{type(exc).__name__}: {exc}")


def check_tesseract() -> Result:
    found = shutil.which("tesseract")
    candidates = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    found = found or next((str(p) for p in candidates if p.exists()), None)
    if not found:
        return Result("tesseract binary", WARN, "not found; legacy screen reading will not work")
    return Result("tesseract binary", PASS, found)


def check_microphone() -> Result:
    try:
        import sounddevice as sd

        device = sd.query_devices(kind="input")
        with sd.InputStream(channels=1, samplerate=16000, blocksize=1600):
            time.sleep(0.2)
        return Result("microphone", PASS, f"opened '{device['name']}' at 16 kHz")
    except Exception as exc:  # noqa: BLE001
        return Result("microphone", FAIL, f"{type(exc).__name__}: {exc}")


def check_camera(index: int = 0) -> Result:
    try:
        import cv2

        backend = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        try:
            if not cap.isOpened():
                return Result("camera", FAIL, f"camera {index} did not open (in use or missing?)")
            ok, frame = cap.read()
            if not ok or frame is None:
                return Result("camera", FAIL, f"camera {index} opened but returned no frame")
            h, w = frame.shape[:2]
            return Result("camera", PASS, f"camera {index}: {w}x{h} frame")
        finally:
            cap.release()
    except Exception as exc:  # noqa: BLE001
        return Result("camera", FAIL, f"{type(exc).__name__}: {exc}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_models(
    models: Iterable[tuple[str, str]] = MODELS, models_dir: Path = ROOT / "models"
) -> list[Result]:
    models = list(models)
    if not models:
        return [Result("model files", NOT_CONFIGURED, "no models required yet")]
    results = []
    for rel, expected in models:
        path = models_dir / rel
        rel = _short(rel)
        if not path.exists():
            results.append(Result(f"model {rel}", FAIL, "missing"))
        elif sha256(path) != expected.lower():
            results.append(Result(f"model {rel}", FAIL, "hash mismatch"))
        else:
            results.append(Result(f"model {rel}", PASS, "hash ok"))
    return results


def _short(rel: str) -> str:
    """moonshine/download.moonshine.ai/model/small-streaming-en/quantized_26_11_02/x.ort ->
    moonshine/small-streaming-en/x.ort (display only)."""
    parts = rel.split("/")
    if parts[0] == "moonshine" and len(parts) > 4:
        return "/".join([parts[0], parts[3], parts[-1]])
    return rel


def read_env_key_names(env_path: Path) -> set[str]:
    """Return the names of keys with a non-empty value. Values are read but never returned."""
    from dotenv import dotenv_values  # same parser grace.config uses

    return {k for k, v in dotenv_values(env_path).items() if v and v.strip()}


def check_env(env_path: Path = ROOT / ".env", keys: Iterable[str] = ENV_KEYS) -> list[Result]:
    if not env_path.exists():
        return [Result(".env", NOT_CONFIGURED, "no .env file (cloud providers optional)")]
    present = read_env_key_names(env_path)
    return [
        Result(
            f".env {key}",
            PASS if key in present else NOT_CONFIGURED,
            "set" if key in present else "unset",
        )
        for key in keys
    ]


def check_settings(settings_path: Path | None = None, env_path: Path | None = None) -> list[Result]:
    """Validate settings.yaml and report which LLM providers are usable (never key values)."""
    from grace.config import DEFAULT_SETTINGS_PATH, ConfigError, load_settings

    settings_path = settings_path or DEFAULT_SETTINGS_PATH
    try:
        settings = load_settings(settings_path, env_path)
    except ConfigError as exc:
        return [Result("settings.yaml", FAIL, " ".join(str(exc).split()))]
    if settings_path.is_file():
        detail = f"valid; inference.threads={settings.inference.threads}"
        results = [Result("settings.yaml", PASS, detail)]
    else:
        results = [Result("settings.yaml", NOT_CONFIGURED, "missing; using defaults")]
    for status in settings.provider_status():
        state = PASS if status.enabled else NOT_CONFIGURED
        results.append(Result(f"provider {status.provider}", state, status.reason))
    return results


def check_grammar() -> Result:
    """synonyms.yaml compiles and every intent it can produce has a handler (no hardware)."""
    from grace.actions import create_fake_registry
    from grace.intent.grammar import RULES, GrammarError, default_grammar
    from grace.intent.types import APP_INTENTS

    try:
        grammar = default_grammar()
    except GrammarError as exc:
        return Result("command grammar", FAIL, str(exc))
    registry, _ = create_fake_registry()
    names = {r.intent for r in RULES} | {n for n, _ in grammar.phrases} | {"input.type"}
    missing = sorted(n for n in names if n not in registry and n not in APP_INTENTS)
    if missing:
        return Result("command grammar", FAIL, f"intents without a handler: {missing}")
    risky = sorted(s.name for s in registry if s.risky)
    detail = (
        f"{len(grammar.phrases) + len(grammar.shortcuts) + len(grammar.rules)} patterns, "
        f"{len(registry.names())} actions, risky: {', '.join(risky)}"
    )
    return Result("command grammar", PASS, detail)


def check_router(settings_path: Path | None = None, env_path: Path | None = None) -> Result:
    """LLM router: tool schema from the registry, no risky tools, provider chain (no network)."""
    from grace.actions import create_fake_registry
    from grace.config import ConfigError, load_settings
    from grace.intent.tools import EXCLUDED_ACTIONS, build_toolset

    try:
        settings = load_settings(settings_path, env_path)
    except ConfigError:
        return Result("llm router", NOT_CONFIGURED, "settings.yaml invalid (see above)")
    registry = create_fake_registry()[0]
    try:
        toolset = build_toolset(registry, search=True, images=True)
    except ValueError as exc:  # two actions in one group disagree on a parameter
        return Result("llm router", FAIL, f"tool schema: {exc}")
    offered = toolset.intents()
    if offered & EXCLUDED_ACTIONS:
        return Result("llm router", FAIL, "a pointer primitive is exposed to the LLM")
    risky = sorted(a for a in offered if registry.is_risky(a))
    if risky:
        return Result("llm router", FAIL, f"risky actions exposed to the LLM: {risky}")
    chain = settings.enabled_providers()
    detail = (
        f"{len(toolset.tools)} tools (with search + images); risky actions never offered; "
        f"chain: {' -> '.join(chain) if chain else 'none (fixed commands only)'}"
    )
    return Result("llm router", PASS if chain else NOT_CONFIGURED, detail)


def check_assistant(settings_path: Path | None = None, env_path: Path | None = None) -> Result:
    """Router tools run by our code: web_search (Tavily) and generate_image (Cloudflare FLUX).
    Config only; reachability is check_reachability."""
    from grace.config import ConfigError, load_settings

    try:
        settings = load_settings(settings_path, env_path)
    except ConfigError:
        return Result("router tools", NOT_CONFIGURED, "settings.yaml invalid (see above)")
    have = []
    if settings.search.enabled and settings.secret("tavily"):
        have.append("web_search: tavily")
    if (
        settings.images.enabled
        and settings.secret("cloudflare_account_id")
        and settings.secret("cloudflare_api_token")
    ):
        have.append(f"generate_image: cloudflare {settings.images.model.rsplit('/', 1)[-1]}")
    if not have:
        why = "TAVILY_API_KEY and CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set in .env"
        return Result("router tools", NOT_CONFIGURED, why)
    return Result("router tools", PASS, "; ".join(have))


def _reach_provider(name: str, base_url: str, key: str, get=None) -> Result:  # noqa: ANN001
    """GET {base_url}/models: authenticated, spends no tokens or requests of chat quota."""
    import httpx

    get = get or httpx.get
    try:
        response = get(
            base_url.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=REACH_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        return Result(f"reach {name}", WARN, f"not reachable ({type(exc).__name__})")
    if response.status_code == 200:
        return Result(f"reach {name}", PASS, "key accepted (model list)")
    if response.status_code in (401, 403):
        return Result(f"reach {name}", WARN, f"key rejected (HTTP {response.status_code})")
    return Result(f"reach {name}", WARN, f"HTTP {response.status_code}")


def check_reachability(
    settings_path: Path | None = None, env_path: Path | None = None
) -> list[Result]:
    """Each configured cloud service answers an authenticated request that spends no quota:
    the model list (groq, gemini, openrouter), Cloudflare's token verify, Tavily's usage.
    Problems are WARN: cloud services are optional and the app degrades without them."""
    from grace.intent.images import CloudflareFlux
    from grace.intent.search import TavilySearch

    settings = _load_settings_quietly(settings_path, env_path)
    if settings is None:
        return [Result("reach providers", NOT_CONFIGURED, "settings.yaml invalid (see above)")]
    results = []
    for provider in settings.enabled_providers():
        endpoint = settings.router.endpoint(provider)
        results.append(_reach_provider(provider, endpoint.base_url, settings.api_key(provider)))
    account = settings.secret("cloudflare_account_id")
    token = settings.secret("cloudflare_api_token")
    if account and token:
        ok, detail = CloudflareFlux(account, token, registry=None).check()
        results.append(Result("reach cloudflare", PASS if ok else WARN, detail))
    tavily = settings.secret("tavily")
    if tavily:
        ok, detail = TavilySearch(tavily).check()
        results.append(Result("reach tavily", PASS if ok else WARN, detail))
    return results or [Result("reach providers", NOT_CONFIGURED, "no cloud keys in .env")]


def check_gestures(settings_path: Path | None = None, env_path: Path | None = None) -> Result:
    """Gesture map from settings.yaml builds a controller whose intents all have handlers."""
    from grace.actions import create_fake_registry
    from grace.config import ConfigError, load_settings
    from grace.input.gestures import Behaviour, ControllerConfig, GestureController

    try:
        gs = load_settings(settings_path, env_path).gestures
    except ConfigError:
        return Result("gesture map", NOT_CONFIGURED, "settings.yaml invalid (see above)")
    controller = GestureController(ControllerConfig.from_settings(gs, (1920, 1080)))
    registry, _ = create_fake_registry()
    needed = ("input.move_to", "input.click", "input.mouse_down", "input.mouse_up",
              "input.scroll", "volume.step", "brightness.step")  # fmt: skip
    missing = [n for n in needed if n not in registry]
    if missing:
        return Result("gesture map", FAIL, f"actions missing from the registry: {missing}")
    mapping = controller.cfg.gesture_map
    active = sorted(p.value for p, b in mapping.items() if b is not Behaviour.NONE)
    detail = (
        f"{len(active)} poses mapped, hold {gs.hold_frames} frames, "
        f"pinch {gs.pinch_on}/{gs.pinch_off}"
    )
    return Result("gesture map", PASS, detail)


def check_overlay() -> Result:
    """PySide6 imports and reports its Qt version (no window is opened)."""
    try:
        import PySide6
        from PySide6 import QtCore, QtWidgets  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return Result("overlay (PySide6)", FAIL, f"PySide6 unavailable: {exc}")
    return Result(
        "overlay (PySide6)", PASS, f"PySide6 {PySide6.__version__}, Qt {QtCore.qVersion()}"
    )


def check_hotkeys(settings_path: Path | None = None, env_path: Path | None = None) -> Result:
    """Every hotkey in settings.yaml resolves to keys (no hook is installed)."""
    from grace.config import ConfigError, load_settings

    try:
        hotkeys = load_settings(settings_path, env_path).hotkeys
    except ConfigError:
        return Result("hotkeys", NOT_CONFIGURED, "settings.yaml invalid (see above)")
    combos = {name: getattr(hotkeys, name) for name in type(hotkeys).model_fields}
    if not IS_WINDOWS:
        return Result("hotkeys", WARN, f"{len(combos)} configured; global hooks need Windows")
    from grace.input.hotkeys import KeyboardLibBackend

    backend = KeyboardLibBackend()
    bad = []
    for name, combo in combos.items():
        try:
            if not backend.scan_codes(combo):
                bad.append(f"{name}={combo}")
        except Exception:  # noqa: BLE001 - unknown key name
            bad.append(f"{name}={combo}")
    if bad:
        return Result("hotkeys", FAIL, f"unknown keys: {', '.join(bad)}")
    return Result("hotkeys", PASS, ", ".join(f"{c}" for c in combos.values()))


def check_hand_tracker(frames: int = 15) -> Result:
    """Open the camera through MediaPipe Hands and time a few frames (no window)."""
    try:
        from grace.config import load_settings
        from grace.input.gestures import open_mediapipe_tracker

        gs = load_settings().gestures
        started = time.perf_counter()
        tracker = open_mediapipe_tracker(gs)
        try:
            got = [tracker.read() for _ in range(frames)]
        finally:
            tracker.close()
        elapsed = time.perf_counter() - started
        ok = [f for f in got if f is not None]
        if len(ok) < frames // 2:
            return Result("hand tracker", FAIL, f"only {len(ok)}/{frames} frames from the camera")
        hands = sum(f.hand is not None for f in ok)
        return Result(
            "hand tracker",
            PASS,
            f"MediaPipe on camera {gs.camera_index}: {len(ok)} frames in {elapsed:.1f} s "
            f"incl. start-up, hand in {hands}",
        )
    except Exception as exc:  # noqa: BLE001
        return Result("hand tracker", FAIL, f"{type(exc).__name__}: {exc}")


def check_volume() -> Result:
    """Read (never change) the default playback device's volume through pycaw."""
    if not IS_WINDOWS:
        return Result("volume (pycaw)", WARN, "Windows-only; skipped")
    try:
        from grace.actions.system import PycawVolume

        vol = PycawVolume()
        muted = " (muted)" if vol.is_muted() else ""
        return Result(
            "volume (pycaw)", PASS, f"default playback device at {vol.get_level()}%{muted}"
        )
    except Exception as exc:  # noqa: BLE001
        return Result("volume (pycaw)", FAIL, f"{type(exc).__name__}: {exc}")


def check_brightness() -> Result:
    """Read (never change) display 0's brightness. WARN, not FAIL: desktops often have no
    software-controllable monitor."""
    try:
        from grace.actions.system import SbcBrightness

        return Result("brightness", PASS, f"display 0 at {SbcBrightness().get_level()}%")
    except Exception as exc:  # noqa: BLE001
        return Result("brightness", WARN, f"not controllable: {exc}")


def check_not_built(features: Iterable[str] = NOT_BUILT) -> list[Result]:
    return [Result(feature, NOT_CONFIGURED) for feature in features]


# ---------------------------------------------------------------- voice (M5)

FETCH_HINT = "run: uv run python scripts/fetch_models.py"


def _load_settings_quietly(settings_path: Path | None, env_path: Path | None):  # noqa: ANN202
    from grace.config import ConfigError, load_settings

    try:
        return load_settings(settings_path, env_path)
    except ConfigError:
        return None


def check_onnxruntime() -> Result:
    """One onnxruntime, CPU build, the same 1.23 line as the copy moonshine-voice bundles."""
    from importlib import metadata

    dists = {d.metadata["Name"].lower() for d in metadata.distributions()}
    gpu = sorted(d for d in dists if d.startswith("onnxruntime") and d != "onnxruntime")
    if gpu:
        return Result("onnxruntime (one CPU build)", FAIL, f"also installed: {', '.join(gpu)}")
    import onnxruntime as ort

    providers = ort.get_available_providers()
    if "CPUExecutionProvider" not in providers:
        return Result("onnxruntime (one CPU build)", FAIL, f"no CPU provider: {providers}")
    detail = f"onnxruntime {ort.__version__} (CPU)"
    status = PASS
    bundled = _moonshine_ort_version()
    if bundled:
        detail += f"; moonshine-voice bundles {bundled}"
        if ".".join(bundled.split(".")[:2]) != ".".join(ort.__version__.split(".")[:2]):
            status = WARN
    return Result("onnxruntime (one CPU build)", status, detail)


def _moonshine_ort_version() -> str | None:
    if not IS_WINDOWS:
        return None
    try:
        import moonshine_voice
        import win32api

        dll = Path(moonshine_voice.__file__).parent / "onnxruntime.dll"
        info = win32api.GetFileVersionInfo(str(dll), "\\")
        ms = info["FileVersionMS"]  # major.minor (the build field is a date that overflows)
        return f"{ms >> 16}.{ms & 0xFFFF}"
    except Exception:  # noqa: BLE001 - informational only
        return None


def check_vad() -> Result:
    from grace.models import MODELS_DIR, SILERO_VAD

    path = MODELS_DIR / SILERO_VAD
    if not path.is_file():
        return Result("voice activity detection", FAIL, f"Silero VAD missing: {FETCH_HINT}")
    import numpy as np

    from grace.input.audio import FRAME, SileroVad

    vad = SileroVad(path)
    started = time.perf_counter()
    probs = [vad.prob(np.zeros(FRAME, dtype=np.float32)) for _ in range(30)]
    ms = (time.perf_counter() - started) * 1000 / len(probs)
    if max(probs) > 0.5:
        return Result("voice activity detection", FAIL, "Silero VAD calls silence speech")
    return Result("voice activity detection", PASS, f"Silero VAD, {ms:.2f} ms per 32 ms frame")


def check_wakeword(settings_path: Path | None = None, env_path: Path | None = None) -> Result:
    from grace.input.wakeword import load_wake_word
    from grace.models import MODELS_DIR, resolve

    settings = _load_settings_quietly(settings_path, env_path)
    if settings is None:
        return Result("wake word", NOT_CONFIGURED, "settings.yaml invalid (see above)")
    ww = settings.wakeword
    if not ww.enabled:
        return Result("wake word", NOT_CONFIGURED, "off: every utterance is a command")
    path = resolve(ww.model_path)
    if not path.is_file():
        return Result(
            "wake word",
            NOT_CONFIGURED,
            f"no model at {ww.model_path}: saying '{settings.runtime.wake_word}' works meanwhile "
            "(docs/wakeword.md)",
        )
    det, why = load_wake_word(
        True, path, models_dir=MODELS_DIR, threshold=ww.threshold, patience=ww.patience
    )
    if det is None:
        return Result("wake word", FAIL, why)
    from grace.models import sha256 as file_sha

    return Result("wake word", PASS, f"{why}; sha256 {file_sha(path)[:12]}")


def check_stt(settings_path: Path | None = None, env_path: Path | None = None) -> list[Result]:
    """Moonshine (required, English): load + transcribe 1 s of silence. faster-whisper and Groq
    are optional: not configured when their extra / key is missing."""
    from grace.input.stt import whisper_installed

    settings = _load_settings_quietly(settings_path, env_path)
    if settings is None:
        return [Result("speech to text", NOT_CONFIGURED, "settings.yaml invalid (see above)")]
    results = [_check_moonshine(settings.stt.moonshine_model)]
    if not whisper_installed():
        results.append(
            Result(
                "stt faster-whisper",
                NOT_CONFIGURED,
                "extra not installed (uv sync --extra stt-whisper)",
            )  # fmt: skip
        )
    else:
        from grace.models import MODELS_DIR, WHISPER_DIR

        size = settings.stt.whisper_model
        cached = (MODELS_DIR / WHISPER_DIR / f"models--Systran--faster-whisper-{size}").is_dir()
        results.append(
            Result("stt faster-whisper", PASS, f"installed, model '{size}' downloaded")
            if cached
            else Result(
                "stt faster-whisper", WARN, f"model '{size}' not downloaded: {FETCH_HINT} --whisper"
            )
        )
    key = settings.api_key("groq")
    results.append(
        Result("stt groq", PASS, f"GROQ_API_KEY set; {settings.stt.groq_model} (not called)")
        if key
        else Result("stt groq", NOT_CONFIGURED, "GROQ_API_KEY not set in .env (optional)")
    )
    return results


def _check_moonshine(arch: str) -> Result:
    from grace.models import MOONSHINE_ARCHS, moonshine_dir

    name = f"stt moonshine {arch}"
    if not moonshine_dir(arch).is_dir():
        return Result(name, FAIL, f"model missing: {FETCH_HINT}")
    import numpy as np

    from grace.input.audio import FRAME, frames
    from grace.input.stt.base import SttError, transcribe
    from grace.input.stt.moonshine import MoonshineStt

    engine = MoonshineStt(moonshine_dir(arch), MOONSHINE_ARCHS[arch][1])
    started = time.perf_counter()
    try:
        engine.load()
        load_ms = (time.perf_counter() - started) * 1000
        final, _ = transcribe(engine, frames(np.zeros(FRAME * 31, dtype=np.float32)))
    except SttError as exc:
        return Result(name, FAIL, str(exc))
    return Result(
        name, PASS, f"loaded in {load_ms:.0f} ms; 1 s of audio -> text in {final.latency_ms:.0f} ms"
    )


# ---------------------------------------------------------------- runner


def run_checks(hardware: bool = True, network: bool = True) -> list[Result]:
    checks: list[Callable[[], Result | list[Result]]] = [
        check_python,
        check_windows_build,
        check_uv_pip_check,
        check_imports,
        check_mediapipe_legacy_hands,
        check_tesseract,
        check_grammar,
        check_router,
        check_assistant,
        check_gestures,
        check_hotkeys,
        check_overlay,
        check_onnxruntime,
        check_vad,
        check_wakeword,
        check_stt,
    ]
    if hardware:
        checks += [
            check_microphone,
            check_camera,
            check_hand_tracker,
            check_volume,
            check_brightness,
        ]
    checks += [check_models, check_env, check_settings, check_not_built]
    if network:
        checks.append(check_reachability)

    results: list[Result] = []
    for check in checks:
        try:
            out = check()
        except Exception as exc:  # noqa: BLE001 - a crashing check is a failure, not a crash
            out = Result(check.__name__, FAIL, f"check crashed: {type(exc).__name__}: {exc}")
        results.extend(out if isinstance(out, list) else [out])
    if not hardware:
        results += [
            Result("microphone", WARN, "skipped (--no-hardware)"),
            Result("camera", WARN, "skipped (--no-hardware)"),
            Result("hand tracker", WARN, "skipped (--no-hardware)"),
            Result("volume (pycaw)", WARN, "skipped (--no-hardware)"),
            Result("brightness", WARN, "skipped (--no-hardware)"),
        ]
    return results


def format_results(results: list[Result]) -> str:
    width = max(len(r.name) for r in results)
    lines = [f"  [{r.status:^14}] {r.name:<{width}}  {r.detail}".rstrip() for r in results]
    counts = {s: sum(r.status == s for r in results) for s in (PASS, WARN, FAIL, NOT_CONFIGURED)}
    summary = ", ".join(f"{n} {s}" for s, n in counts.items())
    return "GRACE doctor\n" + "\n".join(lines) + f"\n\nSummary: {summary}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the GRACE environment.")
    parser.add_argument(
        "--no-hardware",
        action="store_true",
        help="skip microphone, camera, volume and brightness checks",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="skip the provider reachability checks (model list, token verify, usage)",
    )
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    args = parser.parse_args(argv)

    results = run_checks(hardware=not args.no_hardware, network=not args.no_network)
    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
    else:
        print(format_results(results))
    return 1 if any(r.status == FAIL for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
