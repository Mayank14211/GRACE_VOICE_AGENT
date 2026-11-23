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
)

# Model files: (relative path under models/, expected sha256). Filled in from M5 onward.
MODELS: tuple[tuple[str, str], ...] = ()

# Keys GRACE may read from .env (all optional: cloud providers are accelerators only).
# Must match grace.config.ENV_KEYS (tests/test_doctor.py checks this).
ENV_KEYS: tuple[str, ...] = ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY")

# Features from docs/PLAN.md that do not exist yet in the codebase.
NOT_BUILT: tuple[str, ...] = (
    "wake word model (M5)",
    "local STT: moonshine-voice (M5)",
    "llama-server (M6)",
)


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
        if not path.exists():
            results.append(Result(f"model {rel}", FAIL, "missing"))
        elif sha256(path) != expected.lower():
            results.append(Result(f"model {rel}", FAIL, "hash mismatch"))
        else:
            results.append(Result(f"model {rel}", PASS, "hash ok"))
    return results


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


def check_assistant(settings_path: Path | None = None, env_path: Path | None = None) -> Result:
    """Gemini assistant config and tool list (no network call; Gemini is optional)."""
    from grace.actions import create_fake_registry
    from grace.config import ConfigError, load_settings
    from grace.intent.assistant import registry_tools

    try:
        settings = load_settings(settings_path, env_path)
    except ConfigError:
        return Result("gemini assistant", NOT_CONFIGURED, "settings.yaml invalid (see above)")
    tools, names = registry_tools(create_fake_registry()[0])
    if any(create_fake_registry()[0].is_risky(a) for a in names.values()):
        return Result("gemini assistant", FAIL, "a risky action is exposed to the LLM")
    if "gemini" not in settings.enabled_providers():
        return Result("gemini assistant", NOT_CONFIGURED, "GEMINI_API_KEY not set in .env")
    g = settings.gemini
    extras = [n for n, on in (("web search", g.web_search), ("images", g.image_generation)) if on]
    detail = f"{g.model}, {len(tools)} action tools"
    if extras:
        detail += f"; {' + '.join(extras)} need a billing-enabled key"
    return Result("gemini assistant", PASS, detail)


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


# ---------------------------------------------------------------- runner


def run_checks(hardware: bool = True) -> list[Result]:
    checks: list[Callable[[], Result | list[Result]]] = [
        check_python,
        check_windows_build,
        check_uv_pip_check,
        check_imports,
        check_mediapipe_legacy_hands,
        check_tesseract,
        check_grammar,
        check_assistant,
        check_gestures,
        check_hotkeys,
        check_overlay,
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
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    args = parser.parse_args(argv)

    results = run_checks(hardware=not args.no_hardware)
    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
    else:
        print(format_results(results))
    return 1 if any(r.status == FAIL for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
