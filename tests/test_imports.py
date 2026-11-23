"""Every remaining module imports without hardware (mic, camera, audio endpoint, display)."""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def fresh_import(name: str) -> ModuleType:
    sys.modules.pop(name, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # google-generativeai FutureWarning, protobuf notices
        return importlib.import_module(name)


@pytest.mark.parametrize(
    "module",
    ["tts", "speech_recognition_module", "screen_reader", "tesseract_check"],
)
def test_legacy_module_imports(module: str) -> None:
    assert fresh_import(module) is not None


@pytest.mark.windows
def test_main_imports() -> None:
    main = fresh_import("main")
    import grace.__main__

    assert main.main is grace.__main__.main  # main.py is a thin shim (S05)
    assert hasattr(fresh_import("grace.legacy_voice"), "GraceVoiceAgent")


def test_runtime_modules_import_without_hardware() -> None:
    # Bus, state, hotkeys, runtime and the voice worker must not pull in Qt, audio or pyautogui.
    code = (
        "import sys; import grace.core.bus, grace.core.state, grace.input.hotkeys, "
        "grace.input.voice, grace.runtime, grace.metrics.log, grace.feedback.earcons; "
        "bad = [m for m in ('PySide6', 'pyautogui', 'speech_recognition', 'sounddevice', "
        "'keyboard') if m in sys.modules]; assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_gesture_engine_imports_without_camera_or_mediapipe() -> None:
    # The pure modules must not pull in cv2 / mediapipe (lazy in the tracker) or touch the OS.
    code = (
        "import sys; import grace.input.gestures, grace.input.gestures.tracker, "
        "grace.actions.executor; "
        "bad = [m for m in ('cv2', 'mediapipe', 'pyautogui') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


@pytest.mark.windows
def test_main_help_runs_without_image_generation() -> None:
    """The old entry point still starts; image-generation flags are gone (M0)."""
    proc = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--wake-word" in proc.stdout
    for removed in ("--stability-key", "--huggingface-token"):
        assert removed not in proc.stdout


def test_doctor_imports() -> None:
    assert callable(fresh_import("grace.doctor").main)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        shim = fresh_import("doctor")  # scripts/doctor.py still works
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    assert shim.main.__module__ == "grace.doctor"
