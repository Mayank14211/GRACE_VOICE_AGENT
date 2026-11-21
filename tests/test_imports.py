"""Every remaining module imports without hardware (mic, camera, audio endpoint, display)."""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

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
    assert hasattr(main, "GraceVoiceAgent")


@pytest.mark.windows
def test_hand_gesture_imports_without_hardware(mocker) -> None:
    # Module level calls pyautogui.size() and opens the default speaker via pycaw.
    mocker.patch("pyautogui.size", return_value=(1920, 1080))
    speakers = MagicMock()
    mocker.patch("pycaw.pycaw.AudioUtilities.GetSpeakers", return_value=speakers)
    mocker.patch("ctypes.cast", return_value=MagicMock())

    module = fresh_import("Hand_Gesture")

    assert (module.screen_width, module.screen_height) == (1920, 1080)
    speakers.Activate.assert_called_once()
    assert hasattr(module, "GestureController")


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
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        doctor = fresh_import("doctor")
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    assert callable(doctor.main)
