"""Action layer: the only code that touches the OS. Build one registry per process."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grace.actions.apps import Launcher, WindowsLauncher, register_app_actions
from grace.actions.input_ctl import (
    ClipboardBackend,
    InputBackend,
    PyAutoGuiInput,
    PyperclipClipboard,
    register_input_actions,
)
from grace.actions.registry import (
    ActionRegistry,
    ActionResult,
    ConfirmationRequiredError,
)
from grace.actions.screen import (
    DEFAULT_SCREENSHOT_DIR,
    PyAutoGuiScreenshot,
    ScreenshotBackend,
    register_screen_actions,
)
from grace.actions.system import (
    BrightnessBackend,
    PowerBackend,
    PycawVolume,
    SbcBrightness,
    VolumeBackend,
    WindowsPower,
    register_system_actions,
)

__all__ = [
    "ActionRegistry",
    "ActionResult",
    "ConfirmationRequiredError",
    "create_registry",
]


def create_registry(
    *,
    volume: VolumeBackend | None = None,
    brightness: BrightnessBackend | None = None,
    power: PowerBackend | None = None,
    keyboard_mouse: InputBackend | None = None,
    clipboard: ClipboardBackend | None = None,
    launcher: Launcher | None = None,
    screenshots: ScreenshotBackend | None = None,
    screenshot_dir: Path = DEFAULT_SCREENSHOT_DIR,
    **input_options: Any,
) -> ActionRegistry:
    """Registry with every M2 action. Missing backends default to the real Windows ones."""
    keyboard_mouse = keyboard_mouse or PyAutoGuiInput()
    registry = ActionRegistry()
    register_system_actions(
        registry, volume or PycawVolume(), brightness or SbcBrightness(), power or WindowsPower()
    )
    register_input_actions(
        registry, keyboard_mouse, clipboard or PyperclipClipboard(), **input_options
    )
    launcher = launcher or WindowsLauncher()
    register_app_actions(registry, launcher, keyboard_mouse)
    register_screen_actions(
        registry, screenshots or PyAutoGuiScreenshot(), screenshot_dir, opener=launcher
    )
    return registry


def create_fake_registry(backends: Any = None, **kwargs: Any) -> tuple[ActionRegistry, Any]:
    """Registry wired to in-memory fakes (tests, `--dry-run` style checks)."""
    from grace.actions.fakes import FakeBackends

    b = backends or FakeBackends()
    registry = create_registry(
        volume=b.volume,
        brightness=b.brightness,
        power=b.power,
        keyboard_mouse=b.input,
        clipboard=b.clipboard,
        launcher=b.launcher,
        screenshots=b.screenshots,
        copy_wait_s=0.0,
        **kwargs,
    )
    return registry, b
