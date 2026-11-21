"""Screen actions. M2: screenshot only (OCR "read screen" moves here later)."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from grace.actions.registry import ActionError, ActionRegistry, ActionResult, NoArgs

DEFAULT_SCREENSHOT_DIR = Path.home() / "Pictures" / "GRACE"


class ScreenshotBackend(Protocol):
    def save(self, path: Path) -> None: ...


class PyAutoGuiScreenshot:
    def save(self, path: Path) -> None:
        import pyautogui

        pyautogui.screenshot().save(path)


def register_screen_actions(
    registry: ActionRegistry,
    screenshots: ScreenshotBackend,
    folder: Path = DEFAULT_SCREENSHOT_DIR,
    now: Callable[[], float] = time.time,
) -> None:
    @registry.register("screen.screenshot", NoArgs, description="Save a screenshot.")
    def screenshot(_: NoArgs) -> ActionResult:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now()))
        path = folder / f"screenshot_{stamp}.png"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            screenshots.save(path)
        except OSError as exc:
            raise ActionError("Couldn't save the screenshot.") from exc
        return ActionResult(True, f"Screenshot saved to {folder.name}.", {"path": str(path)})
