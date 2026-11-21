"""Screen actions: screenshot, and showing an image GRACE saved.

OCR "read screen" moves here later.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import Field

from grace.actions.registry import ActionArgs, ActionError, ActionRegistry, ActionResult, NoArgs

DEFAULT_SCREENSHOT_DIR = Path.home() / "Pictures" / "GRACE"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


class ScreenshotBackend(Protocol):
    def save(self, path: Path) -> None: ...


class FileOpener(Protocol):
    def open_file(self, path: Path) -> None: ...


class PyAutoGuiScreenshot:
    def save(self, path: Path) -> None:
        import pyautogui

        pyautogui.screenshot().save(path)


class ImagePathArgs(ActionArgs):
    path: str = Field(min_length=1, max_length=500)


def register_screen_actions(
    registry: ActionRegistry,
    screenshots: ScreenshotBackend,
    folder: Path = DEFAULT_SCREENSHOT_DIR,
    now: Callable[[], float] = time.time,
    opener: FileOpener | None = None,
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

    @registry.register(
        "screen.show_image", ImagePathArgs, description="Open an image GRACE saved in its folder."
    )
    def show_image(args: ImagePathArgs) -> ActionResult:
        path = Path(args.path).resolve()
        # Only images GRACE itself saved: never an arbitrary path from speech or the LLM.
        if not path.is_relative_to(folder.resolve()) or path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ActionError("I can only open images in GRACE's own picture folder.")
        if not path.is_file():
            raise ActionError("That image doesn't exist.")
        if opener is None:
            raise ActionError("Opening images isn't available.")
        opener.open_file(path)
        return ActionResult(True, "Showing the image.", {"path": str(path)})
