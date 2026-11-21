"""Mouse, keyboard and clipboard actions (ported from system_control.py).

The real backends import pyautogui / pyperclip lazily so tests and headless machines can import
this module. pyautogui's fail-safe (mouse in a screen corner aborts) stays on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import Field, field_validator

from grace.actions.registry import ActionArgs, ActionError, ActionRegistry, ActionResult, NoArgs

WHEEL_DELTA = 120  # Windows: one wheel notch; pyautogui.scroll() passes the raw delta through

MODIFIER_KEYS = ("ctrl", "alt", "shift", "win")
NAMED_KEYS = (
    "enter",
    "esc",
    "tab",
    "space",
    "backspace",
    "delete",
    "insert",
    "home",
    "end",
    "pageup",
    "pagedown",
    "up",
    "down",
    "left",
    "right",
    "capslock",
    "printscreen",
    *(f"f{i}" for i in range(1, 13)),
)
# Every name here is also a valid pyautogui key name.
ALLOWED_KEYS = frozenset(
    (*MODIFIER_KEYS, *NAMED_KEYS, *"abcdefghijklmnopqrstuvwxyz", *"0123456789")
)
KEY_ALIASES = {
    "control": "ctrl",
    "ctl": "ctrl",
    "windows": "win",
    "escape": "esc",
    "return": "enter",
    "del": "delete",
    "page up": "pageup",
    "page down": "pagedown",
    "spacebar": "space",
}


# --------------------------------------------------------------------------- argument models
class ClickArgs(ActionArgs):
    button: Literal["left", "right", "middle"] = "left"
    count: int = Field(default=1, ge=1, le=3)


class ScrollArgs(ActionArgs):
    direction: Literal["up", "down"] = "down"
    notches: int = Field(default=3, description="Wheel notches, clamped to 1-50.")

    @field_validator("notches")
    @classmethod
    def _clamp(cls, v: int) -> int:
        return max(1, min(50, v))


class TextArgs(ActionArgs):
    text: str = Field(min_length=1, max_length=5000)


class HotkeyArgs(ActionArgs):
    keys: list[str] = Field(min_length=1, max_length=4)

    @field_validator("keys")
    @classmethod
    def _known_keys(cls, keys: list[str]) -> list[str]:
        out = []
        for key in keys:
            k = KEY_ALIASES.get(key.strip().lower(), key.strip().lower())
            if k not in ALLOWED_KEYS:
                raise ValueError(f"unknown key {key!r}")
            out.append(k)
        if len(set(out)) != len(out):
            raise ValueError("a key is repeated")
        return out


# --------------------------------------------------------------------------- interfaces
class InputBackend(Protocol):
    def click(self, button: str, clicks: int) -> None: ...
    def scroll(self, amount: int) -> None: ...
    def write(self, text: str) -> None: ...
    def hotkey(self, *keys: str) -> None: ...


class ClipboardBackend(Protocol):
    def get(self) -> str: ...
    def set(self, text: str) -> None: ...


class PyAutoGuiInput:
    def __init__(self) -> None:
        import pyautogui

        pyautogui.PAUSE = 0.02  # default 0.1 s after every call makes key combos sluggish
        self._gui = pyautogui

    def click(self, button: str, clicks: int) -> None:
        self._gui.click(button=button, clicks=clicks, interval=0.05)

    def scroll(self, amount: int) -> None:
        self._gui.scroll(amount)

    def write(self, text: str) -> None:
        self._gui.write(text)

    def hotkey(self, *keys: str) -> None:
        if len(keys) == 1:
            self._gui.press(keys[0])
        else:
            self._gui.hotkey(*keys)


class PyperclipClipboard:
    def get(self) -> str:
        import pyperclip

        return pyperclip.paste() or ""

    def set(self, text: str) -> None:
        import pyperclip

        pyperclip.copy(text)


# --------------------------------------------------------------------------- registration
def register_input_actions(
    registry: ActionRegistry,
    keyboard_mouse: InputBackend,
    clipboard: ClipboardBackend,
    *,
    copy_wait_s: float = 0.3,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    @registry.register("input.click", ClickArgs, description="Click the mouse where it is.")
    def click(args: ClickArgs) -> ActionResult:
        keyboard_mouse.click(args.button, args.count)
        words = [{1: "", 2: "double", 3: "triple"}[args.count]]
        words += [args.button if args.button != "left" else "", "click"]
        return ActionResult(True, " ".join(w for w in words if w).capitalize() + ".")

    @registry.register("input.scroll", ScrollArgs, description="Scroll the window under the mouse.")
    def scroll(args: ScrollArgs) -> ActionResult:
        sign = 1 if args.direction == "up" else -1
        keyboard_mouse.scroll(sign * args.notches * WHEEL_DELTA)
        return ActionResult(True, f"Scrolled {args.direction}.", {"notches": args.notches})

    @registry.register("input.type", TextArgs, description="Type text at the cursor.")
    def type_text(args: TextArgs) -> ActionResult:
        if args.text.isascii():
            keyboard_mouse.write(args.text)
        else:  # pyautogui.write silently drops non-ASCII; paste it, then restore the clipboard
            saved = clipboard.get()
            clipboard.set(args.text)
            keyboard_mouse.hotkey("ctrl", "v")
            sleep(copy_wait_s)
            clipboard.set(saved)
        return ActionResult(True, "Typed.", {"chars": len(args.text)})

    @registry.register("input.hotkey", HotkeyArgs, description="Press a key or key combination.")
    def hotkey(args: HotkeyArgs) -> ActionResult:
        keyboard_mouse.hotkey(*args.keys)
        return ActionResult(True, f"Pressed {' plus '.join(args.keys)}.", {"keys": args.keys})

    @registry.register("clipboard.copy", NoArgs, description="Copy the selection.")
    def copy(_: NoArgs) -> ActionResult:
        keyboard_mouse.hotkey("ctrl", "c")
        sleep(copy_wait_s)  # let the target app update the clipboard
        text = clipboard.get()
        if not text.strip():
            return ActionResult(False, "No text selected to copy.", {"text": ""})
        return ActionResult(True, "Copied.", {"text": text})

    @registry.register("clipboard.paste", NoArgs, description="Paste the clipboard.")
    def paste(_: NoArgs) -> ActionResult:
        keyboard_mouse.hotkey("ctrl", "v")
        return ActionResult(True, "Pasted.")

    @registry.register("clipboard.set", TextArgs, description="Put text on the clipboard.")
    def set_clipboard(args: TextArgs) -> ActionResult:
        try:
            clipboard.set(args.text)
        except Exception as exc:  # pyperclip.PyperclipException
            raise ActionError("Couldn't use the clipboard.") from exc
        return ActionResult(True, "Copied to the clipboard.")
