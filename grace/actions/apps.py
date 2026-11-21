"""App, website and window actions (ported from system_control.py / main.py).

Only names in the catalog below, or strings that look like a domain ("github.com",
"github dot com"), can be opened. The old `os.system(f"start {name}")` fallback for arbitrary
spoken text is gone: it passed user speech to a shell.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import webbrowser
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from pydantic import Field, field_validator

from grace.actions.input_ctl import InputBackend
from grace.actions.registry import ActionArgs, ActionError, ActionRegistry, ActionResult, NoArgs

# Spoken name -> command resolved by `start` (App Paths / PATH).
APPLICATIONS: Mapping[str, str] = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "firefox": "firefox",
    "notepad": "notepad",
    "note pad": "notepad",
    "calculator": "calc",
    "explorer": "explorer",
    "file explorer": "explorer",
    "files": "explorer",
    "word": "winword",
    "microsoft word": "winword",
    "excel": "excel",
    "microsoft excel": "excel",
    "powerpoint": "powerpnt",
    "power point": "powerpnt",
    "code": "code",
    "vs code": "code",
    "vscode": "code",
    "visual studio code": "code",
    "visual studio": "devenv",
    "terminal": "wt",
    "windows terminal": "wt",
    "command prompt": "cmd",
    "cmd": "cmd",
    "powershell": "powershell",
    "power shell": "powershell",
    "paint": "mspaint",
    "settings": "ms-settings:",
    "control panel": "control",
    "task manager": "taskmgr",
}

WEBSITES: Mapping[str, str] = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "facebook": "https://www.facebook.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "instagram": "https://www.instagram.com",
    "linkedin": "https://www.linkedin.com",
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
    "amazon": "https://www.amazon.com",
    "netflix": "https://www.netflix.com",
    "whatsapp": "https://web.whatsapp.com",
    "whatsapp web": "https://web.whatsapp.com",
    "spotify": "https://open.spotify.com",
}

_DOMAIN_RE = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}(?:/\S*)?")
_FILLER_RE = re.compile(r"^(?:the|my|a)\s+|\s+(?:app|application|website|site|browser)$")


def resolve_target(name: str) -> tuple[str, str] | None:
    """Spoken name -> ("app", command) | ("url", url) | None. Pure."""
    n = " ".join(name.lower().split())
    for _ in range(2):  # "the chrome browser" -> "chrome"
        n = _FILLER_RE.sub("", n).strip()
    if n in APPLICATIONS:
        return "app", APPLICATIONS[n]
    if n in WEBSITES:
        return "url", WEBSITES[n]
    domain = re.sub(r"\s*\bdot\b\s*", ".", n).replace(" ", "")
    if _DOMAIN_RE.fullmatch(domain):
        return "url", "https://" + domain
    return None


# --------------------------------------------------------------------------- argument models
class OpenArgs(ActionArgs):
    name: str = Field(min_length=1, max_length=100, description="App or website name, or a domain")


class UrlArgs(ActionArgs):
    url: str = Field(max_length=2000)

    @field_validator("url")
    @classmethod
    def _http_only(cls, v: str) -> str:
        parts = urlsplit(v.strip())
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("only http(s) URLs can be opened")
        return v.strip()


# --------------------------------------------------------------------------- interfaces
class Launcher(Protocol):
    def launch(self, command: str) -> None: ...
    def open_url(self, url: str) -> None: ...
    def open_file(self, path: Path) -> None: ...


class WindowsLauncher:
    def launch(self, command: str) -> None:
        if sys.platform != "win32":
            raise ActionError("Opening apps is only supported on Windows.")
        # `command` comes from the APPLICATIONS table only, never from speech.
        try:
            subprocess.Popen(
                ["cmd", "/c", "start", "", command],
                creationflags=subprocess.CREATE_NO_WINDOW,  # type: ignore[attr-defined]
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise ActionError(f"Couldn't start {command}.") from exc

    def open_url(self, url: str) -> None:
        if not webbrowser.open(url):
            raise ActionError("Couldn't open the browser.")

    def open_file(self, path: Path) -> None:
        """Open a file with its default app. Callers restrict which paths may get here."""
        if sys.platform != "win32":
            raise ActionError("Opening files is only supported on Windows.")
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except OSError as exc:
            raise ActionError("Couldn't open the file.") from exc


# --------------------------------------------------------------------------- registration
def register_app_actions(
    registry: ActionRegistry, launcher: Launcher, keyboard_mouse: InputBackend
) -> None:
    @registry.register("apps.open", OpenArgs, description="Open an app or website by name.")
    def open_app(args: OpenArgs) -> ActionResult:
        target = resolve_target(args.name)
        if target is None:
            raise ActionError(f"I don't know an app called {args.name}.")
        kind, value = target
        if kind == "app":
            launcher.launch(value)
        else:
            launcher.open_url(value)
        return ActionResult(True, f"Opening {args.name}.", {"kind": kind, "target": value})

    @registry.register("apps.open_url", UrlArgs, description="Open a web address.")
    def open_url(args: UrlArgs) -> ActionResult:
        launcher.open_url(args.url)
        return ActionResult(True, "Opening the link.", {"url": args.url})

    @registry.register("apps.close_window", NoArgs, description="Close the active window.")
    def close_window(_: NoArgs) -> ActionResult:
        keyboard_mouse.hotkey("alt", "f4")
        return ActionResult(True, "Closing the window.")

    @registry.register("apps.close_tab", NoArgs, description="Close the active tab.")
    def close_tab(_: NoArgs) -> ActionResult:
        keyboard_mouse.hotkey("ctrl", "w")
        return ActionResult(True, "Closing the tab.")
