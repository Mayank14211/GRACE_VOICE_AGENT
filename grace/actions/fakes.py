"""In-memory fakes for every OS backend. Used by tests; safe on any OS, touch nothing real."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FakeVolume:
    level: int = 50
    muted: bool = False
    calls: list[tuple[str, object]] = field(default_factory=list)

    def get_level(self) -> int:
        return self.level

    def set_level(self, level: int) -> None:
        self.calls.append(("set_level", level))
        self.level = level

    def is_muted(self) -> bool:
        return self.muted

    def set_muted(self, muted: bool) -> None:
        self.calls.append(("set_muted", muted))
        self.muted = muted


@dataclass
class FakeBrightness:
    level: int = 50
    calls: list[int] = field(default_factory=list)

    def get_level(self) -> int:
        return self.level

    def set_level(self, level: int) -> None:
        self.calls.append(level)
        self.level = level


@dataclass
class FakePower:
    calls: list[str] = field(default_factory=list)

    def shutdown(self) -> None:
        self.calls.append("shutdown")

    def restart(self) -> None:
        self.calls.append("restart")

    def sleep(self) -> None:
        self.calls.append("sleep")


@dataclass
class FakeInput:
    calls: list[tuple] = field(default_factory=list)

    def click(self, button: str, clicks: int) -> None:
        self.calls.append(("click", button, clicks))

    def scroll(self, amount: int) -> None:
        self.calls.append(("scroll", amount))

    def write(self, text: str) -> None:
        self.calls.append(("write", text))

    def hotkey(self, *keys: str) -> None:
        self.calls.append(("hotkey", *keys))


@dataclass
class FakeClipboard:
    text: str = ""
    history: list[str] = field(default_factory=list)

    def get(self) -> str:
        return self.text

    def set(self, text: str) -> None:
        self.history.append(text)
        self.text = text


@dataclass
class FakeLauncher:
    launched: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)

    def launch(self, command: str) -> None:
        self.launched.append(command)

    def open_url(self, url: str) -> None:
        self.urls.append(url)


@dataclass
class FakeScreenshot:
    saved: list[Path] = field(default_factory=list)

    def save(self, path: Path) -> None:
        self.saved.append(path)


@dataclass
class FakeBackends:
    volume: FakeVolume = field(default_factory=FakeVolume)
    brightness: FakeBrightness = field(default_factory=FakeBrightness)
    power: FakePower = field(default_factory=FakePower)
    input: FakeInput = field(default_factory=FakeInput)
    clipboard: FakeClipboard = field(default_factory=FakeClipboard)
    launcher: FakeLauncher = field(default_factory=FakeLauncher)
    screenshots: FakeScreenshot = field(default_factory=FakeScreenshot)
