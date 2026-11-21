"""System actions: volume (pycaw), brightness (screen-brightness-control), power (risky).

Backends are small interfaces so tests use the fakes in grace/actions/fakes.py. The real backends
import their Windows libraries lazily: a missing package disables one action instead of startup.

COM rule (docs/PLAN.md): any thread that touches pycaw calls comtypes.CoInitialize() first and
CoUninitialize() on exit. PycawVolume does both around every call and releases the endpoint
before uninitialising, so it is safe from any thread (voice, hotkey, gesture, executor).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol, TypeVar

from pydantic import field_validator

from grace.actions.registry import (
    ActionArgs,
    ActionError,
    ActionRegistry,
    ActionResult,
    NoArgs,
    clamp,
)

T = TypeVar("T")


# --------------------------------------------------------------------------- argument models
class LevelArgs(ActionArgs):
    """An absolute level in percent. Out-of-range values are clamped to 0-100."""

    level: int

    @field_validator("level")
    @classmethod
    def _clamp(cls, v: int) -> int:
        return clamp(v, 0, 100)


class StepArgs(ActionArgs):
    """A relative change in percentage points (negative = down), clamped to -100..100."""

    delta: int

    @field_validator("delta")
    @classmethod
    def _clamp(cls, v: int) -> int:
        return clamp(v, -100, 100)


class MuteArgs(ActionArgs):
    mute: bool = True


# --------------------------------------------------------------------------- interfaces
class VolumeBackend(Protocol):
    def get_level(self) -> int: ...
    def set_level(self, level: int) -> None: ...
    def is_muted(self) -> bool: ...
    def set_muted(self, muted: bool) -> None: ...


class BrightnessBackend(Protocol):
    def get_level(self) -> int: ...
    def set_level(self, level: int) -> None: ...


class PowerBackend(Protocol):
    def shutdown(self) -> None: ...
    def restart(self) -> None: ...
    def sleep(self) -> None: ...


# --------------------------------------------------------------------------- real backends
RPC_E_CHANGED_MODE = -2147417850  # 0x80010106: thread already initialised with another model


@contextmanager
def com_initialized() -> Iterator[None]:
    """CoInitialize for the current thread, balanced with CoUninitialize on exit."""
    import comtypes

    try:
        comtypes.CoInitialize()
    except OSError as exc:
        if getattr(exc, "winerror", None) != RPC_E_CHANGED_MODE:
            raise
        yield  # already initialised (e.g. MTA) by someone else; they own CoUninitialize
        return
    try:
        yield
    finally:
        comtypes.CoUninitialize()


def _default_endpoint() -> Any:
    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    speakers = AudioUtilities.GetSpeakers()  # default render endpoint (re-read: devices change)
    interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


class PycawVolume:
    """Master volume of the default playback device, exact to 1 %."""

    def __init__(
        self,
        endpoint_factory: Callable[[], Any] = _default_endpoint,
        com: Callable[[], Any] = com_initialized,
    ) -> None:
        self._endpoint_factory = endpoint_factory
        self._com = com

    def _call(self, op: Callable[[Any], T]) -> T:
        error: str | None = None
        with self._com():
            endpoint = self._endpoint_factory()
            try:
                return op(endpoint)
            except Exception as exc:  # COMError / OSError from the audio stack
                error = f"{type(exc).__name__}: {exc}"
            finally:
                # Drop the COM pointer before CoUninitialize. `exc` (and its traceback, which
                # references `endpoint`) is already cleared when the except block ends.
                endpoint = None
        raise ActionError(f"Volume control failed ({error}).")

    def get_level(self) -> int:
        return round(self._call(lambda ep: ep.GetMasterVolumeLevelScalar()) * 100)

    def set_level(self, level: int) -> None:
        self._call(lambda ep: ep.SetMasterVolumeLevelScalar(clamp(level, 0, 100) / 100, None))

    def is_muted(self) -> bool:
        return bool(self._call(lambda ep: ep.GetMute()))

    def set_muted(self, muted: bool) -> None:
        self._call(lambda ep: ep.SetMute(int(muted), None))


class SbcBrightness:
    """Brightness of one display (default: the first one, the laptop panel) via sbc."""

    def __init__(self, display: int = 0) -> None:
        self.display = display

    def _sbc(self) -> Any:
        try:
            import screen_brightness_control as sbc
        except ImportError as exc:
            raise ActionError("Brightness control is not installed.") from exc
        return sbc

    def get_level(self) -> int:
        sbc = self._sbc()
        try:
            levels = sbc.get_brightness(display=self.display)
        except Exception as exc:  # ScreenBrightnessError, WMI errors
            raise ActionError(f"Couldn't read the brightness ({type(exc).__name__}).") from exc
        if not levels or levels[0] is None:
            raise ActionError("This display doesn't report its brightness.")
        return int(levels[0])

    def set_level(self, level: int) -> None:
        sbc = self._sbc()
        try:
            sbc.set_brightness(clamp(level, 0, 100), display=self.display)
        except Exception as exc:
            raise ActionError(f"Couldn't set the brightness ({type(exc).__name__}).") from exc


class WindowsPower:
    """Power actions. No /f: apps with unsaved work may still block the shutdown."""

    def __init__(self, run: Callable[..., Any] = subprocess.run) -> None:
        self._run = run

    def _shutdown_exe(self, *args: str) -> None:
        if sys.platform != "win32":
            raise ActionError("Power actions are only supported on Windows.")
        try:
            self._run(["shutdown", *args], check=True, timeout=10, capture_output=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ActionError(f"The power command failed ({type(exc).__name__}).") from exc

    def shutdown(self) -> None:
        self._shutdown_exe("/s", "/t", "0")

    def restart(self) -> None:
        self._shutdown_exe("/r", "/t", "0")

    def sleep(self) -> None:
        if sys.platform != "win32":
            raise ActionError("Power actions are only supported on Windows.")
        import ctypes

        # SetSuspendState(bHibernate, bForce, bWakeupEventsDisabled). Called directly rather than
        # via rundll32, which passes garbage arguments and can hibernate instead of sleeping.
        if not ctypes.windll.powrprof.SetSuspendState(False, False, False):
            raise ActionError("Windows refused to sleep.")


# --------------------------------------------------------------------------- registration
def register_system_actions(
    registry: ActionRegistry,
    volume: VolumeBackend,
    brightness: BrightnessBackend,
    power: PowerBackend,
) -> None:
    @registry.register("volume.get", NoArgs, description="Report the current volume.")
    def volume_get(_: NoArgs) -> ActionResult:
        level, muted = volume.get_level(), volume.is_muted()
        state = " and muted" if muted else ""
        return ActionResult(
            True, f"Volume is {level} percent{state}.", {"level": level, "muted": muted}
        )

    @registry.register("volume.set", LevelArgs, description="Set the volume to a percentage.")
    def volume_set(args: LevelArgs) -> ActionResult:
        volume.set_level(args.level)
        if args.level > 0 and volume.is_muted():
            volume.set_muted(False)  # like dragging the Windows slider
        return ActionResult(True, f"Volume set to {args.level} percent.", {"level": args.level})

    @registry.register("volume.step", StepArgs, description="Change the volume by some points.")
    def volume_step(args: StepArgs) -> ActionResult:
        level = clamp(volume.get_level() + args.delta, 0, 100)
        volume.set_level(level)
        if args.delta > 0 and volume.is_muted():
            volume.set_muted(False)
        return ActionResult(True, f"Volume {level} percent.", {"level": level})

    @registry.register("volume.mute", MuteArgs, description="Mute or unmute the speakers.")
    def volume_mute(args: MuteArgs) -> ActionResult:
        volume.set_muted(args.mute)
        return ActionResult(True, "Muted." if args.mute else "Unmuted.", {"muted": args.mute})

    @registry.register("brightness.get", NoArgs, description="Report the screen brightness.")
    def brightness_get(_: NoArgs) -> ActionResult:
        level = brightness.get_level()
        return ActionResult(True, f"Brightness is {level} percent.", {"level": level})

    @registry.register("brightness.set", LevelArgs, description="Set screen brightness (percent).")
    def brightness_set(args: LevelArgs) -> ActionResult:
        brightness.set_level(args.level)
        return ActionResult(True, f"Brightness set to {args.level} percent.", {"level": args.level})

    @registry.register("brightness.step", StepArgs, description="Change brightness by some points.")
    def brightness_step(args: StepArgs) -> ActionResult:
        level = clamp(brightness.get_level() + args.delta, 0, 100)
        brightness.set_level(level)
        return ActionResult(True, f"Brightness {level} percent.", {"level": level})

    @registry.register(
        "system.shutdown",
        NoArgs,
        risky=True,
        confirm_label="shutdown",
        description="Shut down the computer.",
    )
    def shutdown(_: NoArgs) -> ActionResult:
        power.shutdown()
        return ActionResult(True, "Shutting down.")

    @registry.register(
        "system.restart",
        NoArgs,
        risky=True,
        confirm_label="restart",
        description="Restart the computer.",
    )
    def restart(_: NoArgs) -> ActionResult:
        power.restart()
        return ActionResult(True, "Restarting.")

    @registry.register(
        "system.sleep",
        NoArgs,
        risky=True,
        confirm_label="sleep",
        description="Put the computer to sleep.",
    )
    def sleep(_: NoArgs) -> ActionResult:
        power.sleep()
        return ActionResult(True, "Going to sleep.")
