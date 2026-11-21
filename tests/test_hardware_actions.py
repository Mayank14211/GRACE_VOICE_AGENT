"""M2 acceptance on the base machine: volume and brightness reach the exact requested values.

Run manually (changes volume / brightness briefly, then restores them):
    uv run pytest -m hardware tests/test_hardware_actions.py -v
"""

from __future__ import annotations

import threading
import time

import pytest

pytestmark = [pytest.mark.hardware, pytest.mark.windows]

VOLUME_LEVELS = (37, 63, 5, 0, 81)
BRIGHTNESS_LEVELS = (42, 70, 15)


def _poll(read, expected: int, timeout_s: float = 1.5) -> int:
    deadline = time.monotonic() + timeout_s
    value = read()
    while value != expected and time.monotonic() < deadline:
        time.sleep(0.05)
        value = read()
    return value


def test_volume_exact_levels() -> None:
    from grace.actions.system import PycawVolume

    vol = PycawVolume()
    original, muted = vol.get_level(), vol.is_muted()
    try:
        for level in VOLUME_LEVELS:
            vol.set_level(level)
            assert vol.get_level() == level
    finally:
        vol.set_level(original)
        vol.set_muted(muted)


def test_volume_from_worker_thread() -> None:
    """COM is initialised per call, so a thread that never called CoInitialize works."""
    from grace.actions.system import PycawVolume

    vol = PycawVolume()
    original = vol.get_level()
    seen: list[int] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            vol.set_level(44)
            seen.append(vol.get_level())
        except BaseException as exc:  # surfaced in the main thread below
            errors.append(exc)

    t = threading.Thread(target=work)
    t.start()
    t.join(10)
    vol.set_level(original)
    assert not errors and seen == [44]


def test_registry_volume_set_and_step() -> None:
    from grace.actions import create_registry
    from grace.actions.system import PycawVolume

    vol = PycawVolume()
    original, muted = vol.get_level(), vol.is_muted()
    registry = create_registry()
    try:
        assert registry.execute("volume.set", {"level": 40}).ok
        assert vol.get_level() == 40
        assert registry.execute("volume.step", {"delta": 15}).ok
        assert vol.get_level() == 55
    finally:
        vol.set_level(original)
        vol.set_muted(muted)


def test_brightness_exact_levels() -> None:
    from grace.actions.system import SbcBrightness

    b = SbcBrightness()
    original = b.get_level()
    try:
        for level in BRIGHTNESS_LEVELS:
            b.set_level(level)
            assert _poll(b.get_level, level) == level
    finally:
        b.set_level(original)
