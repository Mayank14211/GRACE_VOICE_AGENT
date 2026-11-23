from __future__ import annotations

import os
import sys

import pytest

# Qt tests run headless (pytest-qt, overlay smoke test): no window ever appears on screen.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if sys.platform == "win32":
        return
    skip = pytest.mark.skip(reason="Windows-only")
    for item in items:
        if "windows" in item.keywords:
            item.add_marker(skip)
