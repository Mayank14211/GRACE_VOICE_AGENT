"""Moved to grace/doctor.py in S05: use `uv run python -m grace doctor`. This shim keeps
`uv run python scripts/doctor.py [--no-hardware] [--json]` working."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grace.doctor import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
