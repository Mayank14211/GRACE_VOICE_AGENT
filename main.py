#!/usr/bin/env python3
"""GRACE: kept so `python main.py` still works. The app lives in grace/ (`python -m grace`).

The old voice agent moved to grace/legacy_voice.py in S05.
"""

import sys

from grace.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
