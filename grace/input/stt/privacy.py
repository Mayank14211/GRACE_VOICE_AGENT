"""First-run notice: speech audio is sent to the configured cloud provider (docs/privacy.md).

    text = first_run_notice(uses_cloud=True, wake_uploads=False)   # str the first time, then None

Shown once (subtitle + console), never as a dialog: it must not block gestures or hotkeys. The
"already shown" marker lives next to the quota counters in GRACE's data folder (logs/, git-ignored;
Phase 2 moves both to the user's profile). A new ``NOTICE_VERSION`` shows it again, for when what
is sent changes.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from grace.core.quota import DEFAULT_STATE

log = logging.getLogger(__name__)

NOTICE_PATH = DEFAULT_STATE.parent / "privacy_notice.json"
NOTICE_VERSION = 1


def notice_text(*, wake_uploads: bool) -> str:
    """What the user is told. ``wake_uploads``: the spoken wake word is checked by the cloud
    engine (no wake-word model and no local engine), so speech before it is sent too."""
    when = (
        "everything you say while the mic is on (to listen for the wake word)"
        if wake_uploads
        else "what you say after the wake word or while holding push-to-talk"
    )
    return (
        f"Privacy: GRACE sends {when} to Groq to turn speech into text. Only detected speech is "
        "sent, never while the mic is off. Gestures, the camera and the wake-word model stay on "
        "this PC. Details: docs/privacy.md"
    )


def first_run_notice(
    *, uses_cloud: bool, wake_uploads: bool = False, path: Path = NOTICE_PATH
) -> str | None:
    """The notice if it has not been shown for this version yet (and remember that it has);
    None if nothing is sent to the cloud or it was shown before. Never raises."""
    if not uses_cloud:
        return None
    try:
        seen = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        seen = {}
    if isinstance(seen, dict) and seen.get("version") == NOTICE_VERSION:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"version": NOTICE_VERSION, "shown": datetime.now(UTC).isoformat()}),
            encoding="utf-8",
        )
    except OSError:
        log.warning("privacy notice: could not save %s; it will show again", path.name)
    return notice_text(wake_uploads=wake_uploads)
