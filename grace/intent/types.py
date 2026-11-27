"""Intent: what the user asked for, independent of how it was recognised (grammar or LLM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Intents handled by the app itself rather than the action registry (they don't touch the OS).
APP_INTENTS = frozenset(
    {
        "app.confirm",
        "app.cancel",
        "app.help",
        "app.exit",
        "app.end_session",
        "app.reset_conversation",
        "app.dictation_start",
        "app.dictation_stop",
        "app.read_screen",
        "app.read_selection",
        "app.analyze_code",
        "app.analyze_screen",
        "app.debug_code",
        # Router-only (S07): a spoken answer, and the Gemini extras.
        "app.answer",
        "app.web_search",
        "app.generate_image",
    }
)

# grammar: fixed command; followup: "a bit more" / "undo that" resolved from the last action;
# router: chosen by an LLM.
Source = Literal["grammar", "followup", "router", "gesture", "hotkey"]


@dataclass(frozen=True)
class Intent:
    name: str  # an action registry name, or one of APP_INTENTS
    args: dict[str, Any] = field(default_factory=dict, hash=False)
    text: str = ""  # the normalised utterance it came from
    source: Source = "grammar"
