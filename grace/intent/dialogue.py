"""Dialogue state and deterministic follow-ups.

``Dialogue`` keeps the last N user / GRACE exchanges (sent to the LLM router as context) and the
last action that ran, with the value it replaced, so follow-ups work:

    "volume 40"        -> volume.set 40          (grammar)
    "a bit more"       -> volume.step +5         (follow-up, no LLM)
    "now the brightness" -> brightness.step +5   (follow-up)
    "undo that"        -> brightness.step -5     (follow-up; plain "undo" stays Ctrl+Z)

``resolve_followup`` is pure. Like the grammar, it matches the WHOLE normalised utterance
against a closed set of follow-up forms (no keyword-contains matching), and it never repeats or
reverses a risky action. Anything it can't resolve goes to the router, which gets the same
context as a line in the user message, so the LLM can resolve the rest ("make the second one
louder").
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from grace.intent.types import Intent

# A `.set` action and the action that reads the value it replaces (for "undo that").
READ_BEFORE_SET = {"volume.set": "volume.get", "brightness.set": "brightness.get"}
DEFAULT_STEPS = {"small": 5, "normal": 10, "large": 30}
DEFAULT_SCROLL = {"small": 1, "normal": 3, "large": 10}


@dataclass(frozen=True)
class Exchange:
    user: str
    grace: str
    t: float


@dataclass(frozen=True)
class LastAction:
    intent: Intent
    before: Mapping[str, Any] = field(default_factory=dict)  # e.g. {"level": 70} before a set
    t: float = 0.0
    exchange: int = 0  # number of exchanges finished before the one that ran it


class Dialogue:
    """Conversation memory for the router and follow-ups. Safe to call from any thread."""

    def __init__(
        self,
        *,
        max_turns: int = 6,
        memory_reset_s: float = 600.0,
        followup_s: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_turns = max_turns
        self.memory_reset_s = memory_reset_s
        self.followup_s = followup_s
        self._clock = clock
        self._lock = threading.Lock()
        self._turns: deque[Exchange] = deque(maxlen=max(max_turns, 1))
        self._last: LastAction | None = None
        self._exchanges = 0  # finished exchanges, ever (follow-ups only refer to the latest)

    def _expire(self, now: float) -> None:
        if self._turns and self.memory_reset_s and now - self._turns[-1].t > self.memory_reset_s:
            self._turns.clear()

    def add(self, user: str, grace: str) -> None:
        user, grace = user.strip(), grace.strip()
        if not user:
            return
        with self._lock:
            self._exchanges += 1
            if self.max_turns == 0:
                return
            now = self._clock()
            self._expire(now)
            self._turns.append(Exchange(user, grace, now))

    def record(self, intent: Intent, before: Mapping[str, Any] | None = None) -> None:
        """An action ran successfully: it becomes what follow-ups refer to."""
        with self._lock:
            self._last = LastAction(intent, dict(before or {}), self._clock(), self._exchanges)

    def last(self) -> LastAction | None:
        """The action from the latest exchange (or the one in progress). None once anything
        else was said since ("volume 40", "what time is it", "a bit more" is not a follow-up)
        or after followup_s."""
        with self._lock:
            last = self._last
            if last is None or self._clock() - last.t > self.followup_s:
                return None
            if self._exchanges - last.exchange > 1:
                return None
            return last

    def history(self) -> list[Exchange]:
        with self._lock:
            self._expire(self._clock())
            return list(self._turns)

    def messages(self) -> list[dict[str, str]]:
        """History as chat messages (oldest first), for the router."""
        out: list[dict[str, str]] = []
        for ex in self.history():
            out.append({"role": "user", "content": ex.user})
            if ex.grace:
                out.append({"role": "assistant", "content": ex.grace})
        return out

    def reset(self) -> None:
        with self._lock:
            self._turns.clear()
            self._last = None


# --------------------------------------------------------------------------- follow-ups
_SMALL = r"a little bit|a little|a bit|a tad|slightly|little|bit"
_LARGE = r"a lot|lots|much|way|loads|a great deal"
_SIZE = rf"(?P<size>{_SMALL}|{_LARGE}|even|some)"
_LEAD = r"(?:(?:and|now|ok|okay|then|so|also|just) )*"
_TAIL = r"(?: (?:please|now|then|too|again))?"
_TARGETS = {
    "volume": "volume",
    "sound": "volume",
    "the volume": "volume",
    "the sound": "volume",
    "brightness": "brightness",
    "the brightness": "brightness",
    "screen brightness": "brightness",
    "the screen brightness": "brightness",
}
_TARGET = "|".join(sorted(map(re.escape, _TARGETS), key=len, reverse=True))

_FORMS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (kind, re.compile(rf"{_LEAD}(?:{body}){_TAIL}"))
    for kind, body in (
        ("repeat", r"again|do (?:it|that) again|repeat(?: that| it)?|once more|(?:one|1) more time"
                   r"|same again|(?:do )?the same(?: again)?"),
        ("more", rf"(?:{_SIZE} )?more|keep going|go on|more more"),
        ("less", rf"(?:{_SIZE} )?less|not so much|too much|back (?:it )?off(?: a bit)?"),
        ("dir", rf"(?:(?:turn|move|take|bring|push|put|go|make) (?:it )?)?(?P<dir>up|down|higher"
                rf"|lower)(?: {_SIZE})?"),
        ("undo", r"(?:undo|revert|reverse) (?:that|it|this|the last (?:one|thing|change)"
                 r"|what you (?:just )?did)|put it back|change it back|set it back"
                 r"|back to (?:how|what) it was"),
        ("level", r"(?:(?:make|set|change|put|turn) it (?:to |at )?|to |at )?(?P<level>\d{1,3})"
                  r"(?: percent)?"),
        ("transfer", rf"(?:(?:and|now|also|what about|how about|same for|same with|the same for"
                     rf"|do the same for|do the same with) )+(?P<target>{_TARGET})"
                     rf"(?: (?:too|as well|also))?"
                     rf"|(?P<target_b>{_TARGET}) (?:too|as well|also)"),
    )
)  # fmt: skip
UNDO_FORM = _FORMS[4][1]


def is_undo_phrase(normalized: str) -> bool:
    """ "undo that" and friends (they outrank the grammar's Ctrl+Z when the last action can be
    reversed). Bare "undo" is not one: it always means Ctrl+Z."""
    return UNDO_FORM.fullmatch(normalized) is not None


def _target(name: str) -> str | None:
    if name.startswith("volume."):
        return "volume"
    if name.startswith("brightness."):
        return "brightness"
    if name == "input.scroll":
        return "scroll"
    return None


def _size(word: str | None, sizes: Mapping[str, int], default: int) -> int:
    if not word or word in ("even", "some"):
        return default
    return sizes["small"] if re.fullmatch(_SMALL, word) else sizes["large"]


def _sign(intent: Intent) -> int | None:
    """Direction of the last change: +1 up, -1 down, None when it has none (get, mute)."""
    if intent.name.endswith(".step"):
        delta = int(intent.args.get("delta", 0))
        return (delta > 0) - (delta < 0) or None
    if intent.name.endswith(".set"):
        return 1
    if intent.name == "input.scroll":
        return 1 if intent.args.get("direction", "down") == "up" else -1
    return None


def _change(target: str, sign: int, amount: int) -> Intent:
    if target == "scroll":
        return Intent(
            "input.scroll", {"direction": "up" if sign > 0 else "down", "notches": amount}
        )
    return Intent(f"{target}.step", {"delta": sign * amount})


def _undo(last: LastAction) -> Intent | None:
    intent = last.intent
    if intent.name.endswith(".step") and _target(intent.name):
        return Intent(intent.name, {"delta": -int(intent.args.get("delta", 0))})
    if intent.name in READ_BEFORE_SET and "level" in last.before:
        return Intent(intent.name, {"level": int(last.before["level"])})
    if intent.name == "volume.mute":
        return Intent("volume.mute", {"mute": not intent.args.get("mute", True)})
    if intent.name == "input.scroll":
        back = "down" if intent.args.get("direction", "down") == "up" else "up"
        return Intent("input.scroll", {"direction": back, "notches": intent.args.get("notches", 3)})
    return None


def resolve_followup(
    normalized: str,
    last: LastAction | None,
    *,
    is_risky: Callable[[str], bool] = lambda _name: False,
    steps: Mapping[str, int] = DEFAULT_STEPS,
    scroll: Mapping[str, int] = DEFAULT_SCROLL,
) -> Intent | None:
    """A follow-up to ``last`` as a new Intent, or None. ``normalized`` is grammar-normalised
    text (lower case, digits, fillers stripped). Pure."""
    if last is None or not normalized or is_risky(last.intent.name):
        return None
    for kind, pattern in _FORMS:
        m = pattern.fullmatch(normalized)
        if m is None:
            continue
        intent = _resolve(kind, m, last, steps, scroll)
        if intent is None or is_risky(intent.name):
            return None
        return Intent(intent.name, intent.args, normalized, "followup")
    return None


def _resolve(
    kind: str,
    m: re.Match[str],
    last: LastAction,
    steps: Mapping[str, int],
    scroll: Mapping[str, int],
) -> Intent | None:
    prev = last.intent
    target = _target(prev.name)
    sizes = scroll if target == "scroll" else steps
    if kind == "repeat":
        return Intent(prev.name, dict(prev.args))
    if kind == "undo":
        return _undo(last)
    if target is None:
        return None
    groups = m.groupdict()
    if kind in ("more", "less"):
        sign = _sign(prev)
        if sign is None:
            return None
        if kind == "more":  # same amount as last time, unless a size is said
            same = abs(int(prev.args.get("delta", 0))) or int(prev.args.get("notches", 0))
            default = same or sizes["normal"]
            return _change(target, sign, _size(groups.get("size"), sizes, default))
        return _change(target, -sign, _size(groups.get("size"), sizes, sizes["small"]))
    if kind == "dir":
        sign = 1 if groups["dir"] in ("up", "higher") else -1
        return _change(target, sign, _size(groups.get("size"), sizes, sizes["normal"]))
    if kind == "level":
        level = int(groups["level"])
        if target == "scroll" or level > 100:
            return None
        return Intent(f"{target}.set", {"level": level})
    if kind == "transfer":
        new = _TARGETS[groups.get("target") or groups["target_b"]]
        if target == "scroll" or new == target:
            return None
        verb = prev.name.split(".", 1)[1]
        if verb not in ("set", "step", "get"):
            return None
        return Intent(f"{new}.{verb}", dict(prev.args))
    return None
