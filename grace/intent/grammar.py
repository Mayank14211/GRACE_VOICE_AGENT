"""Deterministic command grammar: ``parse(text) -> Intent | None``.

Every pattern must match the WHOLE normalised utterance (``re.fullmatch``). There is no
keyword-contains matching anywhere: "I'm shutting the door", "close call" or "turn off the light"
never reach an action. Anything that doesn't parse returns None and goes to the LLM router (S07).

Risky intents (shutdown, restart, sleep) only match the exact phrase family
"<verb> [the|my|this] <computer|pc|laptop|system|machine>", and still need confirmation
(grace/actions/confirmation.py) before they run.

Vocabulary (synonyms, fixed phrases, shortcuts, key names) lives in synonyms.yaml; the pattern
rules with arguments live in RULES below. Template syntax is described at the top of the YAML.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from grace.intent.types import Intent

SYNONYMS_PATH = Path(__file__).with_name("synonyms.yaml")


class GrammarError(ValueError):
    """The grammar table is malformed (bad YAML, unknown group in a template...)."""


# --------------------------------------------------------------------------- normalisation
_UNITS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS = {
    w: 10 * i
    for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)
}
_CONTRACTIONS = {
    "what's": "what is",
    "whats": "what is",
    "it's": "it is",
    "that's": "that is",
    "i'm": "i am",
    "don't": "do not",
    "dont": "do not",
    "can't": "cannot",
    "won't": "will not",
    "let's": "let us",
    "isn't": "is not",
}


def words_to_digits(words: Sequence[str]) -> list[str]:
    """["forty", "five", "percent"] -> ["45", "percent"]; "a hundred" -> "100". Pure."""
    out: list[str] = []
    i = 0
    while i < len(words):
        w, nxt = words[i], words[i + 1] if i + 1 < len(words) else None
        if w in ("a", "one") and nxt == "hundred":
            out.append("100")
            i += 2
        elif w == "hundred":
            out.append("100")
            i += 1
        elif w in _TENS:
            value = _TENS[w]
            i += 1
            if i < len(words) and 1 <= _UNITS.get(words[i], 0) <= 9:
                value += _UNITS[words[i]]
                i += 1
            out.append(str(value))
        elif w in _UNITS:
            out.append(str(_UNITS[w]))
            i += 1
        else:
            out.append(w)
            i += 1
    return out


def _strip_fillers(text: str, leading: Sequence[str], trailing: Sequence[str]) -> str:
    changed = True
    while changed and text:
        changed = False
        for f in leading:
            if text == f or text.startswith(f + " "):
                text, changed = text[len(f) :].lstrip(), True
        for f in trailing:
            if text == f or text.endswith(" " + f):
                text, changed = text[: -len(f)].rstrip(), True
    return text


def normalize(text: str, leading: Sequence[str] = (), trailing: Sequence[str] = ()) -> str:
    """Lower case, punctuation stripped, numbers as digits, fillers removed. Pure."""
    t = text.replace("’", "'").replace("‘", "'").lower()
    t = " ".join(_CONTRACTIONS.get(w, w) for w in t.split())
    t = t.replace("'", "").replace("%", " percent ").replace("+", " plus ")
    t = re.sub(r"(?<![a-z0-9])\.|\.(?![a-z0-9])", " ", t)  # keep dots inside "github.com"
    t = re.sub(r"[^a-z0-9. ]+", " ", t)
    t = " ".join(words_to_digits(t.split()))
    longest_first = sorted(leading, key=len, reverse=True), sorted(trailing, key=len, reverse=True)
    return _strip_fillers(t, *longest_first)


# --------------------------------------------------------------------------- templates
_GROUP_TOKEN = re.compile(r"\{([a-z_]+)(\?)?\}")
_CAPTURE_TOKEN = re.compile(r"<([a-z_]+):([a-z_]+)>(\?)?")
_LITERAL_TOKEN = re.compile(r"[a-z0-9.]+")
_MACRO = re.compile(r"\{([a-z_]+)\}")

# Built-in groups. `{name}` inside them expands a YAML group. They contain named captures, so a
# template may use each of them at most once.
BUILTINS: Mapping[str, str] = {
    "num": r"\d{1,4}",
    "text": r".+",
    "level": r"(?:(?P<level>\d{1,4})(?: {percent})?|(?P<lmax>{maximum})|(?P<lmin>{minimum})"
    r"|(?P<lhalf>{half}))",
    "amount": r"(?:by )?(?:(?P<n>\d{1,4})(?: {percent})?|(?P<small>{small})|(?P<large>{large}))",
    "scroll_amount": r"(?:(?P<n>\d{1,4})(?: times| notches| clicks)?|(?P<small>{small})"
    r"|(?P<large>{large}))",
}


def _alternation(phrases: Sequence[str]) -> str:
    return "(?:" + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + ")"


def compile_template(template: str, groups: Mapping[str, Sequence[str]]) -> re.Pattern[str]:
    """Template -> regex matched with fullmatch against " " + normalised text."""

    def expand(name: str) -> str:
        if name in BUILTINS:
            return _MACRO.sub(lambda m: expand(m.group(1)), BUILTINS[name])
        if name in groups:
            return _alternation(groups[name])
        raise GrammarError(f"unknown group {{{name}}} in template {template!r}")

    parts = []
    for token in template.split():
        if m := _GROUP_TOKEN.fullmatch(token):
            body = expand(m.group(1))
            parts.append(f"(?: {body})?" if m.group(2) else f" {body}")
        elif m := _CAPTURE_TOKEN.fullmatch(token):
            body = f"(?P<{m.group(1)}>{expand(m.group(2))})"
            parts.append(f"(?: {body})?" if m.group(3) else f" {body}")
        elif _LITERAL_TOKEN.fullmatch(token):
            parts.append(" " + re.escape(token))
        else:
            raise GrammarError(f"bad token {token!r} in template {template!r}")
    if not parts:
        raise GrammarError("empty template")
    try:
        return re.compile("".join(parts))
    except re.error as exc:
        raise GrammarError(f"template {template!r}: {exc}") from exc


# --------------------------------------------------------------------------- argument builders
Builder = Callable[[re.Match[str], "Grammar"], dict[str, Any] | None]


def _no_args(m: re.Match[str], g: Grammar) -> dict[str, Any]:
    return {}


def _const(**args: Any) -> Builder:
    return lambda m, g: dict(args)


def _level(m: re.Match[str], g: Grammar) -> dict[str, Any]:
    d = m.groupdict()
    if d.get("lmax"):
        return {"level": 100}
    if d.get("lmin"):
        return {"level": 0}
    if d.get("lhalf"):
        return {"level": 50}
    return {"level": int(d["level"])}


def _size(m: re.Match[str], sizes: Mapping[str, int]) -> int:
    d = m.groupdict()
    if d.get("n"):
        return int(d["n"])
    if d.get("small"):
        return sizes["small"]
    if d.get("large"):
        return sizes["large"]
    return sizes["normal"]


def _step(sign: int | None = None) -> Builder:
    def build(m: re.Match[str], g: Grammar) -> dict[str, Any]:
        s = sign if sign is not None else (1 if m.group("dir") == "up" else -1)
        return {"delta": s * _size(m, g.steps)}

    return build


def _click(m: re.Match[str], g: Grammar) -> dict[str, Any]:
    d = m.groupdict()
    count = {"double": 2, "triple": 3}.get(d.get("count") or "", 1)
    return {"button": d.get("button") or "left", "count": count}


def _scroll(m: re.Match[str], g: Grammar) -> dict[str, Any]:
    return {"direction": m.groupdict().get("dir") or "down", "notches": _size(m, g.scroll)}


def _hotkey(m: re.Match[str], g: Grammar) -> dict[str, Any] | None:
    keys = g.parse_keys(m.group("keys"))
    return {"keys": keys} if keys else None


def _open(m: re.Match[str], g: Grammar) -> dict[str, Any] | None:
    # Only claim names we can open; "open the door" / "go to sleep" fall through to the router.
    name = m.group("name")
    return {"name": name} if g.can_open(name) else None


@dataclass(frozen=True)
class Rule:
    intent: str
    template: str
    build: Builder = _no_args


# Order matters: the first rule whose template matches AND whose builder accepts wins.
# Fixed phrases and shortcuts from the YAML are checked before these.
RULES: tuple[Rule, ...] = (
    # Risky: exact phrases only (see module docstring). Confirmation is enforced elsewhere.
    Rule("system.shutdown", "{shut_down} {det?} {machine}"),
    Rule("system.restart", "{restart} {det?} {machine}"),
    Rule("system.sleep", "put {det?} {machine} to sleep"),
    Rule("system.sleep", "{sleep} {det?} {machine}"),
    # Volume
    Rule("volume.get", "{query} {det?} {volume}"),
    Rule("volume.get", "how loud is it"),
    Rule("volume.mute", "{mute} {det?} {volume?}", _const(mute=True)),
    Rule("volume.mute", "{unmute} {det?} {volume?}", _const(mute=False)),
    Rule("volume.mute", "turn off {det?} {volume}", _const(mute=True)),
    Rule("volume.mute", "turn on {det?} {volume}", _const(mute=False)),
    Rule("volume.set", "{set?} {det?} {volume} {to?} {level}", _level),
    Rule("volume.set", "{turn} {det?} {volume} {updown} {to} {level}", _level),
    Rule("volume.set", "{turn} {updown} {det?} {volume} {to} {level}", _level),
    Rule("volume.step", "{volume} <dir:updown> {amount?}", _step()),
    Rule("volume.step", "{turn} {det?} {volume} <dir:updown> {amount?}", _step()),
    Rule("volume.step", "{turn} <dir:updown> {det?} {volume} {amount?}", _step()),
    Rule("volume.step", "{increase} {det?} {volume} {amount?}", _step(+1)),
    Rule("volume.step", "{decrease} {det?} {volume} {amount?}", _step(-1)),
    Rule("volume.step", "<small:small>? {louder}", _step(+1)),
    Rule("volume.step", "<small:small>? {quieter}", _step(-1)),
    # Brightness
    Rule("brightness.get", "{query} {det?} {brightness}"),
    Rule("brightness.get", "how bright is {det?} {screen}"),
    Rule("brightness.set", "{set?} {det?} {brightness} {to?} {level}", _level),
    Rule("brightness.set", "{turn} {det?} {brightness} {updown} {to} {level}", _level),
    Rule("brightness.set", "{turn} {updown} {det?} {brightness} {to} {level}", _level),
    Rule("brightness.step", "{brightness} <dir:updown> {amount?}", _step()),
    Rule("brightness.step", "{turn} {det?} {brightness} <dir:updown> {amount?}", _step()),
    Rule("brightness.step", "{turn} <dir:updown> {det?} {brightness} {amount?}", _step()),
    Rule("brightness.step", "{increase} {det?} {brightness} {amount?}", _step(+1)),
    Rule("brightness.step", "{decrease} {det?} {brightness} {amount?}", _step(-1)),
    Rule("brightness.step", "<small:small>? {brighter}", _step(+1)),
    Rule("brightness.step", "<small:small>? {dimmer}", _step(-1)),
    Rule("brightness.step", "make {det?} {screen} <small:small>? {brighter}", _step(+1)),
    Rule("brightness.step", "make {det?} {screen} <small:small>? {dimmer}", _step(-1)),
    Rule("brightness.step", "{brighten} {det?} {screen} <small:small>?", _step(+1)),
    Rule("brightness.step", "{dim} {det?} {screen} <small:small>?", _step(-1)),
    # Mouse and keyboard
    Rule("input.click", "{left?} {click} {here?}", _click),
    Rule("input.click", "<count:multi> {left?} {click} {here?}", _click),
    Rule("input.click", "<count:multi>? <button:button> {click} {here?}", _click),
    Rule("input.click", "{click} {det?} mouse", _click),
    Rule("input.scroll", "{scroll} <dir:updown>? {scroll_amount?}", _scroll),
    Rule("input.hotkey", "{press} {det?} <keys:text>", _hotkey),
    # Apps (after shortcuts such as "open the start menu" / "open a new tab")
    Rule("apps.open", "{open} <name:text>", _open),
    Rule("apps.open", "go to <name:text>", _open),
)


# --------------------------------------------------------------------------- grammar
_SKIP_KEY_WORDS = frozenset({"plus", "and", "key", "keys", "button"})


def _catalog_can_open(name: str) -> bool:
    from grace.actions.apps import resolve_target  # pure lookup, no OS access

    return resolve_target(name) is not None


class Grammar:
    def __init__(
        self,
        table: Mapping[str, Any],
        rules: Sequence[Rule] = RULES,
        can_open: Callable[[str], bool] = _catalog_can_open,
    ) -> None:
        self.can_open = can_open
        try:
            fillers = table.get("fillers", {})
            self.leading: list[str] = list(fillers.get("leading", []))
            self.trailing: list[str] = list(fillers.get("trailing", []))
            groups: dict[str, list[str]] = {k: list(v) for k, v in table["groups"].items()}
            self.steps: dict[str, int] = dict(table["steps"])
            self.scroll: dict[str, int] = dict(table["scroll"])
            key_names: dict[str, list[str]] = dict(table["keys"])
            phrases: dict[str, list[str]] = dict(table.get("phrases", {}))
            shortcuts: list[dict[str, Any]] = list(table.get("shortcuts", []))
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise GrammarError(f"malformed grammar table: {exc!r}") from exc
        everything = [*self.leading, *self.trailing, *(p for v in groups.values() for p in v)]
        everything += [p for v in key_names.values() for p in v]
        everything += [p for v in phrases.values() for p in v]
        everything += [p for s in shortcuts for p in [*s.get("say", []), *s.get("keys", [])]]
        if bad := [p for p in everything if not isinstance(p, str) or not p.strip()]:
            raise GrammarError(f"grammar entries must be non-empty strings (quote yes/no): {bad}")
        for size in ("small", "normal", "large"):
            if size not in self.steps or size not in self.scroll:
                raise GrammarError(f"steps/scroll need a {size!r} entry")

        self.phrases: list[tuple[str, re.Pattern[str]]] = [
            (intent, compile_template(p, groups)) for intent, ps in phrases.items() for p in ps
        ]
        self.shortcuts: list[tuple[tuple[str, ...], re.Pattern[str]]] = [
            (tuple(s["keys"]), compile_template(p, groups)) for s in shortcuts for p in s["say"]
        ]
        self.rules: list[tuple[Rule, re.Pattern[str]]] = [
            (r, compile_template(r.template, groups)) for r in rules
        ]
        self._key_phrases: list[tuple[tuple[str, ...], str]] = sorted(
            ((tuple(spoken.split()), canon) for canon, ps in key_names.items() for spoken in ps),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )
        lead = "|".join(re.escape(f) for f in sorted(self.leading, key=len, reverse=True))
        self._type_re = re.compile(
            rf"(?:(?:{lead})[\s,]+)*type[\s,:]+(?P<text>\S.*)"
            if lead
            else r"type[\s,:]+(?P<text>\S.*)",
            re.IGNORECASE | re.DOTALL,
        )

    @classmethod
    def from_yaml(cls, path: Path = SYNONYMS_PATH) -> Grammar:
        try:
            table = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise GrammarError(f"can't load {path.name}: {exc}") from exc
        if not isinstance(table, dict):
            raise GrammarError(f"{path.name} must be a mapping")
        return cls(table)

    def normalize(self, text: str) -> str:
        return normalize(text, self.leading, self.trailing)

    def parse_keys(self, spoken: str) -> list[str] | None:
        """Spoken keys -> key names, e.g. "control plus c" -> ["ctrl", "c"]. None if unknown."""
        tokens = spoken.split()
        keys: list[str] = []
        i = 0
        while i < len(tokens):
            for phrase, canon in self._key_phrases:
                if tuple(tokens[i : i + len(phrase)]) == phrase:
                    keys.append(canon)
                    i += len(phrase)
                    break
            else:
                tok, nxt = tokens[i], tokens[i + 1] if i + 1 < len(tokens) else ""
                if tok in _SKIP_KEY_WORDS:
                    i += 1
                elif (m := re.fullmatch(r"f(\d{1,2})", tok)) and 1 <= int(m.group(1)) <= 12:
                    keys.append(tok)
                    i += 1
                elif tok == "f" and nxt.isdigit() and 1 <= int(nxt) <= 12:
                    keys.append(f"f{int(nxt)}")
                    i += 2
                elif re.fullmatch(r"[a-z0-9]", tok):
                    keys.append(tok)
                    i += 1
                else:
                    return None
        if not 1 <= len(keys) <= 4 or len(set(keys)) != len(keys):
            return None
        return keys

    def parse(self, text: str) -> Intent | None:
        if not text or not text.strip():
            return None
        # "type ..." keeps the user's original casing and punctuation.
        if m := self._type_re.fullmatch(text.strip()):
            return Intent("input.type", {"text": m.group("text").strip()}, text=text.strip())
        norm = self.normalize(text)
        if not norm:
            return None
        subject = " " + norm
        for intent, pattern in self.phrases:
            if pattern.fullmatch(subject):
                return Intent(intent, {}, text=norm)
        for keys, pattern in self.shortcuts:
            if pattern.fullmatch(subject):
                return Intent("input.hotkey", {"keys": list(keys)}, text=norm)
        for rule, pattern in self.rules:
            if (m := pattern.fullmatch(subject)) and (args := rule.build(m, self)) is not None:
                return Intent(rule.intent, args, text=norm)
        return None


@lru_cache(maxsize=1)
def default_grammar() -> Grammar:
    return Grammar.from_yaml(SYNONYMS_PATH)


def parse(text: str) -> Intent | None:
    """Parse one utterance with the default vocabulary (synonyms.yaml)."""
    return default_grammar().parse(text)
