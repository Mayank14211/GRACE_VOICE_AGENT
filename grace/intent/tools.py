"""LLM tool schema, generated from the action registry, and tool call -> Intent validation.

The router offers 8-10 grouped tools instead of one per action (models choose far better from
a short list, and a short list costs fewer tokens of free quota). Each group is one
OpenAI-style function whose ``action`` parameter picks the sub-action; the other parameters
are the union of the sub-actions' pydantic args models, so the schema always matches the
registry:

    system_control  volume / brightness
    mouse           click, scroll
    keyboard_text   type, key presses, clipboard, start dictation
    apps_windows    open apps / sites / links, close window / tab
    read_screen     read the screen or selection aloud, analyse / debug code, screenshot
    notifications   any registry action named notifications.* (S09)
    web_search      our search tool (Tavily), offered when TAVILY_API_KEY is set
    generate_image  Cloudflare FLUX.1 schnell, offered when the Cloudflare keys are set
    answer_user     a short spoken answer

Groups whose actions are not registered are left out. Pointer primitives (move_to, mouse
down / up, release_all) and screen.show_image are never offered: the model can't see the
screen, and a held button must always be released. Risky actions (shutdown, restart, sleep)
are NEVER offered: they need the exact phrase plus confirmation (docs/PLAN.md section 3).

Every tool except answer_user also takes ``more``: true when the user asked for another step
after this one. Groq's gpt-oss-20b can't make parallel tool calls, so the router asks again for
the next step instead of losing it ("set the sound to 30 and then open notepad").

``ToolSet.tools`` is built once and serialised the same way every call, so the system prompt +
tool list stay byte-identical (provider prompt caches stay warm). Pure: no network, no OS.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field, ValidationError

from grace.actions.registry import ActionArgs, ActionRegistry, NoArgs
from grace.intent.types import Intent

# Actions the model may never call (see module doc).
EXCLUDED_ACTIONS = frozenset(
    {
        "screen.show_image",
        "input.move_to",
        "input.mouse_down",
        "input.mouse_up",
        "input.release_all",
    }
)
# JSON-schema keywords every provider accepts (Gemini's OpenAI endpoint takes an OpenAPI subset).
_SCHEMA_KEYS = frozenset(
    {"type", "description", "enum", "properties", "required", "items", "minimum", "maximum"}
)
MAX_ANSWER_CHARS = 600
MORE = "more"
MORE_PARAM = {
    "type": "boolean",
    "description": "true only if the user asked for another action after this one.",
}


class AnswerArgs(ActionArgs):
    text: str = Field(
        min_length=1,
        max_length=MAX_ANSWER_CHARS,
        description="The spoken reply: at most two short sentences, plain text.",
    )


class WebSearchArgs(ActionArgs):
    query: str = Field(min_length=1, max_length=300, description="What to search for.")


class ImageArgs(ActionArgs):
    prompt: str = Field(
        min_length=1, max_length=2000, description="Detailed description of the image."
    )


@dataclass(frozen=True)
class AppAction:
    """An intent the app handles itself (not in the registry), offered as a tool sub-action."""

    intent: str
    description: str
    args_model: type[ActionArgs] = NoArgs


# Read-screen intents are handled by the voice agent (OCR stays on the machine; only code the
# user asked to analyse goes to Gemini).
READ_SCREEN_APP_ACTIONS: dict[str, AppAction] = {
    "read_screen": AppAction("app.read_screen", "Read the text on the screen aloud."),
    "read_selection": AppAction("app.read_selection", "Read the selected text aloud."),
    "analyze_code": AppAction("app.analyze_code", "Explain / review the selected code."),
    "analyze_screen": AppAction("app.analyze_screen", "Explain / review the code on screen."),
    "debug_code": AppAction("app.debug_code", "Find the bug in the selected code."),
}
DICTATION_APP_ACTIONS: dict[str, AppAction] = {
    "start_dictation": AppAction(
        "app.dictation_start", "Start dictation mode: everything said next is typed."
    ),
}
ANSWER = AppAction("app.answer", "Say a short answer to the user.", AnswerArgs)
WEB_SEARCH = AppAction(
    "app.web_search", "Search the web for current or factual information.", WebSearchArgs
)
GENERATE_IMAGE = AppAction(
    "app.generate_image", "Create an image from a description and show it.", ImageArgs
)


@dataclass(frozen=True)
class GroupDef:
    name: str
    description: str
    actions: tuple[str, ...] = ()  # registry names (in this order)
    prefixes: tuple[str, ...] = ()  # plus every registry action starting with one of these
    app_actions: Mapping[str, AppAction] = field(default_factory=dict)


GROUPS: tuple[GroupDef, ...] = (
    GroupDef(
        "system_control",
        "Volume and screen brightness (levels are percent, deltas are points).",
        (
            "volume.set",
            "volume.step",
            "volume.mute",
            "volume.get",
            "brightness.set",
            "brightness.step",
            "brightness.get",
            "system.shutdown",
            "system.restart",
            "system.sleep",
        ),
    ),
    GroupDef(
        "mouse", "Click or scroll where the mouse pointer is.", ("input.click", "input.scroll")
    ),
    GroupDef(
        "keyboard_text",
        "Type text, press keys or shortcuts, use the clipboard, start dictation.",
        ("input.type", "input.hotkey", "clipboard.copy", "clipboard.paste", "clipboard.set"),
        app_actions=DICTATION_APP_ACTIONS,
    ),
    GroupDef(
        "apps_windows",
        "Open an app, website or link; close the active window or tab.",
        ("apps.open", "apps.open_url", "apps.close_window", "apps.close_tab"),
    ),
    GroupDef(
        "read_screen",
        "Read the screen or the selection aloud, explain or debug code, take a screenshot.",
        ("screen.screenshot",),
        app_actions=READ_SCREEN_APP_ACTIONS,
    ),
    GroupDef(
        "notifications", "Read or act on Windows notifications.", prefixes=("notifications.",)
    ),
)
# Single-action tools: no `action` parameter.
WEB_SEARCH_TOOL = "web_search"
IMAGE_TOOL = "generate_image"
ANSWER_TOOL = "answer_user"


class ToolCallError(ValueError):
    """A tool call that can't become an Intent. The message is fed back to the model once."""


def sub_name(action: str) -> str:
    """Registry name -> sub-action name ("volume.set" -> "volume_set")."""
    return action.replace(".", "_")


def clean_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Pydantic JSON schema -> the subset every provider accepts. Pure."""
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _SCHEMA_KEYS:
            continue  # title, default, additionalProperties, minLength, maxItems...
        if key == "properties":
            out[key] = {name: clean_schema(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = clean_schema(value)
        else:
            out[key] = value
    if "default" in schema and "description" not in out:
        out["description"] = f"Default {schema['default']!r}."
    return out


@dataclass(frozen=True)
class SubAction:
    tool: str
    name: str  # value of the `action` parameter ("" for single-action tools)
    intent: str  # registry action or app intent
    args_model: type[ActionArgs]
    description: str


@dataclass
class ToolSet:
    """The tools offered to the model, and the reverse mapping used to validate calls."""

    tools: list[dict[str, Any]]
    subs: dict[str, dict[str, SubAction]]  # tool -> action value -> sub-action

    @property
    def names(self) -> list[str]:
        return [t["function"]["name"] for t in self.tools]

    def intents(self) -> set[str]:
        return {s.intent for subs in self.subs.values() for s in subs.values()}

    def fingerprint(self) -> str:
        """Stable serialisation (what the byte-identical prompt cache depends on)."""
        return json.dumps(self.tools, sort_keys=True, separators=(",", ":"))

    # ------------------------------------------------------------------ calls -> intents
    def to_intent(self, tool: str, arguments: str | Mapping[str, Any] | None, text: str) -> Intent:
        """Validate one tool call. Raises ToolCallError with a short reason for the model."""
        return self.parse_call(tool, arguments, text)[0]

    def parse_call(
        self, tool: str, arguments: str | Mapping[str, Any] | None, text: str
    ) -> tuple[Intent, bool]:
        """(Intent, whether the model says another step follows). Raises ToolCallError."""
        subs = self.subs.get(tool)
        if subs is None:
            raise ToolCallError(f"unknown tool {tool!r}; use one of: {', '.join(self.names)}")
        args = _parse_arguments(arguments)
        more = args.pop(MORE, False) is True
        if "" in subs:
            sub = subs[""]
        else:
            action = args.pop("action", None)
            if not isinstance(action, str) or action not in subs:
                raise ToolCallError(
                    f"{tool}: 'action' must be one of {', '.join(subs)} (got {action!r})"
                )
            sub = subs[action]
        fields = sub.args_model.model_fields
        # Models often fill every union parameter (with null): keep only this action's own.
        own = {k: v for k, v in args.items() if k in fields and v is not None}
        try:
            parsed = sub.args_model.model_validate(own)
        except ValidationError as exc:
            what = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                for e in exc.errors()
            )
            label = f"{tool} {sub.name}" if sub.name else tool
            raise ToolCallError(f"{label}: invalid {what}") from None
        return Intent(sub.intent, parsed.model_dump(), text, "router"), more

    def describe(self, intent: Intent) -> str:
        """An intent in tool terms, e.g. system_control(volume_set, level=40)."""
        for tool, subs in self.subs.items():
            for sub in subs.values():
                if sub.intent == intent.name:
                    args = ", ".join(f"{k}={v!r}" for k, v in intent.args.items())
                    head = sub.name if sub.name else ""
                    inner = ", ".join(p for p in (head, args) if p)
                    return f"{tool}({inner})"
        return intent.name


def _parse_arguments(arguments: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, Mapping):
        return dict(arguments)
    try:
        value = json.loads(arguments)
    except (TypeError, ValueError):
        raise ToolCallError("arguments are not valid JSON") from None
    if not isinstance(value, dict):
        raise ToolCallError("arguments must be a JSON object")
    return value


def _merge_property(
    props: dict[str, Any], users: dict[str, list[str]], name: str, schema: dict[str, Any], who: str
) -> None:
    if name in props:
        a = {k: v for k, v in props[name].items() if k != "description"}
        b = {k: v for k, v in schema.items() if k != "description"}
        if a != b:
            raise ValueError(f"parameter {name!r} has two different schemas in one tool ({who})")
    else:
        props[name] = dict(schema)
    users.setdefault(name, []).append(who)


def _function(name: str, description: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": params},
    }


def _group_tool(group: str, description: str, subs: list[SubAction]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    users: dict[str, list[str]] = {}
    for sub in subs:
        schema = clean_schema(sub.args_model.model_json_schema())
        for pname, pschema in schema.get("properties", {}).items():
            _merge_property(props, users, pname, pschema, sub.name)
    for pname, who in users.items():
        base = props[pname].get("description", "").rstrip(".")
        note = f"For {', '.join(who)}."
        props[pname]["description"] = f"{base}. {note}" if base else note
    lines = "; ".join(f"{s.name}: {s.description.rstrip('.')}" for s in subs)
    action = {
        "type": "string",
        "enum": [s.name for s in subs],
        "description": f"Which action. {lines}.",
    }
    properties = {"action": action, **props, MORE: dict(MORE_PARAM)}
    params = {"type": "object", "properties": properties, "required": ["action"]}
    return _function(group, description, params)


def _single_tool(name: str, app: AppAction) -> tuple[dict[str, Any], SubAction]:
    schema = clean_schema(app.args_model.model_json_schema())
    properties = dict(schema.get("properties", {}))
    if app is not ANSWER:
        properties[MORE] = dict(MORE_PARAM)
    params = {"type": "object", "properties": properties}
    if schema.get("required"):
        params["required"] = schema["required"]
    sub = SubAction(name, "", app.intent, app.args_model, app.description)
    return _function(name, app.description, params), sub


def build_toolset(
    registry: ActionRegistry,
    *,
    search: bool = False,
    images: bool = False,
    app_intents: Iterable[str] | None = None,
    groups: tuple[GroupDef, ...] = GROUPS,
) -> ToolSet:
    """Generate the grouped tools from ``registry``.

    ``search`` / ``images`` add web_search / generate_image (their executors are configured).
    ``app_intents`` limits which app-handled sub-actions are offered (None = all of them).
    Risky actions are always left out.
    """
    allowed_app = None if app_intents is None else set(app_intents)
    tools: list[dict[str, Any]] = []
    subs: dict[str, dict[str, SubAction]] = {}
    names = registry.names()
    for g in groups:
        wanted = list(g.actions) + [
            n for n in names if g.prefixes and n.startswith(g.prefixes) and n not in g.actions
        ]
        group_subs: list[SubAction] = []
        for action in wanted:
            if action not in registry or action in EXCLUDED_ACTIONS:
                continue
            spec = registry.spec(action)
            if spec.risky:
                continue
            group_subs.append(
                SubAction(g.name, sub_name(action), action, spec.args_model, spec.description)
            )
        for sname, app in g.app_actions.items():
            if allowed_app is None or app.intent in allowed_app:
                group_subs.append(
                    SubAction(g.name, sname, app.intent, app.args_model, app.description)
                )
        if not group_subs:
            continue
        tools.append(_group_tool(g.name, g.description, group_subs))
        subs[g.name] = {s.name: s for s in group_subs}
    singles = [(WEB_SEARCH_TOOL, WEB_SEARCH)] if search else []
    singles += [(IMAGE_TOOL, GENERATE_IMAGE)] if images else []
    for name, app in [*singles, (ANSWER_TOOL, ANSWER)]:
        tool, sub = _single_tool(name, app)
        tools.append(tool)
        subs[name] = {"": sub}
    return ToolSet(tools, subs)
