"""Gemini assistant: answers what the fixed grammar doesn't understand.

- Free-form questions get short spoken answers, with memory across turns
  (`previous_interaction_id`, forgotten after `memory_reset_s` of silence).
- Tool calling: GRACE's registry actions are exposed as functions, so "turn it down a bit and open
  youtube" works. Risky actions (shutdown, restart, sleep) are NEVER exposed: they keep needing
  the exact phrase plus confirmation (docs/PLAN.md). Only an action's spoken message is sent
  back to Gemini, never its data (e.g. copied clipboard text stays on the machine).
- `web_search` runs a separate Gemini call with Google Search grounding.
- `generate_image` calls the image model, saves to Pictures/GRACE and opens it via the registry.
Web search and image generation need a billing-enabled key; on the free tier they fail with a
clear spoken explanation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grace.actions.registry import ActionRegistry
from grace.actions.screen import DEFAULT_SCREENSHOT_DIR
from grace.intent.providers import (
    IMAGE_API_REVISION,
    GeminiClient,
    Interaction,
    ProviderError,
    QuotaExceededError,
)

log = logging.getLogger(__name__)

# Byte-identical on every call (docs/PLAN.md: keeps provider prompt caches warm).
SYSTEM_INSTRUCTION = (
    "You are GRACE, a voice assistant running on the user's Windows computer. Your replies are "
    "spoken aloud by text-to-speech: answer in one to three short sentences of plain text, with "
    "no markdown, lists, emoji or URLs. When the user asks you to do something on the computer, "
    "call the matching tool; never claim you did something without calling a tool. Use "
    "web_search for current events, prices, weather, news or facts you are not sure about. Use "
    "generate_image when the user asks for a picture, drawing or image. You cannot shut down, "
    "restart or sleep the computer: tell the user to say 'shut down the computer' (or restart / "
    "put the computer to sleep) themselves, which asks for confirmation."
)
SEARCH_INSTRUCTION = (
    "Answer the query from Google Search results in two or three plain sentences with concrete "
    "facts, numbers and dates. No markdown."
)

BILLING_NOTICE = (
    "{feature} needs a Gemini API key from a billing-enabled Google project; "
    "the free tier doesn't include it."
)

# Actions the model may never call, on top of every risky action. The pointer primitives are for
# gestures: the model can't see the screen, and a held button must always be released.
LLM_EXCLUDED_ACTIONS = frozenset(
    {
        "screen.show_image",
        "input.move_to",
        "input.mouse_down",
        "input.mouse_up",
        "input.release_all",
    }
)
_SCHEMA_KEYS = frozenset(
    {"type", "description", "enum", "properties", "required", "items", "minimum", "maximum"}
)
ASPECT_RATIOS = ("1:1", "16:9", "9:16", "4:3", "3:4")

WEB_SEARCH_TOOL = {
    "type": "function",
    "name": "web_search",
    "description": "Search the web (Google) for current or factual information.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for."}},
        "required": ["query"],
    },
}
IMAGE_TOOL = {
    "type": "function",
    "name": "generate_image",
    "description": "Create an image from a text description, save it and show it on screen.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Detailed description of the image."},
            "aspect_ratio": {"type": "string", "enum": list(ASPECT_RATIOS)},
        },
        "required": ["prompt"],
    },
}


def tool_name(action: str) -> str:
    """Registry name -> function name ("volume.set" -> "volume_set")."""
    return action.replace(".", "_")


def to_gemini_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Pydantic JSON schema -> the OpenAPI subset Gemini function declarations accept. Pure."""
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _SCHEMA_KEYS:
            continue  # title, default, additionalProperties, minLength, maxItems...
        if key == "properties":
            out[key] = {name: to_gemini_schema(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = to_gemini_schema(value)
        else:
            out[key] = value
    if "default" in schema and "description" not in out:
        out["description"] = f"Default {schema['default']!r}."
    return out


def registry_tools(registry: ActionRegistry) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Function declarations for every safe action, and function name -> action name."""
    tools, names = [], {}
    for spec in sorted(registry, key=lambda s: s.name):
        if spec.risky or spec.name in LLM_EXCLUDED_ACTIONS:
            continue
        decl: dict[str, Any] = {
            "type": "function",
            "name": tool_name(spec.name),
            "description": spec.description,
        }
        params = to_gemini_schema(spec.args_model.model_json_schema())
        if params.get("properties"):
            decl["parameters"] = params
        tools.append(decl)
        names[decl["name"]] = spec.name
    return tools, names


@dataclass
class AssistantReply:
    text: str
    ok: bool = True
    actions: list[str] = field(default_factory=list)
    images: list[Path] = field(default_factory=list)
    sources: list[tuple[str, str]] = field(default_factory=list)
    # Facts the user must hear even if the model words its answer differently.
    notices: list[str] = field(default_factory=list)


def _result(call: Mapping[str, Any], text: str, *, error: bool = False) -> dict[str, Any]:
    step: dict[str, Any] = {
        "type": "function_result",
        "name": call.get("name", ""),
        "call_id": call.get("id", ""),
        "result": [{"type": "text", "text": text}],
    }
    if error:
        step["is_error"] = True
    return step


class GeminiAssistant:
    def __init__(
        self,
        client: GeminiClient,
        registry: ActionRegistry,
        *,
        model: str = "gemini-3.5-flash-lite",
        image_model: str = "gemini-3.1-flash-image",
        web_search: bool = True,
        image_generation: bool = True,
        thinking_level: str = "low",
        image_timeout_s: float = 90.0,
        memory_reset_s: float = 600.0,
        image_dir: Path = DEFAULT_SCREENSHOT_DIR,
        max_tool_rounds: int = 5,
        notify: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.client = client
        self.registry = registry
        self.model = model
        self.image_model = image_model
        self.web_search_enabled = web_search
        self.image_generation_enabled = image_generation
        self.thinking_level = thinking_level
        self.image_timeout_s = image_timeout_s
        self.memory_reset_s = memory_reset_s
        self.image_dir = image_dir
        self.max_tool_rounds = max_tool_rounds
        self.notify = notify or (lambda _msg: None)
        self._clock = clock
        self._now = now
        self._tools, self._tool_actions = registry_tools(registry)
        if web_search:
            self._tools.append(WEB_SEARCH_TOOL)
        if image_generation:
            self._tools.append(IMAGE_TOOL)
        self._previous_id: str | None = None
        self._last_turn = 0.0

    # -- conversation ---------------------------------------------------------------------------
    def reset(self) -> None:
        self._previous_id = None

    @property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in self._tools]

    def _body(self, user_input: Any, previous_id: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "system_instruction": SYSTEM_INSTRUCTION,
            "tools": self._tools,  # tool declarations are not stored server-side: resend
            "input": user_input,
            "generation_config": {"thinking_level": self.thinking_level},
        }
        if previous_id:
            body["previous_interaction_id"] = previous_id
        return body

    def ask(self, text: str) -> AssistantReply:
        """Answer one utterance, running any tools the model calls. Never raises."""
        if (
            self._previous_id
            and self.memory_reset_s
            and (self._clock() - self._last_turn > self.memory_reset_s)
        ):
            self._previous_id = None
        reply = AssistantReply("")
        try:
            interaction = self.client.create(self._body(text, self._previous_id))
            for _ in range(self.max_tool_rounds):
                if not interaction.function_calls:
                    break
                results = [self._run_tool(call, reply) for call in interaction.function_calls]
                interaction = self.client.create(self._body(results, interaction.id))
            else:
                if interaction.function_calls:
                    reply.text = "I stopped because that needed too many steps."
        except ProviderError as exc:
            reply.ok, reply.text = False, str(exc)
            if not isinstance(exc, QuotaExceededError):
                self._previous_id = None  # don't continue from a broken turn
            return reply
        self._previous_id, self._last_turn = interaction.id, self._clock()
        reply.text = reply.text or interaction.text or ("Done." if reply.actions else "")
        if not reply.text:
            reply.text = "Sorry, I don't have an answer for that."
        if reply.notices and "billing" not in reply.text.lower():
            reply.text = f"{reply.text} {' '.join(dict.fromkeys(reply.notices))}"
        return reply

    def complete(self, prompt: str, *, instruction: str | None = None) -> str:
        """One-shot text answer with no tools and no memory (code analysis, summaries)."""
        body = {
            "model": self.model,
            "input": prompt,
            "generation_config": {"thinking_level": self.thinking_level},
        }
        if instruction:
            body["system_instruction"] = instruction
        try:
            return self.client.create(body).text or "Gemini returned no answer."
        except ProviderError as exc:
            return str(exc)

    # -- tools ----------------------------------------------------------------------------------
    def _run_tool(self, call: Mapping[str, Any], reply: AssistantReply) -> dict[str, Any]:
        name = str(call.get("name", ""))
        args = call.get("arguments") or {}
        if not isinstance(args, Mapping):
            return _result(call, "Arguments must be an object.", error=True)
        log.info("assistant tool call %s", name)  # args not logged (may hold typed text)
        if name == "web_search" and self.web_search_enabled:
            return self._web_search(call, str(args.get("query", "")), reply)
        if name == "generate_image" and self.image_generation_enabled:
            return self._generate_image(call, args, reply)
        action = self._tool_actions.get(name)
        if action is None or self.registry.is_risky(action):
            return _result(call, f"Unknown tool {name}.", error=True)
        result = self.registry.execute(action, dict(args))
        if result.ok:
            reply.actions.append(action)
        return _result(call, result.message, error=not result.ok)

    def _web_search(
        self, call: Mapping[str, Any], query: str, reply: AssistantReply
    ) -> dict[str, Any]:
        if not query.strip():
            return _result(call, "Empty search query.", error=True)
        self.notify("Searching the web.")
        try:
            found: Interaction = self.client.create(
                {
                    "model": self.model,
                    "system_instruction": SEARCH_INSTRUCTION,
                    "tools": [{"type": "google_search"}],
                    "input": query,
                }
            )
        except QuotaExceededError as exc:
            if exc.billing_required:
                reply.notices.append(BILLING_NOTICE.format(feature="Web search"))
            why = (
                "Web search is unavailable: it needs a Gemini key from a billing-enabled project."
                if exc.billing_required
                else "Web search quota is used up for now."
            )
            return _result(
                call,
                why + " Answer from your own knowledge if you can, and say it may be out of date.",
                error=True,
            )
        except ProviderError as exc:
            return _result(call, f"Web search failed: {exc}", error=True)
        reply.sources.extend(s for s in found.citations if s not in reply.sources)
        return _result(call, found.text or "No results.")

    def _generate_image(
        self, call: Mapping[str, Any], args: Mapping[str, Any], reply: AssistantReply
    ) -> dict[str, Any]:
        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return _result(call, "Empty image prompt.", error=True)
        fmt: dict[str, Any] = {"type": "image", "mime_type": "image/jpeg"}
        if args.get("aspect_ratio") in ASPECT_RATIOS:
            fmt["aspect_ratio"] = args["aspect_ratio"]
        self.notify("Creating the image. This can take a little while.")
        try:
            made = self.client.create(
                {
                    "model": self.image_model,
                    "input": [{"type": "text", "text": prompt}],
                    "response_format": fmt,
                },
                timeout_s=self.image_timeout_s,
                headers={"Api-Revision": IMAGE_API_REVISION},
            )
        except QuotaExceededError as exc:
            if exc.billing_required:
                reply.notices.append(BILLING_NOTICE.format(feature="Image generation"))
            why = (
                "Image generation is unavailable: it needs a Gemini key from a billing-enabled "
                "project. Tell the user this."
                if exc.billing_required
                else "Image generation quota is used up for now."
            )
            return _result(call, why, error=True)
        except ProviderError as exc:
            return _result(call, f"Image generation failed: {exc}", error=True)
        if not made.images:
            return _result(call, "The image model returned no image.", error=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._now()))
        try:
            self.image_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for i, data in enumerate(made.images):
                path = self.image_dir / f"image_{stamp}{f'_{i + 1}' if i else ''}.jpg"
                path.write_bytes(data)
                paths.append(path)
        except OSError:
            return _result(call, "Couldn't save the image.", error=True)
        reply.images.extend(paths)
        shown = self.registry.execute("screen.show_image", {"path": str(paths[0])})
        saved = f"Image saved as {paths[0].name} in the {self.image_dir.name} pictures folder."
        return _result(call, saved + (" It is open on screen." if shown.ok else ""))
