"""Gemini assistant: tool loop, safety exclusions, search, images, memory. No network."""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from grace.actions import create_fake_registry
from grace.intent.assistant import (
    SYSTEM_INSTRUCTION,
    GeminiAssistant,
    registry_tools,
    to_gemini_schema,
)
from grace.intent.providers import Interaction, ProviderUnavailableError, QuotaExceededError


def text(t: str, id_: str = "i") -> Interaction:
    return Interaction(
        id_, "completed", [{"type": "model_output", "content": [{"type": "text", "text": t}]}]
    )


def calls(*fns: tuple[str, dict], id_: str = "c") -> Interaction:
    steps = [{"type": "thought", "signature": "s"}]
    steps += [
        {"type": "function_call", "id": f"call{i}", "name": n, "arguments": a}
        for i, (n, a) in enumerate(fns)
    ]
    return Interaction(id_, "requires_action", steps)


class FakeClient:
    """Answers each create() from a script: an Interaction, an exception, or a callable(body)."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.bodies: list[dict] = []
        self.kwargs: list[dict] = []

    def create(self, body, **kwargs) -> Interaction:
        self.bodies.append(body)
        self.kwargs.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, Callable):
            return item(body)
        return item


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def setup(tmp_path: Path):
    registry, b = create_fake_registry(screenshot_dir=tmp_path)
    return registry, b, tmp_path


def make(registry, client, tmp_path, **kw) -> GeminiAssistant:
    return GeminiAssistant(client, registry, image_dir=tmp_path, now=lambda: 0.0, **kw)


# --------------------------------------------------------------------------- tool schema
def test_risky_and_path_actions_are_never_tools(setup) -> None:
    registry, _, _ = setup
    tools, names = registry_tools(registry)
    actions = set(names.values())
    assert not {"system.shutdown", "system.restart", "system.sleep"} & actions
    assert "screen.show_image" not in actions
    assert {"volume.set", "apps.open", "input.type", "input.hotkey"} <= actions
    assert all("." not in t["name"] for t in tools)


def test_schema_conversion_keeps_only_supported_keywords(setup) -> None:
    registry, _, _ = setup
    allowed = {
        "type",
        "description",
        "enum",
        "properties",
        "required",
        "items",
        "minimum",
        "maximum",
    }

    def check(schema: dict) -> None:
        assert set(schema) <= allowed, schema
        for sub in schema.get("properties", {}).values():
            check(sub)
        if "items" in schema:
            check(schema["items"])

    for tool in registry_tools(registry)[0]:
        if "parameters" in tool:
            assert tool["parameters"]["type"] == "object"
            check(tool["parameters"])
    click = to_gemini_schema(registry.spec("input.click").args_model.model_json_schema())
    assert click["properties"]["button"]["enum"] == ["left", "right", "middle"]


def test_body_is_stable_across_calls(setup) -> None:
    registry, _, tmp = setup
    client = FakeClient(text("a", "i1"), text("b", "i2"))
    a = make(registry, client, tmp)
    a.ask("one")
    a.ask("two")
    first, second = client.bodies
    assert first["system_instruction"] == second["system_instruction"] == SYSTEM_INSTRUCTION
    assert first["tools"] == second["tools"]
    assert {"web_search", "generate_image"} <= set(a.tool_names)


# --------------------------------------------------------------------------- answers + tools
def test_plain_answer_and_memory(setup) -> None:
    registry, _, tmp = setup
    client = FakeClient(text("84.", "i1"), text("28.", "i2"))
    a = make(registry, client, tmp)
    assert a.ask("what is 12 times 7").text == "84."
    assert a.ask("and divided by 3").text == "28."
    assert "previous_interaction_id" not in client.bodies[0]
    assert client.bodies[1]["previous_interaction_id"] == "i1"


def test_memory_resets_after_idle(setup) -> None:
    registry, _, tmp = setup
    clock = Clock()
    client = FakeClient(text("a", "i1"), text("b", "i2"))
    a = GeminiAssistant(client, registry, image_dir=tmp, memory_reset_s=60, clock=clock)
    a.ask("one")
    clock.now += 61
    a.ask("two")
    assert "previous_interaction_id" not in client.bodies[1]


def test_tool_calls_run_through_registry(setup) -> None:
    registry, b, tmp = setup
    client = FakeClient(
        calls(("volume_set", {"level": 30}), ("apps_open", {"name": "notepad"}), id_="c1"),
        text("Done: volume 30 and Notepad.", "i2"),
    )
    reply = make(registry, client, tmp).ask("volume thirty and open notepad")
    assert b.volume.level == 30 and b.launcher.launched == ["notepad"]
    assert reply.actions == ["volume.set", "apps.open"] and reply.text.startswith("Done")
    second = client.bodies[1]
    assert second["previous_interaction_id"] == "c1"
    assert [r["call_id"] for r in second["input"]] == ["call0", "call1"]
    assert second["input"][0]["result"] == [{"type": "text", "text": "Volume set to 30 percent."}]


def test_only_the_message_goes_back_never_action_data(setup) -> None:
    registry, b, tmp = setup
    b.clipboard.text = "my private selection"
    client = FakeClient(calls(("clipboard_copy", {})), text("Copied."))
    make(registry, client, tmp).ask("copy that")
    assert "my private selection" not in str(client.bodies[1])


def test_risky_tool_call_is_refused_even_if_the_model_tries(setup) -> None:
    registry, b, tmp = setup
    client = FakeClient(calls(("system_shutdown", {})), text("I can't do that."))
    make(registry, client, tmp).ask("shut it all down")
    assert b.power.calls == []
    assert client.bodies[1]["input"][0]["is_error"] is True


def test_invalid_tool_args_are_reported_as_errors(setup) -> None:
    registry, b, tmp = setup
    client = FakeClient(calls(("input_hotkey", {"keys": ["ctrl", "nope"]})), text("Sorry."))
    make(registry, client, tmp).ask("press something")
    assert b.input.calls == []
    assert client.bodies[1]["input"][0]["is_error"] is True


def test_tool_loop_is_bounded(setup) -> None:
    registry, _, tmp = setup
    client = FakeClient(*[calls(("volume_get", {})) for _ in range(10)])
    reply = make(registry, client, tmp, max_tool_rounds=3).ask("loop forever")
    assert "too many steps" in reply.text
    assert len(client.bodies) == 4


def test_provider_errors_become_spoken_replies(setup) -> None:
    registry, _, tmp = setup
    a = make(registry, FakeClient(ProviderUnavailableError("I couldn't reach Gemini.")), tmp)
    reply = a.ask("hello")
    assert not reply.ok and reply.text == "I couldn't reach Gemini."


# --------------------------------------------------------------------------- web search
def test_web_search_uses_grounding_and_collects_sources(setup) -> None:
    registry, _, tmp = setup
    found = Interaction(
        "s",
        "completed",
        [
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": "India won.",
                        "annotations": [
                            {"type": "url_citation", "url": "https://n.example", "title": "N"}
                        ],
                    }
                ],
            }
        ],
    )
    notes: list[str] = []
    client = FakeClient(
        calls(("web_search", {"query": "cricket world cup"})), found, text("India.")
    )
    reply = make(registry, client, tmp, notify=notes.append).ask("who won the world cup")
    assert client.bodies[1]["tools"] == [{"type": "google_search"}]
    assert client.bodies[2]["input"][0]["result"][0]["text"] == "India won."
    assert reply.sources == [("N", "https://n.example")]
    assert notes == ["Searching the web."]


def test_web_search_without_billing_is_explained(setup) -> None:
    registry, _, tmp = setup
    client = FakeClient(
        calls(("web_search", {"query": "news"})),
        QuotaExceededError("x", billing_required=True),
        text("I can't search the web right now."),
    )
    reply = make(registry, client, tmp).ask("search the news")
    assert "billing-enabled" in reply.text  # deterministic notice, whatever the model says


# --------------------------------------------------------------------------- images
def test_generate_image_saves_and_shows(setup) -> None:
    registry, b, tmp = setup
    jpeg = b"\xff\xd8\xff fake jpeg"
    made = Interaction(
        "img",
        "completed",
        [
            {
                "type": "model_output",
                "content": [{"type": "image", "data": base64.b64encode(jpeg).decode()}],
            }
        ],
    )
    client = FakeClient(
        calls(("generate_image", {"prompt": "a cat", "aspect_ratio": "16:9"})), made, text("Here.")
    )
    reply = make(registry, client, tmp, image_model="img-model").ask("draw a cat")
    [path] = reply.images
    assert path.read_bytes() == jpeg and path.parent == tmp and path.suffix == ".jpg"
    assert b.launcher.files == [path.resolve()]
    body, kwargs = client.bodies[1], client.kwargs[1]
    assert body["model"] == "img-model"
    assert body["response_format"] == {
        "type": "image",
        "mime_type": "image/jpeg",
        "aspect_ratio": "16:9",
    }
    assert kwargs["headers"]["Api-Revision"]


def test_generate_image_without_billing_is_explained(setup) -> None:
    registry, b, tmp = setup
    client = FakeClient(
        calls(("generate_image", {"prompt": "a cat"})),
        QuotaExceededError("x", billing_required=True),
        text("Sorry, no image."),
    )
    reply = make(registry, client, tmp).ask("draw a cat")
    assert "billing-enabled" in reply.text and reply.images == [] and b.launcher.files == []


def test_disabled_features_are_not_offered(setup) -> None:
    registry, _, tmp = setup
    a = make(registry, FakeClient(), tmp, web_search=False, image_generation=False)
    assert not {"web_search", "generate_image"} & set(a.tool_names)


def test_show_image_only_opens_files_in_grace_folder(setup, tmp_path) -> None:
    registry, b, tmp = setup
    outside = tmp_path.parent / "elsewhere.jpg"
    outside.write_bytes(b"x")
    assert not registry.execute("screen.show_image", {"path": str(outside)}).ok
    (tmp / "notes.txt").write_text("x")
    assert not registry.execute("screen.show_image", {"path": str(tmp / "notes.txt")}).ok
    assert b.launcher.files == []


def test_complete_is_single_turn_without_tools(setup) -> None:
    registry, _, tmp = setup
    client = FakeClient(text("Looks fine."))
    assert make(registry, client, tmp).complete("code", instruction="review") == "Looks fine."
    assert "tools" not in client.bodies[0] and "previous_interaction_id" not in client.bodies[0]
