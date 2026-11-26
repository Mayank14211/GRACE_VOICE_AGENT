"""Gemini extras: web search, image generation, code analysis. No network.

Conversation and tool calling moved to the router (tests/test_router.py).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from grace.actions import create_fake_registry
from grace.config import load_settings
from grace.intent.assistant import GeminiExtras, build_extras
from grace.intent.providers import Interaction, ProviderUnavailableError, QuotaExceededError

KEY = "AI" + "za" + "FAKE-KEY-FOR-TESTS-ONLY-0123456789"  # assembled: not flagged as a secret


def text(t: str, id_: str = "i") -> Interaction:
    return Interaction(
        id_, "completed", [{"type": "model_output", "content": [{"type": "text", "text": t}]}]
    )


class FakeClient:
    """Answers each create() from a script: an Interaction or an exception."""

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
        return item


@pytest.fixture
def setup(tmp_path: Path):
    registry, b = create_fake_registry(screenshot_dir=tmp_path)
    return registry, b, tmp_path


def make(registry, client, tmp_path, **kw) -> GeminiExtras:
    return GeminiExtras(client, registry, image_dir=tmp_path, now=lambda: 0.0, **kw)


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
    client = FakeClient(found)
    result = make(registry, client, tmp, notify=notes.append).web_search("cricket world cup")
    assert client.bodies[0]["tools"] == [{"type": "google_search"}]
    assert client.bodies[0]["input"] == "cricket world cup"
    assert result.ok and result.text == "India won."
    assert result.sources == [("N", "https://n.example")]
    assert notes == ["Searching the web."]


def test_web_search_without_billing_is_explained(setup) -> None:
    registry, _, tmp = setup
    client = FakeClient(QuotaExceededError("x", billing_required=True))
    result = make(registry, client, tmp).web_search("news")
    assert not result.ok and "billing-enabled" in result.text


def test_web_search_errors_and_empty_query(setup) -> None:
    registry, _, tmp = setup
    a = make(registry, FakeClient(ProviderUnavailableError("I couldn't reach Gemini.")), tmp)
    assert a.web_search("weather").text == "Web search failed: I couldn't reach Gemini."
    assert not a.web_search("   ").ok
    off = make(registry, FakeClient(), tmp, web_search=False)
    assert not off.web_search("x").ok and off.client.bodies == []


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
    client = FakeClient(made)
    result = make(registry, client, tmp, image_model="img-model").generate_image("a cat", "16:9")
    [path] = result.images
    assert path.read_bytes() == jpeg and path.parent == tmp and path.suffix == ".jpg"
    assert b.launcher.files == [path.resolve()]
    body, kwargs = client.bodies[0], client.kwargs[0]
    assert body["model"] == "img-model"
    assert body["response_format"] == {
        "type": "image",
        "mime_type": "image/jpeg",
        "aspect_ratio": "16:9",
    }
    assert kwargs["headers"]["Api-Revision"]
    assert "open on screen" in result.text


def test_generate_image_without_billing_is_explained(setup) -> None:
    registry, b, tmp = setup
    client = FakeClient(QuotaExceededError("x", billing_required=True))
    result = make(registry, client, tmp).generate_image("a cat")
    assert "billing-enabled" in result.text and result.images == [] and b.launcher.files == []


def test_disabled_features(setup) -> None:
    registry, _, tmp = setup
    a = make(registry, FakeClient(), tmp, web_search=False, image_generation=False)
    assert not a.offers_tools
    assert not a.generate_image("a cat").ok


def test_show_image_only_opens_files_in_grace_folder(setup, tmp_path) -> None:
    registry, b, tmp = setup
    outside = tmp_path.parent / "elsewhere.jpg"
    outside.write_bytes(b"x")
    assert not registry.execute("screen.show_image", {"path": str(outside)}).ok
    (tmp / "notes.txt").write_text("x")
    assert not registry.execute("screen.show_image", {"path": str(tmp / "notes.txt")}).ok
    assert b.launcher.files == []


# --------------------------------------------------------------------------- code analysis
def test_complete_is_single_turn_without_tools(setup) -> None:
    registry, _, tmp = setup
    client = FakeClient(text("Looks fine."))
    assert make(registry, client, tmp).complete("code", instruction="review") == "Looks fine."
    assert "tools" not in client.bodies[0] and "previous_interaction_id" not in client.bodies[0]
    assert client.bodies[0]["system_instruction"] == "review"


def test_complete_errors_are_spoken(setup) -> None:
    registry, _, tmp = setup
    client = FakeClient(ProviderUnavailableError("I couldn't reach Gemini."))
    assert make(registry, client, tmp).complete("code") == "I couldn't reach Gemini."


# --------------------------------------------------------------------------- building
def test_build_extras_needs_a_key_and_no_demo_mode(tmp_path: Path) -> None:
    registry, _ = create_fake_registry()
    none = load_settings(tmp_path / "none.yaml", tmp_path / ".env", environ={})
    assert build_extras(none, registry) is None
    env = tmp_path / ".env"
    env.write_text(f"GEMINI_API_KEY={KEY}\n")
    keyed = load_settings(tmp_path / "none.yaml", env, environ={})
    extras = build_extras(keyed, registry)
    assert extras is not None and KEY not in repr(extras.client)
    demo = tmp_path / "demo.yaml"
    demo.write_text("demo_mode: true\n")
    assert build_extras(load_settings(demo, env, environ={}), registry) is None  # no cloud
