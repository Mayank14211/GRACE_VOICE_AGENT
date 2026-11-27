"""Router tool executors behind `extras`: search, images, code analysis (never Gemini)."""

from __future__ import annotations

from pathlib import Path

from grace.actions import create_fake_registry
from grace.config import load_settings
from grace.intent.assistant import PRIVATE_EXCLUDE, Extras, build_extras
from grace.intent.images import ImageResult
from grace.intent.search import FakeSearch


class FakeImages:
    name = "fake"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> ImageResult:
        self.prompts.append(prompt)
        return ImageResult(True, "Image saved.", [Path("x.jpg")])


class FakeAnalyst:
    def __init__(self) -> None:
        self.calls: list[tuple[str, frozenset[str]]] = []

    def complete(self, prompt, *, instruction=None, exclude=frozenset()):
        self.calls.append((prompt, exclude))
        return "Looks fine."


def test_search_and_images_delegate_and_notify() -> None:
    notes: list[str] = []
    search, images = FakeSearch("Sunny."), FakeImages()
    extras = Extras(search=search, images=images, notify=notes.append)
    found = extras.web_search("weather in delhi")
    assert (
        found.ok
        and found.text == "Sunny."
        and found.sources == [("Example", "https://example.com")]
    )
    made = extras.generate_image("a cat in sunglasses")
    assert made.ok and made.images == [Path("x.jpg")]
    assert search.queries == ["weather in delhi"] and images.prompts == ["a cat in sunglasses"]
    assert notes == ["Searching the web.", "Creating the image."]
    assert extras.offers_search and extras.offers_images


def test_missing_tools_say_which_key_is_needed() -> None:
    extras = Extras()
    assert not extras.offers_search and not extras.offers_images
    assert "TAVILY_API_KEY" in extras.web_search("x").text
    assert "CLOUDFLARE_ACCOUNT_ID" in extras.generate_image("x").text
    assert "Groq or OpenRouter" in extras.complete("code")


def test_code_analysis_always_excludes_gemini() -> None:
    analyst = FakeAnalyst()
    assert Extras(analyst=analyst).complete("```x```", instruction="review") == "Looks fine."
    assert analyst.calls == [("```x```", PRIVATE_EXCLUDE)] and "gemini" in PRIVATE_EXCLUDE


def test_build_extras_from_keys(tmp_path: Path) -> None:
    registry, _ = create_fake_registry()
    none = build_extras(load_settings(tmp_path / "n.yaml", tmp_path / ".env", environ={}), registry)
    assert not none.offers_search and not none.offers_images
    env = tmp_path / ".env"
    env.write_text(
        "TAVILY_API_KEY=tvly-x\nCLOUDFLARE_ACCOUNT_ID=a\nCLOUDFLARE_API_TOKEN=" + "t" * 20 + "\n"
    )
    full = build_extras(load_settings(tmp_path / "n.yaml", env, environ={}), registry)
    assert full.offers_search and full.offers_images
    off = tmp_path / "off.yaml"
    off.write_text("images: {enabled: false}\nsearch: {enabled: false}\n")
    assert not build_extras(load_settings(off, env, environ={}), registry).offers_images
