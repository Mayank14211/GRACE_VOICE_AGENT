"""Gemini extras: web search, image generation and code analysis (needs GEMINI_API_KEY).

Conversation, memory and tool calling moved to the router in S07 (grace/intent/router.py),
which reaches Gemini through its OpenAI-compatible endpoint like every other provider. What
stays here needs Gemini's Interactions API, which the OpenAI endpoint doesn't offer:

- ``web_search``: a Gemini call with Google Search grounding; the answer is spoken (two
  sentences) and sources are printed. Router tool ``web_search``.
- ``generate_image``: the image model; saved to Pictures/GRACE and opened through the registry
  (``screen.show_image``, which only opens files in GRACE's own folder). Router tool
  ``generate_image``.
- ``complete``: one-shot text with no tools and no memory (analyse / debug selected code).

Web search and image generation need a key from a billing-enabled project; on the free tier
they fail with a clear spoken explanation (HTTP 429 "limit: 0").
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grace.actions.screen import DEFAULT_SCREENSHOT_DIR
from grace.intent.providers import (
    IMAGE_API_REVISION,
    GeminiClient,
    ProviderError,
    QuotaExceededError,
)
from grace.intent.tools import ASPECT_RATIOS

log = logging.getLogger(__name__)

SEARCH_INSTRUCTION = (
    "Answer the query from Google Search results in two short plain sentences with concrete "
    "facts, numbers and dates. No markdown."
)
BILLING_NOTICE = (
    "{feature} needs a Gemini API key from a billing-enabled Google project; "
    "the free tier doesn't include it."
)


@dataclass
class ExtraResult:
    text: str  # what to say
    ok: bool = True
    images: list[Path] = field(default_factory=list)
    sources: list[tuple[str, str]] = field(default_factory=list)  # (title, url)


class GeminiExtras:
    def __init__(
        self,
        client: GeminiClient,
        registry: Any,
        *,
        model: str = "gemini-3.5-flash-lite",
        image_model: str = "gemini-3.1-flash-image",
        web_search: bool = True,
        image_generation: bool = True,
        thinking_level: str = "low",
        image_timeout_s: float = 90.0,
        image_dir: Path = DEFAULT_SCREENSHOT_DIR,
        notify: Callable[[str], None] | None = None,
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
        self.image_dir = image_dir
        self.notify = notify or (lambda _msg: None)
        self._now = now

    @property
    def offers_tools(self) -> bool:
        """Whether the router should offer web_search / generate_image."""
        return self.web_search_enabled or self.image_generation_enabled

    # ------------------------------------------------------------------ code analysis
    def complete(self, prompt: str, *, instruction: str | None = None) -> str:
        """One-shot text answer with no tools and no memory (code analysis, summaries)."""
        body: dict[str, Any] = {
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

    # ------------------------------------------------------------------ web search
    def web_search(self, query: str) -> ExtraResult:
        query = query.strip()
        if not self.web_search_enabled:
            return ExtraResult("Web search is turned off in settings.", ok=False)
        if not query:
            return ExtraResult("What should I search for?", ok=False)
        self.notify("Searching the web.")
        try:
            found = self.client.create(
                {
                    "model": self.model,
                    "system_instruction": SEARCH_INSTRUCTION,
                    "tools": [{"type": "google_search"}],
                    "input": query,
                }
            )
        except QuotaExceededError as exc:
            if exc.billing_required:
                return ExtraResult(BILLING_NOTICE.format(feature="Web search"), ok=False)
            return ExtraResult("Web search quota is used up for now.", ok=False)
        except ProviderError as exc:
            return ExtraResult(f"Web search failed: {exc}", ok=False)
        return ExtraResult(found.text or "I found nothing on that.", sources=found.citations)

    # ------------------------------------------------------------------ images
    def generate_image(self, prompt: str, aspect_ratio: str = "1:1") -> ExtraResult:
        prompt = prompt.strip()
        if not self.image_generation_enabled:
            return ExtraResult("Image generation is turned off in settings.", ok=False)
        if not prompt:
            return ExtraResult("What should the image show?", ok=False)
        fmt: dict[str, Any] = {"type": "image", "mime_type": "image/jpeg"}
        if aspect_ratio in ASPECT_RATIOS:
            fmt["aspect_ratio"] = aspect_ratio
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
                return ExtraResult(BILLING_NOTICE.format(feature="Image generation"), ok=False)
            return ExtraResult("Image generation quota is used up for now.", ok=False)
        except ProviderError as exc:
            return ExtraResult(f"Image generation failed: {exc}", ok=False)
        if not made.images:
            return ExtraResult("The image model returned no image.", ok=False)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._now()))
        try:
            self.image_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for i, data in enumerate(made.images):
                path = self.image_dir / f"image_{stamp}{f'_{i + 1}' if i else ''}.jpg"
                path.write_bytes(data)
                paths.append(path)
        except OSError:
            return ExtraResult("Couldn't save the image.", ok=False)
        shown = self.registry.execute("screen.show_image", {"path": str(paths[0])})
        saved = f"Image saved as {paths[0].name} in the {self.image_dir.name} pictures folder."
        return ExtraResult(saved + (" It is open on screen." if shown.ok else ""), images=paths)


def build_extras(settings: Any, registry: Any) -> GeminiExtras | None:
    """GeminiExtras from settings, or None without a Gemini key, or in demo mode (no cloud)."""
    if settings.demo_mode or not settings.api_key("gemini"):
        return None
    g = settings.gemini
    client = GeminiClient(
        settings.api_key("gemini"), timeout_s=g.timeout_s, max_retries=settings.router.max_retries
    )
    return GeminiExtras(
        client,
        registry,
        model=g.model,
        image_model=g.image_model,
        web_search=g.web_search,
        image_generation=g.image_generation,
        thinking_level=g.thinking_level,
        image_timeout_s=g.image_timeout_s,
    )
