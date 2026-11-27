"""The router's tool executors, behind one object the voice agent calls (``extras``).

The LLM only emits tool calls; GRACE's own code runs them (2026-11-26 decision):

- ``web_search``: our search engine (grace/intent/search.py, Tavily). The spoken answer is
  at most two sentences; sources are printed.
- ``generate_image``: Cloudflare Workers AI FLUX.1 [schnell] (grace/intent/images.py), after
  the prompt filter. Saved to Pictures/GRACE and opened through the registry.
- ``complete``: "analyze / debug this code". Screen text and selected code go through the
  router's providers WITHOUT Gemini (Gemini's free tier may train on inputs, so it never
  receives screen text).

Until S07 this module held a Gemini assistant (Interactions API with server-side memory,
Gemini search grounding and Gemini images). All of that is gone: conversation lives in the
router, which sends the history itself on every call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grace.intent.images import ImageGenerator, build_images
from grace.intent.search import SearchEngine, build_search

PRIVATE_EXCLUDE = frozenset({"gemini"})  # providers that never see screen text


@dataclass
class ExtraResult:
    text: str  # what to say
    ok: bool = True
    images: list[Path] = field(default_factory=list)
    sources: list[tuple[str, str]] = field(default_factory=list)  # (title, url)


class Extras:
    def __init__(
        self,
        *,
        search: SearchEngine | None = None,
        images: ImageGenerator | None = None,
        analyst: Any = None,  # grace.intent.router.Router (complete without tools)
        notify: Any = None,
    ) -> None:
        self.search = search
        self.images = images
        self.analyst = analyst
        self.notify = notify or (lambda _msg: None)

    @property
    def offers_search(self) -> bool:
        return self.search is not None

    @property
    def offers_images(self) -> bool:
        return self.images is not None

    def web_search(self, query: str) -> ExtraResult:
        if self.search is None:
            return ExtraResult("Web search needs a TAVILY_API_KEY in the .env file.", ok=False)
        self.notify("Searching the web.")
        found = self.search.search(query)
        return ExtraResult(found.answer, ok=found.ok, sources=list(found.sources))

    def generate_image(self, prompt: str) -> ExtraResult:
        if self.images is None:
            return ExtraResult(
                "Image generation needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env.",
                ok=False,
            )
        self.notify("Creating the image.")
        made = self.images.generate(prompt)
        return ExtraResult(made.message, ok=made.ok, images=list(made.images))

    def complete(self, prompt: str, *, instruction: str | None = None) -> str:
        """Code analysis on screen / selected text: never Gemini."""
        if self.analyst is None:
            return "Code analysis needs a Groq or OpenRouter key in the .env file."
        return self.analyst.complete(prompt, instruction=instruction, exclude=PRIVATE_EXCLUDE)


def build_extras(settings: Any, registry: Any, *, router: Any = None, quota: Any = None) -> Extras:
    """Tool executors from settings. A tool without its key is simply not offered."""
    return Extras(
        search=build_search(settings, quota),
        images=build_images(settings, registry, quota),
        analyst=router,
    )
