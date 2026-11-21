"""LLM provider clients.

Gemini is called through the Interactions REST API with the already-pinned httpx (no SDK, no new
dependency). Every call has a timeout and a retry limit; failures raise a ProviderError subclass
with a message that is safe to speak. The API key is sent only in the `x-goog-api-key` header and
never appears in logs, errors or reprs.

The S07 router adds Groq / OpenRouter / local llama-server clients here.
"""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
# Sent with image requests, as in Google's image-generation examples.
IMAGE_API_REVISION = "2026-05-20"


class ProviderError(Exception):
    """Base class. `str(exc)` is a short, user-safe explanation."""

    retryable = False


class ProviderUnavailableError(ProviderError):
    """Network error, timeout or 5xx."""

    retryable = True


class ProviderAuthError(ProviderError):
    """401/403: the key is wrong, revoked or lacks access."""


class ProviderRequestError(ProviderError):
    """400/404: the request (model name, tool schema...) was rejected."""


class QuotaExceededError(ProviderError):
    """429. `billing_required` means the free tier has no quota at all for this feature."""

    def __init__(self, message: str, *, billing_required: bool) -> None:
        super().__init__(message)
        self.billing_required = billing_required


@dataclass(frozen=True)
class Interaction:
    """One Interactions API response, with helpers for the parts GRACE uses."""

    id: str
    status: str
    steps: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Interaction:
        return cls(
            str(data.get("id", "")), str(data.get("status", "")), list(data.get("steps") or [])
        )

    def _blocks(self) -> list[dict[str, Any]]:
        return [
            block
            for step in self.steps
            if step.get("type") == "model_output"
            for block in step.get("content") or []
        ]

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self._blocks() if b.get("type") == "text").strip()

    @property
    def function_calls(self) -> list[dict[str, Any]]:
        return [s for s in self.steps if s.get("type") == "function_call"]

    @property
    def images(self) -> list[bytes]:
        out = []
        for b in self._blocks():
            if b.get("type") == "image" and b.get("data"):
                out.append(base64.b64decode(b["data"]))
        return out

    @property
    def citations(self) -> list[tuple[str, str]]:
        seen: dict[str, str] = {}
        for b in self._blocks():
            for a in b.get("annotations") or []:
                if a.get("type") == "url_citation" and a.get("url"):
                    seen.setdefault(a["url"], a.get("title") or a["url"])
        return [(title, url) for url, title in seen.items()]


def _error_for(status: int, body: Any) -> ProviderError:
    message = ""
    if isinstance(body, Mapping) and isinstance(body.get("error"), Mapping):
        message = str(body["error"].get("message", ""))
    lowered = message.lower()
    if status == 429:
        billing = "limit: 0" in lowered or "billing" in lowered or "upgrade your tier" in lowered
        what = "needs a billing-enabled Gemini key" if billing else "free quota is used up for now"
        return QuotaExceededError(f"Gemini {what}.", billing_required=billing)
    if status in (401, 403):
        return ProviderAuthError("The Gemini API key was rejected. Check GEMINI_API_KEY in .env.")
    if status in (400, 404):
        log.warning("gemini rejected the request (%s): %s", status, message[:300])
        return ProviderRequestError("Gemini couldn't handle that request.")
    if status >= 500:
        return ProviderUnavailableError("Gemini is having trouble right now.")
    return ProviderError(f"Gemini returned HTTP {status}.")


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_s: float = 20.0,
        max_retries: int = 1,
        url: str = INTERACTIONS_URL,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("GeminiClient needs an API key")
        self._url = url
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._sleep = sleep
        self._http = httpx.Client(
            transport=transport,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        )

    def __repr__(self) -> str:  # never show the key
        return f"GeminiClient(url={self._url!r})"

    def close(self) -> None:
        self._http.close()

    def create(
        self,
        body: Mapping[str, Any],
        *,
        timeout_s: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Interaction:
        """POST one interaction. Retries timeouts / network errors / 5xx up to max_retries."""
        timeout = timeout_s or self._timeout_s
        last: ProviderError = ProviderUnavailableError("I couldn't reach Gemini.")
        for attempt in range(self._max_retries + 1):
            if attempt:
                self._sleep(min(2.0, 0.5 * 2 ** (attempt - 1)))
            started = time.monotonic()
            try:
                response = self._http.post(
                    self._url, json=dict(body), timeout=timeout, headers=dict(headers or {})
                )
            except httpx.TimeoutException:
                last = ProviderUnavailableError("Gemini took too long to answer.")
                log.warning("gemini timeout after %.1fs (attempt %d)", timeout, attempt + 1)
                continue
            except httpx.HTTPError as exc:
                last = ProviderUnavailableError("I couldn't reach Gemini.")
                log.warning(
                    "gemini network error: %s (attempt %d)", type(exc).__name__, attempt + 1
                )
                continue
            elapsed = time.monotonic() - started
            try:
                data = response.json()
            except ValueError:
                data = None
            if response.status_code == 200 and isinstance(data, Mapping):
                interaction = Interaction.from_json(data)
                log.info(
                    "gemini %s status=%s in %.2fs", body.get("model"), interaction.status, elapsed
                )
                if interaction.status == "failed":
                    raise ProviderError("Gemini couldn't finish that request.")
                return interaction
            last = _error_for(response.status_code, data)
            log.info("gemini HTTP %d in %.2fs", response.status_code, elapsed)
            if not last.retryable:
                raise last
        raise last
