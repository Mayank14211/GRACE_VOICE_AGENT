"""LLM provider clients.

Chat (the router, S07): every provider is an OpenAI-compatible endpoint — Groq, Gemini's OpenAI
endpoint, OpenRouter and the local llama-server — reached through ONE factory,
``make_openai_client``, and wrapped in ``OpenAIChatProvider``. Base URLs and model IDs come from
settings (router.<provider>), never from code.

Gemini extras (web search grounding, image generation, code analysis) use ``GeminiClient`` over
the Interactions REST API with the pinned httpx: the OpenAI endpoint has no Google Search tool.

Every call has a timeout and a retry limit; failures raise a ProviderError subclass with a
message that is safe to speak. API keys are sent only in request headers and never appear in
logs, errors or reprs.
"""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

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


class ProviderTimeoutError(ProviderUnavailableError):
    """The provider did not answer within the router's budget. Not retried (the budget is
    spent); the router moves to the next provider."""

    retryable = False


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


# ============================================================================ chat (router)
LOCAL_API_KEY = "local"  # llama-server ignores it; the openai SDK needs a non-empty value


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text, validated by grace.intent.tools
    # Provider-specific fields sent back unchanged when the call is replayed (Gemini 3 rejects a
    # function-call turn without its extra_content.google.thought_signature). Opaque; not logged.
    extra: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class ChatReply:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    latency_ms: float = 0.0
    usage: Mapping[str, Any] = field(default_factory=dict)
    timings: Mapping[str, Any] = field(default_factory=dict)  # llama-server only


class ChatProvider(Protocol):
    """What the router needs from a provider. Real: OpenAIChatProvider; tests: FakeChatProvider."""

    name: str
    timeout_s: float

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        timeout_s: float | None = None,
    ) -> ChatReply: ...


def make_openai_client(
    base_url: str, api_key: str, *, http_client: httpx.Client | None = None
) -> Any:
    """The one OpenAI-compatible client factory. SDK retries are off: the router decides."""
    import openai  # imported here so a broken openai install only disables the router

    return openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0, http_client=http_client)


def _chat_error(provider: str, exc: Exception) -> ProviderError:
    """openai SDK exception -> ProviderError with a speakable message. Pure."""
    import openai

    label = provider.capitalize()
    if isinstance(exc, openai.APITimeoutError):
        return ProviderTimeoutError(f"{label} took too long to answer.")
    if isinstance(exc, openai.APIConnectionError):
        return ProviderUnavailableError(f"I couldn't reach {label}.")
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        if status == 429:
            return QuotaExceededError(
                f"{label} is busy or out of free quota.", billing_required=False
            )
        if status in (401, 403):
            return ProviderAuthError(f"The {label} API key was rejected. Check .env.")
        if status >= 500:
            return ProviderUnavailableError(f"{label} is having trouble right now.")
        return ProviderRequestError(f"{label} couldn't handle that request.")
    # APIError, or anything else the SDK raised (e.g. the body wasn't valid JSON).
    return ProviderUnavailableError(f"{label} sent a broken reply.")


class OpenAIChatProvider:
    """One OpenAI-compatible chat endpoint (groq, gemini, openrouter or local)."""

    def __init__(
        self,
        name: str,
        client: Any,
        model: str,
        *,
        timeout_s: float,
        max_retries: int = 1,
        reasoning_effort: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.name = name
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = client
        self._extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        self._sleep = sleep
        self._clock = clock

    def __repr__(self) -> str:  # the client holds the key: never show it
        return f"OpenAIChatProvider({self.name!r}, model={self.model!r})"

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        timeout_s: float | None = None,
    ) -> ChatReply:
        """One completion. Network errors / 5xx are retried up to max_retries; 429, timeouts,
        auth and request errors raise at once (the router cools the provider down)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout_s or self.timeout_s,
        }
        if tools:
            kwargs["tools"] = [dict(t) for t in tools]
            kwargs["tool_choice"] = "auto"
        if self._extra:
            kwargs["extra_body"] = self._extra
        last: ProviderError = ProviderUnavailableError(f"I couldn't reach {self.name}.")
        for attempt in range(self.max_retries + 1):
            if attempt:
                self._sleep(min(1.0, 0.25 * 2 ** (attempt - 1)))
            started = self._clock()
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - mapped below; nothing escapes unmapped
                last = _chat_error(self.name, exc)
                log.info(
                    "%s chat failed: %s (attempt %d)", self.name, type(last).__name__, attempt + 1
                )
                if not last.retryable:
                    raise last from None
                continue
            try:
                return _to_reply(response, (self._clock() - started) * 1000)
            except Exception:  # noqa: BLE001 - an unexpected response shape
                log.info("%s sent a reply GRACE can't read", self.name)
                raise ProviderUnavailableError(f"{self.name} sent a broken reply.") from None
        raise last


def _to_reply(response: Any, latency_ms: float) -> ChatReply:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ChatReply(latency_ms=latency_ms)
    message = choices[0].message
    calls = tuple(
        ToolCall(
            str(getattr(c, "id", "") or ""),
            str(getattr(c.function, "name", "") or ""),
            str(getattr(c.function, "arguments", "") or ""),
            dict(getattr(c, "model_extra", None) or {}),
        )
        for c in (getattr(message, "tool_calls", None) or [])
        if getattr(c, "function", None) is not None
    )
    usage = getattr(response, "usage", None)
    extra = getattr(response, "model_extra", None) or {}
    return ChatReply(
        content=str(getattr(message, "content", "") or "").strip(),
        tool_calls=calls,
        latency_ms=latency_ms,
        usage=usage.model_dump() if usage is not None else {},
        timings=dict(extra.get("timings") or {}),
    )


def build_chat_providers(
    settings: Any, *, http_client_factory: Callable[[str], httpx.Client] | None = None
) -> list[OpenAIChatProvider]:
    """Providers in settings order, skipping those without a key (or local not enabled)."""
    providers = []
    router = settings.router
    for name in settings.enabled_providers():
        endpoint = router.endpoint(name)
        key = LOCAL_API_KEY if name == "local" else settings.api_key(name)
        if not key:
            continue
        http_client = http_client_factory(name) if http_client_factory else None
        client = make_openai_client(endpoint.base_url, key, http_client=http_client)
        providers.append(
            OpenAIChatProvider(
                name,
                client,
                endpoint.model,
                timeout_s=router.timeout_s(name),
                max_retries=router.max_retries,
                reasoning_effort=endpoint.reasoning_effort,
            )
        )
    return providers
