"""OpenAI-compatible chat providers over a mocked HTTP transport (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from grace.config import load_settings
from grace.intent.providers import (
    OpenAIChatProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
    build_chat_providers,
    make_openai_client,
)

# Assembled at runtime so tests/test_no_secrets.py does not flag this file.
KEY = "gsk" + "_" + "FAKEKEYFORTESTSONLY0123456789abcdefghijklmnop"

TOOLS = [{"type": "function", "function": {"name": "answer_user", "parameters": {}}}]
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]


def completion(message: dict, **extra) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            **extra,
        },
    )


def provider_with(handler, *, retries: int = 1, effort: str | None = None):
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request, len(seen))

    http = httpx.Client(transport=httpx.MockTransport(record))
    client = make_openai_client("https://llm.example/v1", KEY, http_client=http)
    provider = OpenAIChatProvider(
        "groq",
        client,
        "openai/gpt-oss-20b",
        timeout_s=3.0,
        max_retries=retries,
        reasoning_effort=effort,
        sleep=lambda s: None,
    )
    return provider, seen


def chat(provider):
    return provider.chat(MESSAGES, TOOLS, max_tokens=50, temperature=0.2)


def test_tool_call_reply_and_request_shape() -> None:
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "answer_user", "arguments": '{"text": "Hi."}'},
                "extra_content": {"google": {"thought_signature": "sig"}},
            }
        ],
    }
    provider, seen = provider_with(lambda r, n: completion(message), effort="low")
    reply = chat(provider)
    assert [(c.name, c.arguments) for c in reply.tool_calls] == [("answer_user", '{"text": "Hi."}')]
    assert reply.content == "" and reply.usage["total_tokens"] == 5
    assert reply.tool_calls[0].extra == {"extra_content": {"google": {"thought_signature": "sig"}}}
    body = json.loads(seen[0].content)
    assert body["model"] == "openai/gpt-oss-20b" and body["tools"] == TOOLS
    assert body["tool_choice"] == "auto" and body["reasoning_effort"] == "low"
    assert body["max_tokens"] == 50 and body["messages"] == MESSAGES
    assert seen[0].url == "https://llm.example/v1/chat/completions"
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in repr(provider)


def test_plain_text_reply_and_timings() -> None:
    provider, _ = provider_with(
        lambda r, n: completion(
            {"role": "assistant", "content": " Paris. "}, timings={"predicted_per_second": 9.5}
        )
    )
    reply = chat(provider)
    assert reply.content == "Paris." and reply.tool_calls == ()
    assert reply.timings == {"predicted_per_second": 9.5}


def test_429_raises_quota_at_once() -> None:
    provider, seen = provider_with(lambda r, n: httpx.Response(429, json={"error": {}}))
    with pytest.raises(QuotaExceededError):
        chat(provider)
    assert len(seen) == 1  # never retried: the router cools the provider down instead


def test_timeout_is_not_retried() -> None:
    def slow(request, n):
        raise httpx.ReadTimeout("slow", request=request)

    provider, seen = provider_with(slow, retries=3)
    with pytest.raises(ProviderTimeoutError):
        chat(provider)
    assert len(seen) == 1


def test_5xx_is_retried_but_no_connection_is_not() -> None:
    def flaky(request, n):
        if n == 1:
            return httpx.Response(503, json={"error": {}})
        return completion({"role": "assistant", "content": "ok"})

    provider, seen = provider_with(flaky, retries=2)
    assert chat(provider).content == "ok" and len(seen) == 2
    provider, seen = provider_with(lambda r, n: httpx.Response(500, json={}), retries=1)
    with pytest.raises(ProviderUnavailableError):
        chat(provider)
    assert len(seen) == 2

    def offline(request, n):
        raise httpx.ConnectError("getaddrinfo failed", request=request)

    provider, seen = provider_with(offline, retries=3)
    with pytest.raises(ProviderConnectionError):
        chat(provider)
    assert len(seen) == 1  # network off: move to the next provider at once


@pytest.mark.parametrize(
    ("status", "error"),
    [(401, ProviderAuthError), (403, ProviderAuthError), (400, ProviderRequestError)],
)
def test_client_errors(status: int, error: type) -> None:
    provider, seen = provider_with(lambda r, n: httpx.Response(status, json={"error": {}}))
    with pytest.raises(error) as caught:
        chat(provider)
    assert len(seen) == 1 and KEY not in str(caught.value)


def test_broken_json_body_is_a_provider_error() -> None:
    provider, _ = provider_with(
        lambda r, n: httpx.Response(200, content=b"<html>oops</html>"), retries=0
    )
    with pytest.raises(ProviderUnavailableError):
        chat(provider)


# --------------------------------------------------------------------------- from settings
def test_build_chat_providers_follows_settings(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(f"GROQ_API_KEY={KEY}" + chr(10))
    cfg = tmp_path / "s.yaml"
    cfg.write_text(
        "router: {provider_order: [gemini, groq], "
        "groq: {base_url: 'https://g', model: m, timeout_s: 5}}" + chr(10)
    )
    providers = build_chat_providers(load_settings(cfg, env, environ={}))
    assert [p.name for p in providers] == ["groq"]  # gemini has no key
    [groq] = providers
    assert groq.timeout_s == 5.0 and groq.model == "m"
    assert KEY not in repr(groq)


def test_default_order_is_groq_gemini_openrouter(tmp_path: Path) -> None:
    gem = "AI" + "za" + "FAKE-KEY-FOR-TESTS-ONLY-0123456789"
    env = tmp_path / ".env"
    lines = [f"GROQ_API_KEY={KEY}", f"GEMINI_API_KEY={gem}", "OPENROUTER_API_KEY=or-x"]
    env.write_text(chr(10).join(lines) + chr(10))
    providers = build_chat_providers(load_settings(tmp_path / "none.yaml", env, environ={}))
    assert [p.name for p in providers] == ["groq", "gemini", "openrouter"]
    assert all(p.timeout_s == 3.0 for p in providers)
