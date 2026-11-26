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


def test_plain_text_reply_and_llama_server_timings() -> None:
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


def test_5xx_and_connection_errors_are_retried() -> None:
    def flaky(request, n):
        if n == 1:
            return httpx.Response(503, json={"error": {}})
        if n == 2:
            raise httpx.ConnectError("refused", request=request)
        return completion({"role": "assistant", "content": "ok"})

    provider, seen = provider_with(flaky, retries=2)
    assert chat(provider).content == "ok" and len(seen) == 3
    provider, seen = provider_with(lambda r, n: httpx.Response(500, json={}), retries=1)
    with pytest.raises(ProviderUnavailableError):
        chat(provider)
    assert len(seen) == 2


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
    env.write_text(f"GROQ_API_KEY={KEY}\n")
    cfg = tmp_path / "s.yaml"
    cfg.write_text("router: {provider_order: [local, groq, gemini], local: {enabled: true}}\n")
    providers = build_chat_providers(load_settings(cfg, env, environ={}))
    assert [p.name for p in providers] == ["local", "groq"]  # gemini has no key
    local, groq = providers
    assert (local.timeout_s, groq.timeout_s) == (15.0, 3.0)
    assert groq.model == "openai/gpt-oss-20b" and local.model == "gemma-4-e4b-it"
    assert all(KEY not in repr(p) for p in providers)


def test_demo_mode_builds_local_only(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(f"GROQ_API_KEY={KEY}\n")
    cfg = tmp_path / "s.yaml"
    cfg.write_text("demo_mode: true\n")  # local is switched on by demo mode itself
    assert [p.name for p in build_chat_providers(load_settings(cfg, env, environ={}))] == ["local"]
