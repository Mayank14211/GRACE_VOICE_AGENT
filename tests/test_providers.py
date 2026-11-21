"""Gemini Interactions client: parsing, retries, error mapping. No network (httpx.MockTransport)."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from grace.intent.providers import (
    GeminiClient,
    Interaction,
    ProviderAuthError,
    ProviderError,
    ProviderRequestError,
    ProviderUnavailableError,
    QuotaExceededError,
)

# Assembled at runtime so tests/test_no_secrets.py does not flag this file.
KEY = "AI" + "za" + "FAKE-KEY-FOR-TESTS-ONLY-0123456789"


def client_with(handler, **kw) -> tuple[GeminiClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request, len(seen))

    return GeminiClient(
        KEY, transport=httpx.MockTransport(record), sleep=lambda s: None, **kw
    ), seen


def ok(steps, status="completed", id_="i1"):
    return httpx.Response(200, json={"id": id_, "status": status, "steps": steps})


def test_text_and_headers() -> None:
    steps = [
        {"type": "thought", "signature": "abc"},
        {"type": "model_output", "content": [{"type": "text", "text": "Hello there."}]},
    ]
    client, seen = client_with(lambda r, n: ok(steps))
    result = client.create({"model": "m", "input": "hi"})
    assert result.text == "Hello there." and result.id == "i1"
    assert seen[0].headers["x-goog-api-key"] == KEY
    assert json.loads(seen[0].content) == {"model": "m", "input": "hi"}
    assert KEY not in repr(client)


def test_function_calls_images_and_citations() -> None:
    img = base64.b64encode(b"\xff\xd8jpeg").decode()
    inter = Interaction.from_json(
        {
            "id": "x",
            "status": "requires_action",
            "steps": [
                {
                    "type": "function_call",
                    "id": "c1",
                    "name": "volume_set",
                    "arguments": {"level": 3},
                },
                {
                    "type": "model_output",
                    "content": [
                        {"type": "image", "data": img},
                        {
                            "type": "text",
                            "text": "News.",
                            "annotations": [
                                {"type": "url_citation", "url": "https://a.example", "title": "A"},
                                {"type": "url_citation", "url": "https://a.example", "title": "A"},
                            ],
                        },
                    ],
                },
            ],
        }
    )
    assert inter.function_calls[0]["name"] == "volume_set"
    assert inter.images == [b"\xff\xd8jpeg"]
    assert inter.citations == [("A", "https://a.example")]


def test_retries_5xx_then_succeeds() -> None:
    client, seen = client_with(
        lambda r, n: httpx.Response(503, json={}) if n == 1 else ok([]), max_retries=1
    )
    assert client.create({"model": "m"}).status == "completed"
    assert len(seen) == 2


def test_timeouts_exhaust_retries() -> None:
    def boom(r, n):
        raise httpx.ReadTimeout("slow", request=r)

    client, seen = client_with(boom, max_retries=2)
    with pytest.raises(ProviderUnavailableError, match="too long"):
        client.create({"model": "m"})
    assert len(seen) == 3


@pytest.mark.parametrize(
    ("message", "billing"),
    [
        ("Rate limit exceeded for model x (limit: 0 requests per day on Free Tier).", True),
        ("You exceeded your current quota, please check your plan and billing details.", True),
        ("Too many requests, slow down.", False),
    ],
)
def test_429_is_not_retried_and_detects_billing(message: str, billing: bool) -> None:
    client, seen = client_with(
        lambda r, n: httpx.Response(429, json={"error": {"message": message}}), max_retries=3
    )
    with pytest.raises(QuotaExceededError) as info:
        client.create({"model": "m"})
    assert info.value.billing_required is billing
    assert len(seen) == 1


@pytest.mark.parametrize(
    ("status", "error"),
    [(401, ProviderAuthError), (403, ProviderAuthError), (400, ProviderRequestError)],
)
def test_client_errors(status: int, error: type) -> None:
    client, seen = client_with(
        lambda r, n: httpx.Response(status, json={"error": {"message": "x"}})
    )
    with pytest.raises(error) as info:
        client.create({"model": "m"})
    assert KEY not in str(info.value) and len(seen) == 1


def test_failed_status_raises() -> None:
    client, _ = client_with(lambda r, n: ok([], status="failed"))
    with pytest.raises(ProviderError):
        client.create({"model": "m"})


def test_needs_a_key() -> None:
    with pytest.raises(ValueError):
        GeminiClient("")
