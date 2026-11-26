"""web_search tool (Tavily) over a mocked HTTP transport. No network."""

from __future__ import annotations

import json

import httpx
import pytest

from grace.core.quota import Limits, QuotaGovernor
from grace.intent.search import QUOTA_KEY, TavilySearch

KEY = "tvly" + "-" + "FAKE0123456789"


def engine(handler, **kw):
    seen: list[httpx.Request] = []

    def record(request):
        seen.append(request)
        return handler(request)

    return TavilySearch(KEY, transport=httpx.MockTransport(record), **kw), seen


def ok_reply(answer="Canberra is the capital.", results=None, credits=1):
    results = (
        results
        if results is not None
        else [
            {
                "title": "Canberra",
                "url": "https://en.wikipedia.org/wiki/Canberra",
                "content": "C...",
                "score": 0.9,
            }
        ]
    )
    return httpx.Response(200, json={"query": "q", "answer": answer, "results": results,
                                     "images": [], "response_time": 0.8,
                                     "usage": {"credits": credits}})  # fmt: skip


def test_request_shape_and_answer() -> None:
    search, seen = engine(lambda r: ok_reply())
    result = search.search("  capital   of australia ")
    assert result.ok and result.answer == "Canberra is the capital."
    assert result.sources == [("Canberra", "https://en.wikipedia.org/wiki/Canberra")]
    req = seen[0]
    assert req.url == "https://api.tavily.com/search" and req.method == "POST"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert json.loads(req.content) == {
        "query": "capital of australia",
        "search_depth": "basic",
        "include_answer": "basic",
        "max_results": 3,
        "topic": "general",
    }
    assert KEY not in repr(search)


def test_no_answer_falls_back_to_the_first_result() -> None:
    search, _ = engine(
        lambda r: ok_reply(answer=None, results=[{"url": "https://x", "content": "First."}])
    )
    assert search.search("x").answer == "First."
    empty, _ = engine(lambda r: ok_reply(answer="", results=[]))
    assert empty.search("x").answer == "I found nothing on that."


@pytest.mark.parametrize(
    ("status", "fragment"),
    [(429, "busy"), (401, "TAVILY_API_KEY"), (432, "allowance"), (500, "failed")],
)
def test_http_errors_are_spoken(status: int, fragment: str) -> None:
    search, _ = engine(lambda r: httpx.Response(status, json={"detail": {"error": "x"}}))
    result = search.search("x")
    assert not result.ok and fragment in result.answer


def test_timeouts_and_network_errors() -> None:
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    def down(request):
        raise httpx.ConnectError("no network", request=request)

    assert "too long" in engine(slow)[0].search("x").answer
    assert "couldn't reach" in engine(down)[0].search("x").answer
    assert engine(lambda r: httpx.Response(200, content=b"<html>"))[0].search("x").ok is False


def test_credits_are_counted_and_the_month_limit_holds(tmp_path) -> None:
    quota = QuotaGovernor({QUOTA_KEY: Limits(units_per_month=2)}, state_path=tmp_path / "q.json",
                          clock=lambda: 1_800_000_000)  # fmt: skip
    search, seen = engine(lambda r: ok_reply(), quota=quota)
    assert search.search("a").ok and search.search("b").ok
    third = search.search("c")
    assert not third.ok and "allowance" in third.answer and len(seen) == 2  # never sent


def test_empty_query_is_not_sent() -> None:
    search, seen = engine(lambda r: ok_reply())
    assert not search.search("   ").ok and seen == []


def test_usage_check_for_the_doctor() -> None:
    search, seen = engine(lambda r: httpx.Response(200, json={"key": {"usage": 12, "limit": 1000}}))
    assert search.check() == (True, "key ok, 12 of 1000 credits used")
    assert seen[0].method == "GET" and seen[0].url == "https://api.tavily.com/usage"
    bad, _ = engine(lambda r: httpx.Response(401, json={}))
    assert bad.check() == (False, "HTTP 401")
