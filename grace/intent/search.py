"""web_search tool: our own search, run by GRACE's code (the LLM only emits the tool call).

One free search API behind the ``SearchEngine`` interface: Tavily (choice and reasons in
docs/results/search.md). Every call has a timeout and goes through the quota governor
(1,000 credits a month free; a basic search costs 1 credit). Failures return a short spoken
explanation; nothing raises.

The query is the only thing sent: never screen text, clipboard contents or the conversation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from grace.core.quota import NullQuota, Quota

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_USAGE_URL = "https://api.tavily.com/usage"
QUOTA_KEY = "tavily:search"


@dataclass(frozen=True)
class SearchResult:
    ok: bool
    answer: str  # what to say (already short), or why it failed
    sources: list[tuple[str, str]] = field(default_factory=list)  # (title, url)
    latency_ms: float = 0.0


class SearchEngine(Protocol):
    name: str

    def search(self, query: str) -> SearchResult: ...


class FakeSearch:
    """Scripted engine for tests and dry runs."""

    name = "fake"

    def __init__(self, answer: str = "Fake answer.") -> None:
        self.answer = answer
        self.queries: list[str] = []

    def search(self, query: str) -> SearchResult:
        self.queries.append(query)
        return SearchResult(True, self.answer, [("Example", "https://example.com")])


class TavilySearch:
    name = "tavily"

    def __init__(
        self,
        api_key: str,
        *,
        max_results: int = 3,
        timeout_s: float = 8.0,
        quota: Quota | None = None,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not api_key:
            raise ValueError("TavilySearch needs an API key")
        self.max_results = max_results
        self.timeout_s = timeout_s
        self.quota = quota or NullQuota()
        self._clock = clock
        self._http = httpx.Client(
            transport=transport, headers={"Authorization": f"Bearer {api_key}"}
        )

    def __repr__(self) -> str:  # never show the key
        return "TavilySearch()"

    def search(self, query: str) -> SearchResult:
        query = " ".join(query.split())[:400]
        if not query:
            return SearchResult(False, "What should I search for?")
        if not self.quota.allow(QUOTA_KEY, units=1):
            return SearchResult(False, "The free web search allowance is used up for this month.")
        body = {
            "query": query,
            "search_depth": "basic",  # 1 credit ("advanced" costs 2)
            "include_answer": "basic",
            "max_results": self.max_results,
            "topic": "general",
        }
        started = self._clock()
        try:
            response = self._http.post(TAVILY_URL, json=body, timeout=self.timeout_s)
        except httpx.TimeoutException:
            return SearchResult(False, "Web search took too long.")
        except httpx.HTTPError as exc:
            log.info("tavily network error: %s", type(exc).__name__)
            return SearchResult(False, "I couldn't reach the web search.")
        ms = (self._clock() - started) * 1000
        if response.status_code == 429:
            return SearchResult(False, "Web search is busy right now; try again in a minute.")
        if response.status_code in (401, 403):
            return SearchResult(False, "The search key was rejected. Check TAVILY_API_KEY in .env.")
        if response.status_code == 432 or response.status_code == 433:
            return SearchResult(False, "The free web search allowance is used up.")
        if response.status_code != 200:
            log.info("tavily HTTP %d", response.status_code)
            return SearchResult(False, "Web search failed.")
        try:
            data = response.json()
        except ValueError:
            return SearchResult(False, "Web search sent a broken reply.")
        credits = (data.get("usage") or {}).get("credits", 1)
        self.quota.record(QUOTA_KEY, units=float(credits or 1))
        results = [r for r in data.get("results") or [] if isinstance(r, dict)]
        sources = [
            (str(r.get("title") or r.get("url")), str(r.get("url")))
            for r in results
            if r.get("url")
        ]
        answer = str(data.get("answer") or "").strip()
        if not answer and results:
            answer = str(results[0].get("content") or "").strip()
        log.info("tavily search: %d results in %.0f ms", len(results), ms)
        if not answer:
            return SearchResult(True, "I found nothing on that.", sources, ms)
        return SearchResult(True, answer, sources, ms)

    def check(self) -> tuple[bool, str]:
        """Doctor: is the key accepted? GET /usage (a read; spends no search credit)."""
        try:
            response = self._http.get(TAVILY_USAGE_URL, timeout=5.0)
        except httpx.HTTPError as exc:
            return False, f"not reachable ({type(exc).__name__})"
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"
        try:
            data = response.json()
        except ValueError:
            return False, "unreadable reply"
        key, account = data.get("key") or {}, data.get("account") or {}
        # The plan's limit is on the account; a key's own limit is usually null.
        used = account.get("plan_usage", key.get("usage"))
        limit = account.get("plan_limit", key.get("limit"))
        plan = f" ({account['current_plan']} plan)" if account.get("current_plan") else ""
        shown = limit if limit is not None else "unlimited"
        return True, f"key ok, {used} of {shown} credits used this cycle{plan}"


def build_search(settings: object, quota: Quota | None = None) -> SearchEngine | None:
    s = settings.search  # type: ignore[attr-defined]
    key = settings.secret("tavily")  # type: ignore[attr-defined]
    if not s.enabled or not key:
        return None
    return TavilySearch(key, max_results=s.max_results, timeout_s=s.timeout_s, quota=quota)
