# Web search API for the `web_search` tool (S07, 2026-11-26)

Decision of 2026-11-26: `web_search` is GRACE's own tool, run by our code behind the
`SearchEngine` interface (`grace/intent/search.py`), backed by **one** free search API. Gemini
Search grounding is out, because it needs billing.

## Choice: Tavily

| Option | Free allowance (as published, Nov 2026) | Card needed | Fit for GRACE |
|---|---|---|---|
| **Tavily** | 1,000 credits / month; a basic search = 1 credit | No | **Chosen.** One REST call (`POST /search`, Bearer key) returns a ready short `answer` plus sources, so GRACE speaks two sentences without a second LLM call. `GET /usage` lets the doctor check the key. |
| Brave Search API | $5 monthly credit; the old 2,000/month free tier was removed in 2026 | Yes | A card is required: conflicts with "free tiers only, no card". |
| Serper (Google results) | 2,500 queries, one-time, no monthly refill | No | Runs out for good; results only, no answer. |
| Exa | Reported as 1,000 requests/month or a $20 credit (sources disagree) | Unclear | Results only, so it needs an extra LLM pass to summarise. |
| Keyless options (e.g. Firecrawl keyless) | Capped per IP per day | No | No key to revoke, and an opaque per-IP cap. Too fragile to depend on. |

Sources: [Tavily search API reference](https://docs.tavily.com/documentation/api-reference/endpoint/search),
[Tavily usage endpoint](https://docs.tavily.com/documentation/api-reference/endpoint/usage),
[Firecrawl: best free web search APIs 2026](https://www.firecrawl.dev/blog/best-free-web-search-apis),
[Parallel: free search APIs compared 2026](https://parallel.ai/articles/best-free-web-search-api).
Re-check the free tiers before Phase 2; they changed several times in 2026.

## How GRACE uses it

- Request: `{"query", "search_depth": "basic", "include_answer": "basic", "max_results": 3,
  "topic": "general"}`. It costs 1 credit, 8 s timeout. Only the query is sent: never screen
  text or the conversation.
- The quota governor (`grace/core/quota.py`, key `tavily:search`) counts credits per UTC month
  against `search.credits_per_month` (1,000). At 80 % GRACE warns; at 100 % it says the
  allowance is used up instead of calling.
- The answer is spoken as at most two sentences; sources are printed in the console.
- Failures (timeout, 401, 429, 432/433 plan limits, 5xx) become one spoken sentence; nothing
  raises.
- Swapping the provider means one new class with `search(query) -> SearchResult`. Nothing else
  changes.

## Measured

Not yet: needs `TAVILY_API_KEY` in `.env`. Run
`uv run python scripts/eval_router.py --force-429 5 --save` (3 of its 20 requests are searches)
and record the median latency and answer quality here.
