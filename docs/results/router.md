# M6 results: intent router, tools and conversation

Gate (docs/PLAN.md section 8 and the S07 continuation, 2026-11-26): **no crash on any
provider failure. 20 free-form requests logged with provider, latency and success: 3+ images,
3+ searches, 5 with a forced HTTP 429, the fallback keeping the history, and one run with the
network off.**

Measured with `scripts/eval_router.py`. It sends 20 free-form requests in order through the voice
agent: grammar, then follow-ups, then the router, with one dialogue so the memory cases are
multi-turn. OS actions go to fakes; images land in `logs/eval_images/`. Searches (Tavily) and
images (Cloudflare FLUX.1 schnell) are real calls. `--save` replaces its own section below.

## Findings (S07 continuation, 2026-11-26, base machine i5-12450H)

- **Acceptance run** (`--force-429 5 --pause 12`, Groq -> Gemini): **18/20 succeeded, nothing
  crashed, 8/8 tool calls ran live** (4 searches, 4 images). The two misses:
  - Requests 2 and 3 ("capital of Australia", "how many people live there") were answered
    with `web_search` instead of a direct answer. That is defensible for facts, but this run's
    scoring only accepts `app.answer`, so both count as misses.
- **Forced 429 x5:** every one of requests 1-5 got a real HTTP 429 from Groq and was answered by
  Gemini (median 4.9 s, includes the tool call).
- **Fallback keeps the history, live:** with Groq forced to 429, Gemini got "what is the capital
  of australia" -> "The capital of Australia is Canberra.", then "and how many people live
  there" -> `web_search("population of Canberra 2026")`. It resolved "there" from the history
  the router sent; the provider holds no conversation state.
- **Groq:** median **860 ms** per request. Its free tier allows 1,000 requests/day and **8,000
  tokens/minute** (confirmed from Groq's `x-ratelimit-*` response headers). Each request carries
  about 1.6k tokens of prompt and tools, so Groq takes about **5 requests a minute**. The quota
  governor warned at 80 % and, in a faster run (`--pause 5`), routed the rest to Gemini.
- **Gemini at 3 s missed too often** (1.4-3.5 s, tail 4-6 s on the free tier): in the `--pause 5`
  run it timed out on requests 1 and 15, cooled down, and requests 2-5 and 16-17 then had no
  fallback. Its default timeout is now **6 s**; Groq and OpenRouter keep 3 s.
- **Network off** (simulated: every connection sent to a dead proxy): 5/5 router requests got
  the spoken "can't reach" reply, and 5/5 fixed commands worked. The first router request took
  4.2 s: one attempt per provider at about 2 s each, which is how long Windows takes to refuse a
  localhost connection. With Wi-Fi really off, DNS fails at once. Later requests: 0 ms
  (cooldown). Connection errors are no longer retried; before that fix the first request took
  8.8 s.
- Doctor reachability (calls that spend no quota): Groq and Gemini model list PASS, Cloudflare
  token active (account-scoped token: `/accounts/{id}/tokens/verify`), Tavily 0 of 1,000
  credits used.

## Manual acceptance (team, base machine)

1. `uv run python -m grace doctor`. Expected: `reach groq / gemini / cloudflare / tavily`
   PASS, `router tools PASS`, 0 FAIL.
2. **Wi-Fi really off**: `uv run python scripts/eval_router.py --offline --save`. Expected:
   5/5 spoken fallbacks, each under 1 s (DNS fails at once), 5/5 fixed commands work.
3. Live voice, `uv run python -m grace`: say 20 free-form requests of your own, including 3
   "search the web for ..." and 3 "draw a picture of ...". Expected: each gets an action, a
   spoken answer (at most 2 sentences) or a picture opened from `Pictures\GRACE`. The status
   chip shows the provider. `logs/metrics/session-*.jsonl` has one `router` row per request
   (provider, latency_ms, ok, intents; never the words).
4. Say "draw a naked person". Expected: "I can't make that image: it asks for sexual
   content." Nothing is sent to Cloudflare.
5. Say "analyze this code" with code selected and only `GEMINI_API_KEY` set (remove the Groq
   key for the test). Expected: "Code analysis needs a Groq or OpenRouter key". Screen text
   never goes to Gemini.
6. Say "turn everything off for tonight". Expected: no shutdown. GRACE tells you to say "shut
   down the computer", which asks for confirmation.
7. Speak fast for a minute (more than 5 requests). Expected: an overlay line "Heads up: groq is
   at 80% of its free tpm limit", then answers continue from Gemini.
8. Old settings: put `router: {local: {enabled: true}}` in settings.yaml. Expected: startup
   stops with "router.local was removed: no local LLM since 2026-11-26 ...".

## Earlier runs (first part of S07, old 20-request set without tools)

These used the request list from before the 2026-11-26 decisions (no searches or images), so
they are not the acceptance run. They are kept for the latency history.

<!-- eval_router_gemini:start -->
### eval_router (gemini (8 s))

Run 2026-11-26 15:42 on 12th Gen Intel(R) Core(TM) i5-12450H (Windows 11 build 26300). Chain: gemini (8 s). 15/20 router requests succeeded. gemini: 15 requests, median 1453 ms, none: 5 requests, median 0 ms.

| # | request | provider | ms | intents | success |
|---|---|---|---|---|---|
| 1 | could you make it a little quieter | gemini | 2609 | volume.step | yes |
| 2 | it is too dark, brighten the screen up a bit | gemini | 1407 | brightness.step | yes |
| 3 | put youtube on for me | gemini | 1343 | apps.open | yes |
| 4 | what is the capital of australia | gemini | 1625 | app.answer | yes |
| 5 | and how many people live there | gemini | 1235 | app.answer | yes |
| 6 | move the page down a bit | gemini | 1187 | input.scroll | yes |
| 7 | silence everything | gemini | 1141 | volume.mute | yes |
| 8 | actually bring the sound back | gemini | 1797 | volume.mute | yes |
| 9 | write hello world in here | gemini | 1437 | input.type | yes |
| 10 | copy whatever I have selected | gemini | 1406 | clipboard.copy | yes |
| 11 | get rid of this tab | gemini | 1797 | apps.close_tab | yes |
| 12 | I need the calculator | gemini | 1969 | apps.open | yes |
| 13 | what sort of things can you help me with | gemini | 1453 | app.answer | yes |
| 14 | tell me a short joke | gemini | 4578 | app.answer | yes |
| 15 | read out what is on my screen | gemini | 3047 | app.read_screen | yes |
| 16 | grab a picture of my screen | none | 8188 | gemini: ProviderTimeoutError | no |
| 17 | bring up the task manager | none | 0 | gemini: cooldown | no |
| 18 | how much is fifteen percent of two hundred | none | 0 | gemini: cooldown | no |
| 19 | put the brightness at about seventy | none | 0 | gemini: cooldown | no |
| 20 | turn my computer off | none | 0 | gemini: cooldown | no |
<!-- eval_router_gemini:end -->

<!-- eval_router_groq_gemini_v1:start -->
### eval_router (groq (3 s), gemini (3 s)): team run, old request set

Run 2026-11-26 16:54 on 12th Gen Intel(R) Core(TM) i5-12450H (Windows 11 build 26300). Chain: groq (3 s) -> gemini (3 s). 19/20 router requests succeeded. groq: 13 requests, median 563 ms, gemini: 7 requests, median 1594 ms.

| # | request | provider | ms | intents | success |
|---|---|---|---|---|---|
| 1 | could you make it a little quieter | groq | 984 | volume.step | yes |
| 2 | it is too dark, brighten the screen up a bit | groq | 391 | brightness.step | yes |
| 3 | put youtube on for me | groq | 421 | apps.open | yes |
| 4 | what is the capital of australia | groq | 375 | app.answer | yes |
| 5 | and how many people live there | groq | 563 | app.answer | yes |
| 6 | move the page down a bit | groq | 625 | input.scroll | yes |
| 7 | silence everything | groq | 640 | volume.mute | yes |
| 8 | actually bring the sound back | groq | 640 | volume.mute | yes |
| 9 | write hello world in here | groq | 516 | input.type | yes |
| 10 | copy whatever I have selected | groq | 688 | clipboard.copy | yes |
| 11 | get rid of this tab | groq | 359 | apps.close_tab | yes |
| 12 | I need the calculator | groq | 562 | apps.open | yes |
| 13 | what sort of things can you help me with | groq | 625 | app.answer | yes |
| 14 | tell me a short joke | gemini | 1719 | app.answer | yes |
| 15 | read out what is on my screen | gemini | 1594 | app.read_screen | yes |
| 16 | grab a picture of my screen | gemini | 859 | screen.screenshot | yes |
| 17 | bring up the task manager | gemini | 2516 | input.hotkey | yes |
| 18 | how much is fifteen percent of two hundred | gemini | 921 | apps.open | no |
| 19 | put the brightness at about seventy | gemini | 2938 | brightness.set | yes |
| 20 | turn my computer off | gemini | 1156 | app.answer | yes |
<!-- eval_router_groq_gemini_v1:end -->


## Acceptance runs (S07 continuation)

<!-- eval_router_groq_gemini:start -->
### eval_router (groq (3 s), gemini (6 s)), --force-429 5

Run 2026-11-26 22:17 on 12th Gen Intel(R) Core(TM) i5-12450H (Windows 11 build 26300). Chain: groq (3 s) -> gemini (6 s). 18/20 router requests succeeded. gemini: 5 requests, median 4906 ms, groq: 15 requests, median 860 ms. Tools: 8/8 executor calls ran. Requests 1-5: forced HTTP 429 from the first provider.

| # | request | provider | ms | intents | tool | attempts | success |
|---|---|---|---|---|---|---|---|
| 1 | could you make it a little quieter | gemini | 5516 | volume.step | - | groq: QuotaExceededError | yes |
| 2 | what is the capital of australia | gemini | 5078 | app.web_search | ok | groq: QuotaExceededError | no |
| 3 | and how many people live there | gemini | 4906 | app.web_search | ok | groq: QuotaExceededError | no |
| 4 | search the web for today's weather in Delhi | gemini | 3125 | app.web_search | ok | groq: QuotaExceededError | yes |
| 5 | draw me a picture of a cat wearing sunglasses | gemini | 4000 | app.generate_image | ok | groq: QuotaExceededError | yes |
| 6 | it is too dark, brighten the screen up a bit | groq | 875 | brightness.step | - | - | yes |
| 7 | put youtube on for me | groq | 844 | apps.open | - | - | yes |
| 8 | look up who won the last cricket world cup | groq | 625 | app.web_search | ok | - | yes |
| 9 | make an image of a lighthouse at sunset | groq | 797 | app.generate_image | ok | - | yes |
| 10 | move the page down a bit | groq | 812 | input.scroll | - | - | yes |
| 11 | silence everything | groq | 907 | volume.mute | - | - | yes |
| 12 | actually bring the sound back | groq | 921 | volume.mute | - | - | yes |
| 13 | write hello world in here | groq | 860 | input.type | - | - | yes |
| 14 | find the latest news about the James Webb telescope | groq | 765 | app.web_search | ok | - | yes |
| 15 | create a picture of a red bicycle in the rain | groq | 1218 | app.generate_image | ok | - | yes |
| 16 | get rid of this tab | groq | 906 | apps.close_tab | - | - | yes |
| 17 | tell me a short joke | groq | 609 | app.answer | - | - | yes |
| 18 | read out what is on my screen | groq | 828 | app.read_screen | - | - | yes |
| 19 | put the brightness at about seventy | groq | 860 | brightness.set | - | - | yes |
| 20 | turn my computer off | groq | 875 | app.answer | - | - | yes |
<!-- eval_router_groq_gemini:end -->

<!-- eval_router_offline:start -->
### eval_router --offline

Run 2026-11-27 06:10 on 12th Gen Intel(R) Core(TM) i5-12450H (Windows 11 build 26300), network off (every connection sent to a dead proxy). Chain: groq (3 s) -> gemini (6 s). Router: 5/5 requests got the spoken fallback, slowest 4230 ms. Fixed commands: 5/5 worked.

| request | kind | ms | result |
|---|---|---|---|
| could you make it a little quieter | router | 4230 | groq: ProviderConnectionError, gemini: ProviderConnectionError |
| what is the capital of australia | router | 31 | groq: cooldown, gemini: cooldown |
| and how many people live there | router | 1 | groq: cooldown, gemini: cooldown |
| search the web for today's weather in Delhi | router | 1 | groq: cooldown, gemini: cooldown |
| draw me a picture of a cat wearing sunglasses | router | 1 | groq: cooldown, gemini: cooldown |
| volume 40 | grammar | 2 | volume.set |
| a bit more | followup | 2 | volume.step |
| brightness 60 | grammar | 1 | brightness.set |
| scroll down | grammar | 1 | input.scroll |
| open notepad | grammar | 1 | apps.open |
<!-- eval_router_offline:end -->
