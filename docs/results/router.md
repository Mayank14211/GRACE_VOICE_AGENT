# M6 results: intent router and multi-turn conversation

Gate (docs/PLAN.md section 8): **no crash on any provider failure; 20 free-form requests logged**
with provider, latency and success, at least 5 of them with the network off (local model only).

Measured with `scripts/eval_router.py`. It sends 20 free-form requests in order through the voice
agent: grammar, then follow-ups, then the router, with one dialogue so the memory cases are
multi-turn. OS actions go to fakes. None of the requests is a fixed command.
`scripts/bench_threads.py --llm` measures the local model. `--save` on either script replaces
its own section below.

## Findings so far (S07, 2026-11-26, base machine i5-12450H)

- **Gemini only, no Groq key, no llama-server.** Every request Gemini answered was handled
  correctly: 15/15, including the multi-turn "and how many people live there". Median 1453 ms.
  Request 16 took more than 8 s, so Gemini went into its 60 s cooldown. With no other provider,
  requests 17-20 got the spoken "I can't reach an assistant right now" reply at once. Nothing
  crashed, and fixed commands keep working.
- Gemini's free tier answered in **1.4-3.5 s** with outliers of 4-15 s. Under the 3 s default
  budget it trips about 1 request in 5. An earlier run at 3 s: request 2 timed out, then
  everything else hit the cooldown. Hence `router.gemini.timeout_s` (6-8 s suggested). Groq is
  meant to be the fast primary.
- Back-to-back requests (one every ~2 s) also stall the Gemini free tier. Real speech is slower;
  the eval uses `--pause 8`.
- Multi-step requests ("set the sound to thirty, and then open notepad") work through the
  `more` flag. Gemini 3 needs its `thought_signature` echoed back on the replayed call, and
  GRACE does this. `tests/test_router_live.py` passed against Gemini.
- **Not measured yet:** Groq (no key on this machine), and the local model (llama-server not
  installed). See the manual checklist.

## Manual acceptance (team, base machine)

1. Put `GROQ_API_KEY` in `.env`. Run `uv run python scripts/eval_router.py --pause 3 --save`.
   Expected: at least 18/20 succeed, provider `groq`, median under 1 s.
2. Install llama.cpp (`winget install ggml.llamacpp`). Run `scripts\run_llama_server.ps1`.
   The first run downloads 5.4 GB and checks its hash.
   Expected: "server is listening on http://127.0.0.1:8080".
3. With the server running: `uv run python -m grace doctor`.
   Expected: `llama-server PASS` and the GGUF `hash ok`.
4. **Network off** (Wi-Fi off): `uv run python scripts/eval_router.py --providers local --save`.
   Expected: every request answered by `local`, each under the 15 s budget, no cloud errors.
   This covers the "at least 5 with network off" criterion. Record how many of the 20 succeed.
5. `uv run python scripts/bench_threads.py --llm --threads 4,6,8 --save`.
   Expected: a table below and a thread recommendation. Set `inference.threads` to it.
6. **Live voice**, `uv run python -m grace`: say 20 free-form requests of your own, at least 5
   with the network off and `router.local.enabled: true`. Expected: every one gets an action
   or a spoken answer. The status chip shows the provider. `logs/metrics/session-*.jsonl` has
   one `router` row per request (provider, latency_ms, ok, intents; never the words).
7. Provider failure while running: start GRACE with Groq first, then turn Wi-Fi off mid-session
   and ask something. Expected: one short "can't reach" or a local answer, no crash, and "volume
   40" still works at once.
8. Risky: say "turn everything off for tonight". Expected: no shutdown. GRACE tells you to say
   "shut down the computer". With `router.allow_risky: true` it asks "Confirm shutdown?", and
   only "yes" or Ctrl+Alt+Y runs it.
9. Demo mode: `demo_mode: true`, llama-server running, start GRACE. Expected: the log shows
   "router warm-up on local took N ms", and the first request is no slower than later ones.

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
