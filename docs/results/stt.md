# M5 results: speech to text, wake word, latency

Gate (docs/PLAN.md section 8, S06): on the base machine (Intel i5-12450H), **median end of
speech → final text under 500 ms with Moonshine** (the team may revise the bar).

- **End of speech** = the moment Silero VAD decides the utterance is over (after
  `audio.end_of_speech_ms` = 500 ms of silence) or the push-to-talk key is released.
- **eos → final** = from that moment until the final text is ready. This is the gate number.
- **speech end → final** adds the silence GRACE waits for (≈ end_of_speech_ms): what the user
  feels. Lower `audio.end_of_speech_ms` (e.g. 350) to shorten it, at the risk of cutting long
  pauses inside a command.
- Live runs log the same numbers per utterance to `logs/metrics/*.jsonl` (`"event": "stt"`,
  fields `eos_to_final_ms`, `speech_end_to_final_ms`, `engine`, `audio_s`; no transcript text).

How to reproduce: `uv run python scripts/eval_stt.py --save` (real recordings in `tests/audio/`,
see tests/audio/README.md) and `uv run python scripts/bench_threads.py --save`. Run on mains
power, Windows power mode "Best performance".

## Team recordings (acceptance)

<!-- eval_stt:start -->
### eval_stt (tests/audio)

**Not run yet:** the team records 30–50 commands each with
`scripts/record_audio.py --phrases tests/audio/phrases.txt --prefix <name>`, then runs
`scripts/eval_stt.py --save` on the base machine. This section is replaced by the results.
<!-- eval_stt:end -->

## Synthetic voices (first measurement, before the team's recordings)

Windows SAPI voices (David, Zira) reading tests/audio/phrases.txt with low room noise, made by
`scripts/make_audio_fixtures.py --eval-set data/synthetic_eval`. Cleaner than a real voice on a
laptop mic: latency is representative, accuracy is optimistic.

<!-- eval_stt_synthetic:start -->
### eval_stt_synthetic (synthetic: SAPI David + Zira, 42 phrases each)

Run 2026-11-23 18:22 on 12th Gen Intel(R) Core(TM) i5-12450H (Windows 11 build 26300). Audio: synthetic: SAPI David + Zira, 42 phrases each, 84 recordings, speakers: sapi-david, sapi-zira. end_of_speech_ms 500, Moonshine small-streaming, faster-whisper base at 4 threads.

| engine | files | WER | intent acc. | exact | eos→final median ms | eos→final p90 ms | speech end→final median ms | errors |
|---|---|---|---|---|---|---|---|---|
| moonshine | 84 | 4.4% | 95.2% | 92.9% | 117 | 170 | 633 | 0 |

**Acceptance (M5):** Moonshine median end-of-speech → final text 117 ms vs bar 500 ms: **MET** (synthetic: SAPI David + Zira, 42 phrases each).

Worst moonshine transcripts:

- `sapi-david_025.wav`: "copy" → "Coffee"
- `sapi-zira_024.wav`: "press control plus c" → "Press Ctrl + C"
- `sapi-david_003.wav`: "go to github dot com" → "Go to getthub.com"
- `sapi-zira_003.wav`: "go to github dot com" → "Go to Gethub.com."
- `sapi-david_024.wav`: "press control plus c" → "Press control + c"
<!-- eval_stt_synthetic:end -->

<!-- eval_stt_synthetic_whisper:start -->
### eval_stt_synthetic_whisper (synthetic: SAPI David + Zira, 42 phrases each)

Run 2026-11-23 18:27 on 12th Gen Intel(R) Core(TM) i5-12450H (Windows 11 build 26300). Audio: synthetic: SAPI David + Zira, 42 phrases each, 84 recordings, speakers: sapi-david, sapi-zira. end_of_speech_ms 500, Moonshine small-streaming, faster-whisper base at 4 threads.

| engine | files | WER | intent acc. | exact | eos→final median ms | eos→final p90 ms | speech end→final median ms | errors |
|---|---|---|---|---|---|---|---|---|
| whisper_local | 84 | 4.4% | 92.9% | 92.9% | 1279 | 1831 | 1801 | 0 |

Worst whisper_local transcripts:

- `sapi-david_009.wav`: "unmute" → "On me, right?"
- `sapi-zira_008.wav`: "mute" → "Mude."
- `sapi-david_003.wav`: "go to github dot com" → "Go to gethub.com"
- `sapi-david_024.wav`: "press control plus c" → "Press Ctrl plus C."
- `sapi-zira_024.wav`: "press control plus c" → "Press Ctrl plus C."
<!-- eval_stt_synthetic_whisper:end -->

## Thread count

<!-- bench_threads:start -->
### bench_threads

Run 2026-11-23 18:30 on 12th Gen Intel(R) Core(TM) i5-12450H (Windows 11 build 26300). 20 recordings from `data/synthetic_eval`, fed as fast as possible (no pacing); latency = end of audio → final text (for faster-whisper this is the whole transcription), RTF = compute time / audio time.

| engine | threads | median ms | RTF | load ms |
|---|---|---|---|---|
| faster-whisper base int8 | 4 | 1980 | 1.516 | 1781 |
| faster-whisper base int8 | 6 | 1834 | 1.422 | 1289 |
| faster-whisper base int8 | 8 | 1839 | 1.407 | 1373 |
| moonshine small-streaming | ORT default | 255 | 0.431 | 1192 |
| moonshine small-streaming | 1 | 373 | 0.683 | 889 |

**Recommendation:** `inference.threads: 6` on this machine (fewest threads within 5 % of the fastest faster-whisper run).
<!-- bench_threads:end -->

## Manual acceptance checklist (base machine)

| # | Check | Expected | Result |
|---|---|---|---|
| 1 | `uv run python -m grace doctor` | 0 FAIL; Silero VAD, Moonshine, model hashes PASS | |
| 2 | `uv run python -m grace`, say "listen, open notepad" | tone, Notepad opens, partial text then final text on the subtitle bar | |
| 3 | Then, without the wake word: "type hello world", "close this window" | both run (follow-up window) | |
| 4 | Wait 10 s, say "volume 30" | nothing happens (window closed; chip no longer shows LISTENING) | |
| 5 | Mic off (`Ctrl+Alt+M`), hold `Ctrl+Alt+Space`, say "volume 40", release | volume 40; capture ends on release, not on a pause | |
| 6 | "listen, start dictation", dictate two sentences with a pause, "stop dictation" | both sentences typed, separated by a space | |
| 7 | Start dictation, then press `Ctrl+Alt+M` | dictation ends, chip drops DICTATION, nothing more is typed | |
| 8 | While GRACE is answering out loud, stay silent | GRACE does not hear itself (no new command) | |
| 9 | Unplug / disable the mic while GRACE runs | "Microphone unavailable" on the overlay; hotkeys and gestures keep working | |
| 10 | `uv run pytest -m hardware -k m5_acceptance` after recording | passes (median < 500 ms) | |
| 11 | Wake-word model (after training, docs/wakeword.md): say it 20 times from 1 m | ≥ 19 detections; < 1 false wake in 10 min of normal talk | |
