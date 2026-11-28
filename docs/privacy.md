# Privacy: what GRACE sends, to whom, and what stays on this PC

GRACE is cloud-first (team decision 2026-11-26). Speech recognition and free-form questions use
free cloud APIs, while hand tracking, the wake-word model, voice detection, text-to-speech and the
fixed command grammar always run on this PC. This page lists, feature by feature, what leaves the
machine. GRACE shows a short version of it once, the first time voice starts (see "First-run
notice" below).

All API keys live in `.env` only. They are sent only in the `Authorization` header of requests to
their own provider, and they never appear in logs, metrics or error messages.

## Per feature

| Feature | Sent to | What is sent | When |
|---|---|---|---|
| Speech to text (default) | Groq (`whisper-large-v3-turbo`) | Audio of one utterance: the detected speech plus 200 ms on each side, as a 16 kHz WAV | After the wake word while the mic is on, or while push-to-talk is held. See "Upload rules" |
| Speech to text, `[stt-local]` extra (Moonshine) | Nobody | Nothing | Offline fallback, demo mode, and wake-word checks when installed |
| Speech to text, `[stt-whisper]` extra (faster-whisper) | Nobody | Nothing | Hindi / dictation fallback when installed |
| Wake word, trained model (openWakeWord) | Nobody | Nothing: it runs on this PC and never submits audio | While the mic is on |
| Wake word, spoken (no trained model yet) | Groq, **unless** a local engine is installed | Each detected utterance, so GRACE can check that it starts with the wake word ("listen ...") | While the mic is on. Install `[stt-local]` or train the model (docs/wakeword.md) to keep this on the PC |
| Fixed commands ("volume 30", "open notepad") | Nobody | Nothing: the grammar runs on this PC, no LLM is called | Always |
| Free-form requests and questions | Groq `openai/gpt-oss-20b`; on failure Gemini Flash-Lite, then an OpenRouter `:free` model | The transcript of the request, the recent conversation (last few turns) and the tool list | Only for speech the fixed grammar does not match |
| "Analyze this code" (screen or selected text) | Groq, then OpenRouter. **Never Gemini** | The text on the screen or the selection, and the request | Only when asked |
| Web search ("search the web for ...") | Tavily | Only the search query: never screen text or the conversation | Only when the LLM calls the `web_search` tool |
| Image generation ("draw a picture of ...") | Cloudflare Workers AI (FLUX.1 schnell) | Only the image prompt, after a local filter (refused prompts never leave the PC) | Only when the LLM calls `generate_image` |
| Hand gestures (camera) | Nobody | Nothing: MediaPipe runs on this PC; there is no preview window and no frames are stored | While gestures are on (Ctrl+Alt+G) |
| Text to speech | Nobody | Nothing (Windows SAPI) | Always |
| Metrics (`logs/metrics/*.jsonl`) | Nobody | Timings, engine names and counts. No transcript text, no typed text | When `runtime.metrics` is on |

Provider terms apply to what they receive. Gemini's free tier may use inputs to improve Google's
models, which is one reason screen text never goes to Gemini. Groq, OpenRouter, Tavily and
Cloudflare have their own policies on their websites.

## Upload rules for speech (grace/input/voice.py, tested in tests/test_voice_worker.py)

1. **Only detected speech.** Silero VAD (on this PC) decides where speech starts and ends.
   Pre-roll and trailing silence are cut to `stt.upload_padding_ms` (200 ms) on each side.
2. **Nothing short.** An utterance with less than `stt.upload_min_speech_ms` (300 ms) of speech
   (a click, a cough, a quick tap of the push-to-talk key) is never sent.
3. **Nothing while the mic is off.** The microphone stream is closed while the mic is off
   (Ctrl+Alt+M, or the on-screen button), so no audio exists to send. Turning the mic off or
   pressing panic (Ctrl+Alt+Q) mid-utterance cancels it before upload.
4. **Push-to-talk sends only while held.** Capture starts on key down and ends on key up. With
   the mic off, nothing from before the key press is included.
5. **The wake-word model never uploads.** openWakeWord runs locally. Speech only becomes an
   upload candidate after it fires. When there is no trained model yet, see the table above.

## Free-tier limits

A quota governor (`grace/core/quota.py`, counts in `logs/quota.json`) tracks requests, tokens and
audio seconds per provider and model, warns on the subtitle bar at 80 %, and stops using a
provider at 100 % until its window resets. The data it stores is counts only, never content.

## First-run notice

The first time voice starts with a cloud speech engine configured, GRACE shows a one-line notice
on the subtitle bar and in the console: what is sent to Groq, and that gestures, the camera and
the wake-word model stay on the PC. It never blocks gestures or hotkeys. GRACE remembers that it
was shown in `logs/privacy_notice.json`. Delete that file to see it again.

## Keeping everything on the PC

- `uv sync --extra stt-local` and `uv run python scripts/fetch_models.py --stt-local`: Moonshine
  then answers wake-word checks, and takes over whenever Groq is unavailable.
- Set `stt.engine: moonshine` in `settings.yaml` (English only) so speech is never uploaded. Leave
  `GROQ_API_KEY`, `GEMINI_API_KEY` and `OPENROUTER_API_KEY` out of `.env` so free-form requests
  stay local too (GRACE then handles fixed commands only).
- `demo_mode: true` prefers Moonshine but still uses Groq as its fallback.
