# GRACE Voice Agent

GRACE (General-purpose Responsive Assistant for Computer Environment) is an AI-powered voice assistant that allows you to control your computer using voice commands. This prototype is designed to demonstrate how voice can replace traditional input methods like keyboard and mouse for human-computer interaction.

## Features

- **Speech-to-Text (STT)**: offline and streaming (Moonshine), with live partial text on screen
- **System Control**: Open applications, control windows, perform clicks, and more
- **Screen Reading**: Read and understand text on the screen
- **Code Analysis**: Analyze and debug code with AI assistance
- **Wake Word Detection**: a custom openWakeWord model (offline), or a spoken word until it exists
- **Dictation Mode**: Type continuously using voice

> Project plan and roadmap: [docs/PLAN.md](docs/PLAN.md) (source of truth).

> [!WARNING]
> **Leaked keys: revoke them.** Earlier commits of this repository contained a Google Gemini API
> key, a Stability AI key and a Hugging Face token in plain text (`main.py`, `screen_reader.py`,
> `test_screen_reader.py`, `voice_agent.py`). They are gone from the current code but **remain in
> git history**, so anyone with a clone can read them. Whoever owns those accounts MUST revoke
> them (Google AI Studio → API keys; Stability AI → account keys; Hugging Face → Settings → Access
> Tokens) and create new ones in `.env`. Removing a key from the code does not un-leak it.
>
> History has deliberately **not** been rewritten. Purging it (e.g. with
> [`git filter-repo`](https://github.com/newren/git-filter-repo) `--replace-text`) changes every
> commit hash and needs every collaborator to re-clone, so it is a team decision (see
> docs/PLAN.md §11). Revoking the keys is required either way; purging alone is not enough.

## Requirements

- Windows 10/11, 64-bit
- Python 3.11 (64-bit) — managed by [uv](https://docs.astral.sh/uv/)
- Microphone and webcam
- No internet needed for voice: wake word, English speech-to-text and fixed commands run offline
  (the assistant, its tools and the optional Groq STT need a network)
- Tesseract OCR (for screen reading)

## Installation

1. Install uv (once per machine):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Clone this repository and install the locked dependencies:
```powershell
git clone <repo-url> GRACE_VOICEAGENT
cd GRACE_VOICEAGENT
uv sync
```
`uv sync` downloads Python 3.11 if it is missing, creates `.venv/` and installs exactly what
`uv.lock` pins. `requirements.txt` is exported from the lock for tools that need it
(`uv export --format requirements-txt --no-dev --no-hashes --no-emit-project -o requirements.txt`);
do not edit it by hand.

3. Download the voice models (about 150 MB, into `models/`, hash-checked; once per machine):
```powershell
uv run python scripts/fetch_models.py               # Silero VAD, wake-word features, Moonshine
uv sync --extra stt-whisper                         # optional: Hindi / accurate dictation offline
uv run python scripts/fetch_models.py --whisper     # ...and its faster-whisper model
```

   Check the environment (opens the mic and camera briefly; add `--no-hardware` to skip them):
```powershell
uv run python -m grace doctor        # scripts/doctor.py still works too
```

4. Run the tests:
```powershell
uv run pytest -m "not hardware and not network"   # everyday / CI
uv run pytest -m hardware                          # on the base machine, mic + camera attached
uv run ruff check .
```

Adding a dependency: confirm a cp311 win_amd64 wheel exists, then
`uv add <pkg>==<version>`, `uv lock`, `uv pip check`, and re-export `requirements.txt`.

5. Configure (optional; everything has a default):
```powershell
copy .env.example .env                    # API keys: GROQ_API_KEY, GEMINI_API_KEY, ... (all optional)
copy settings.example.yaml settings.yaml  # hotkeys, thresholds, provider order, inference.threads
uv run python -m grace --check-config     # shows which providers are enabled, then exits
```
See [Configuration](#configuration) below.

6. Install Tesseract OCR:
   - **Windows**: Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt install tesseract-ocr`

7. Make sure to add Tesseract to your system PATH.

8. Verify Tesseract installation:
```bash
uv run python tesseract_check.py
```

## Usage

Run GRACE (voice and hand gestures together in one process, controlled by global hotkeys):

```bash
uv run python -m grace               # same as: uv run python main.py
uv run python -m grace --gestures    # start with gesture control on
uv run python -m grace --dry-run     # fake OS backends: nothing on the PC changes
uv run python -m grace --no-voice    # gestures + hotkeys only
```

| Hotkey | What it does |
|---|---|
| `Ctrl+Alt+M` | Microphone listening on / off |
| `Ctrl+Alt+G` | Gesture control on / off (the camera is only open while it is on) |
| `Ctrl+Alt+Space` (hold) | Push-to-talk: say one command without the wake word (works with the mic off) |
| `Ctrl+Alt+Q` | Panic stop: mic and gestures off, mouse buttons released; GRACE keeps running |
| `Ctrl+Alt+Y` | Confirm a pending risky action (shutdown, restart, sleep) |
| `Ctrl+Alt+H` | Show / hide the conversation history panel |
| `Ctrl+Alt+X` | Mark the last action as wrong (written to `logs/metrics/`) |

A round microphone button sits left of the status chip: click it to switch the mic on or off
(green = on, red slash = off, amber ring = listening for a command: after the wake word, or
while push-to-talk is held). It does the same as `Ctrl+Alt+M`
and never takes keyboard focus, so the app you are typing in stays in front. Turn it off with
`runtime.mic_button: false`.

Left of it a round camera button switches gesture control (and the camera) on or off, like
`Ctrl+Alt+G`; a red ring means the camera could not be opened. Turn it off with
`runtime.gesture_button: false`.

Every hotkey can be changed in the `hotkeys:` section of `settings.yaml`; start-up toggles, the
wake word and the overlay are in `runtime:`. A short tone confirms each toggle.

There is no camera window. The only UI is a transparent overlay that never takes focus and lets
clicks pass through: a subtitle bar at the bottom centre (what GRACE heard and answered; it fades
after 4 s) and a status chip at the bottom right (mic, gestures and the current gesture, e.g.
`Gesture: SCROLL`, plus the LLM provider). Ctrl+C in the console quits; so does saying "exit".

Try hand gestures on their own (console prints the recognised gesture):

```bash
uv run python -m grace.input.gestures --dry-run   # recognise only; the OS is not touched
uv run python -m grace.input.gestures             # live: mouse, scroll, volume, brightness
```

| Pose | Action |
|---|---|
| Index finger up | Move the cursor (the middle 70% of the camera view covers the whole screen) |
| Thumb–index pinch | Left click on release; hold > 0.4 s (or move while pinched) = drag |
| Thumb–middle pinch | Right click |
| Index + middle up | Scroll: move the hand up/down from where the pose started |
| Index + middle + ring up | Brightness, same way |
| Thumb + index + middle up | Volume, same way |
| Fist held 1 s | Pause / resume gestures |
| Open palm | Neutral (resets the starting point) |

Every pose and threshold is in the `gestures:` section of `settings.yaml`. Record landmark
fixtures for the tests with `uv run python scripts/record_gestures.py --hand right`.

### Voice

One microphone stream feeds everything. Silero VAD finds where each utterance starts and ends,
and the transcript streams onto the subtitle bar *while you speak* (in italics; the final text
replaces it). The mic is only open while it is on (`Ctrl+Alt+M`) or push-to-talk is held.

| How you start a command | When |
|---|---|
| Say the **wake word** (a short tone answers), then the command | default |
| Hold **`Ctrl+Alt+Space`**, speak, release | always, even with the mic off; no wake word |
| Just speak | `wakeword.enabled: false` (every utterance is a command while the mic is on) |

The wake word is the team's own openWakeWord model (`models/wakeword/grace.onnx`; train it
with [docs/wakeword.md](docs/wakeword.md)). Until that file exists, GRACE uses a spoken word
instead: start with `runtime.wake_word` ("listen, open notepad"). Change that word without
editing settings:

```bash
uv run python -m grace --wake-word "computer"
uv run python -m grace --settings my-settings.yaml --env-file my.env
```

Speech-to-text (`stt:` in `settings.yaml`): English uses **Moonshine Small Streaming** offline
(final text ~0.1 s after you stop). `language: hi` / `auto`, or dictation with
`stt.dictation_accurate: true`, use **Groq** `whisper-large-v3-turbo` when `GROQ_API_KEY` is set
and the network answers, otherwise **faster-whisper** (extra `stt-whisper`). If Groq fails, the
same utterance is transcribed locally instead and Groq is skipped for a minute.

Measure accuracy and latency on your own voice (results go to
[docs/results/stt.md](docs/results/stt.md)):

```bash
uv run python scripts/record_audio.py --phrases tests/audio/phrases.txt --prefix <name>
uv run python scripts/eval_stt.py --save       # WER, intent accuracy, end-of-speech latency
uv run python scripts/bench_threads.py --save  # fastest inference.threads (4 / 6 / 8)
```

List available voices:
```bash
uv run python -m grace --list-voices
```

### Available Commands

After saying the wake word, you can use these commands. Commands must be the **whole** utterance:
"volume 40" works, "I said volume 40 yesterday" does nothing. Numbers can be spoken ("forty") or
digits, and "please" / "can you" in front are ignored. Anything that isn't a fixed command or a
follow-up goes to the assistant (LLM router, see below) when a provider is configured; otherwise
GRACE says it didn't understand.

**Follow-up commands:** say the wake word once, then keep going: "open notepad", "type Hello,
World!", "close this window". You can also say the command in the same breath as the wake word
("listen, open notepad"). GRACE keeps listening until you pause for `wakeword.follow_up_s` seconds
(default 10; `0` = wake word before every command), or until you say "stop listening" / "that's
all". While GRACE is talking the mic is ignored, so it doesn't hear itself.

- **Volume and brightness** (exact values, 0-100; out-of-range values are clamped):
  - "Volume 40" / "Set the volume to 70 percent" / "Volume max"
  - "Volume up" / "Volume down 20" / "Turn the volume up a lot" / "A bit louder" / "Quieter"
  - "Mute" / "Unmute" / "What's the volume?"
  - "Brightness 60" / "Set brightness to 70 percent" / "Brightness down a lot" / "Dim the screen"
- **Mouse and keyboard**:
  - "Click" / "Double click" / "Right click"
  - "Scroll down" / "Scroll up a little" / "Scroll down a lot" / "Scroll down 5 times"
  - "Type [text]" - Types the text exactly as heard
  - "Press enter" / "Press control plus c" / "Press alt f4"
  - "Copy" / "Paste" / "Cut" / "Select all" / "Undo" / "Redo" / "Save" / "Find" / "Refresh"
  - "Minimize" / "Maximize" / "Switch window" / "Show desktop" / "New tab"
- **Apps and windows**:
  - "Open notepad" / "Launch chrome" / "Open youtube" / "Go to github dot com"
    (known apps and sites, or anything that looks like a web address)
  - "Close this window" / "Close tab"
  - "Take a screenshot" - Saved to `Pictures\GRACE`
- **Power (risky)**: only these exact phrases, and only after confirmation:
  - "Shut down the computer" / "Restart the computer" / "Put the computer to sleep"
  - GRACE asks "Confirm shutdown? Say yes or press Ctrl+Alt+Y." Say **"yes"** or press
    **Ctrl+Alt+Y** (configurable: `hotkeys.confirm` in `settings.yaml`) within 8 seconds.
    "No", any other command, or the timeout cancels.

The full vocabulary (synonyms, shortcuts, key names) is in `grace/intent/synonyms.yaml`.

### Follow-ups (offline)

Right after a volume, brightness or scroll command (within `router.followup_s`, default 2 min),
these refer to it. They are resolved without any LLM, so they work offline:

| Say | After "volume 40" | After "scroll down" |
|---|---|---|
| "A bit more" / "more" / "a lot more" | volume +5 / same step again / +30 | scroll down again |
| "A bit less" / "too much" | volume -5 | scroll up 1 |
| "Turn it up" / "down a bit" | volume +10 / -5 | scroll up / down |
| "Undo that" / "put it back" | volume back to what it was | scroll back |
| "Now the brightness" / "brightness too" | brightness 40 | - |
| "Make it 30" | volume 30 | - |
| "Again" / "do it again" | the last action once more (never a risky one) | |

Plain "undo" is always Ctrl+Z. "Undo that" is Ctrl+Z too when the last action can't be reversed.

### Assistant: the LLM router (optional, cloud)

Speech that is neither a fixed command nor a follow-up goes to the router
(`grace/intent/router.py`). It tries the providers in `router.provider_order`, skipping any
without a key, and turns the model's tool calls into GRACE actions. There is no local LLM: the
assistant needs network. Fixed commands, follow-ups, gestures and hotkeys never do.

| Provider (default order) | Model (`router.<provider>.model`) | Needs | Timeout |
|---|---|---|---|
| Groq | `openai/gpt-oss-20b` | `GROQ_API_KEY` (free) | 3 s |
| Gemini (OpenAI endpoint) | `gemini-3.5-flash-lite` | `GEMINI_API_KEY` (free) | 6 s |
| OpenRouter | `google/gemma-4-26b-a4b-it:free` (must be `:free`) | `OPENROUTER_API_KEY` | 3 s |

- **No crash on any failure.** HTTP 429, a timeout or any error puts that provider in a 60 s
  cooldown and the next one is tried. A provider whose free quota is used up is skipped too.
  If none answers, GRACE says so; fixed commands keep working.
- **Stateless providers, history in GRACE.** The router keeps the last `router.history_turns`
  (6) exchanges plus the last action and sends them on every call, so a fallback provider
  continues the same conversation: "What's the capital of Australia?" then "And how many
  people live there?". "Start over" forgets; after `router.memory_reset_s` (10 min) of silence
  it forgets by itself.
- **Validated output.** Every tool call is checked against the action's pydantic model. A bad
  reply gets one retry, then the next provider is tried; if all fail, GRACE asks you to rephrase.
- **Several steps**: "set the sound to thirty and then open notepad" runs both, in order.
- **Answers** are at most two spoken sentences.
- **Risky actions** (shut down, restart, sleep) are never offered to the LLM. It tells you the
  exact phrase to say ("shut down the computer"), which then asks for confirmation.
- **Quotas.** `router.quota` holds each provider's free-tier limits (requests and tokens per
  minute and per day). GRACE warns at 80 % and skips the provider at 100 % until the window
  resets. Counts survive restarts (`logs/quota.json`).
- The provider that answered last is shown on the status chip (`offline` when none did).
- Timeouts: 3 s for Groq and OpenRouter, 6 s for Gemini. Gemini's free tier answered in
  1.4-3.5 s on the base machine with a tail to 4-6 s, so at 3 s the fallback missed about 1
  request in 5.
- Groq's free tier allows 8,000 tokens a minute, and each request carries about 1.6k tokens of
  prompt and tools. So Groq takes about 5 requests a minute; past that the governor sends
  requests to Gemini.

**Tools run by GRACE's own code** (the model only asks for them):

| Ask | Tool | Needs (free) |
|---|---|---|
| "Search the web for today's weather in Delhi" | `web_search`: Tavily, answer spoken in two sentences, sources printed | `TAVILY_API_KEY` (1,000 searches/month) |
| "Draw me a picture of a lighthouse at sunset" | `generate_image`: Cloudflare Workers AI FLUX.1 schnell; saved to `Pictures\GRACE` and opened | `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN` (10,000 neurons/day, about 170 images) |
| "Analyze the code" / "Debug this code" | the router's chain **without Gemini** | `GROQ_API_KEY` or `OPENROUTER_API_KEY` |

- A tool without its key isn't offered to the model, and asking for it says which key is missing.
- Image prompts asking for sexual content, gore, self-harm or hate are refused on this machine
  and never sent.
- Privacy: what you say to the assistant goes to the provider that answers. Search queries go
  to Tavily and image prompts to Cloudflare. **Screen text and selected code never go to
  Gemini** (its free tier may use inputs for training). The router never sends action data back
  (clipboard contents, typed text, screen text): it sends only your words and the last action's
  name. Screen reading (OCR) stays on the machine.
- Measure it: `uv run python scripts/eval_router.py --force-429 5` runs 20 free-form requests,
  including 3 searches and 3 images. The first 5 get a forced HTTP 429 from Groq, so you can
  see the fallback keep the history. `--offline` simulates no network. `--save` writes
  `docs/results/router.md`.
- `uv run python -m grace doctor` checks each configured service with a call that spends no
  quota: the model list, Cloudflare token verify, and Tavily usage. `--no-network` skips these.

- **Text Input**:
  - "Dictate" - Starts dictation mode
  - "Stop dictation" - Ends dictation mode

- **Screen Reading**:
  - "Read screen" - Reads visible text on screen
  - "Read selection" - Reads text you've selected

- **Code Analysis**:
  - "Analyze code" - Analyzes selected code
  - "Debug code" - Debugs selected code

- **General**:
  - "Help" - Lists available commands
  - "Exit" / "Quit" - Exits the application

### Using Execute Commands

The old habit of adding "execute" still works: "open notepad execute" and "execute open notepad"
both run "open notepad". Only a whole leading or trailing word "execute" is removed.

## Screen Reading and Code Analysis Features

### Screen Reading

The screen reading feature uses Tesseract OCR to read text from your screen. For best results:

1. Make sure the text is clear and easily visible
2. Avoid complex backgrounds that might interfere with OCR
3. Use for reading error messages, dialog boxes, or other on-screen text

**Commands for Screen Reading:**
- "Read screen" - Captures the entire screen and reads the text
- "Read selection" - Reads text you've selected with the mouse

If screen reading doesn't work well, verify Tesseract is properly installed by running:
```bash
python test_screen_reader.py
```

### Code Analysis

The code analysis feature sends the selected code to the assistant (Groq, else OpenRouter;
never Gemini) to analyze and debug it:

1. First, select the code you want to analyze with your mouse
2. Then use one of the following commands:

**Commands for Code Analysis:**
- "Analyze code" - Explains what your code does, its structure, and patterns
- "Debug code" - Identifies potential errors and suggests fixes

For both commands to work:
- `GROQ_API_KEY` (or `OPENROUTER_API_KEY`) must be set in `.env`
- Text must be selected before issuing the command

Example:
1. Select some code in your editor
2. Say "listen analyze code"
3. The assistant will explain what your code does

## Troubleshooting

### Screen Reading Issues
- **No text detected**: Ensure Tesseract OCR is properly installed
- **Incorrect text**: Try selecting text manually instead of using OCR
- **Installation problems**: Run `tesseract_check.py` to diagnose issues

### Code Analysis Issues
- **"Code analysis needs a Groq or OpenRouter key"**: set `GROQ_API_KEY` in `.env`
  (`uv run python -m grace --check-config` shows whether the provider is enabled)
- **No analysis**: Ensure you've selected code before issuing the command
- **Network errors**: Check your internet connection

## Configuration

GRACE reads two files from the repository root. Both are git-ignored and both are optional.

| File | Holds | Template |
|---|---|---|
| `.env` | API keys only: `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `TAVILY_API_KEY` | `.env.example` |
| `settings.yaml` | Hotkeys, audio/gesture thresholds, wake word model path, language, `demo_mode`, `inference.threads`, LLM provider order and timeouts | `settings.example.yaml` (every option documented inline) |

- **Every key is optional.** Wake word, English STT, gestures and fixed commands work offline. A
  missing key disables that cloud provider; startup logs one line per disabled provider.
- **Keys go in `.env` only.** Never in code, `settings.yaml` (it is rejected at startup if it
  contains anything key-like), command-line flags or commit messages. Process environment
  variables with the same names override `.env`.
- **Invalid settings fail fast** with a readable message naming the bad option
  (e.g. `inference.threads: Input should be less than or equal to 8`). Unknown options are typos
  and are rejected.
- `inference.threads`: 4 on the Intel i5-12450H (P-cores), 6 on the Ryzen 5 5500U. Never 12.
- Logs pass through a redaction filter that replaces anything key-shaped (`AIza…`, `sk-…`, `hf_…`,
  `gsk_…`) and every configured key value with `[REDACTED]`.
- `tests/test_no_secrets.py` fails the suite if a key-shaped string appears anywhere in the
  working tree (outside `.venv/`, `.git/` and your local `.env`).

## Architecture

GRACE Voice Agent is built with a modular architecture consisting of:

1. **Voice input** (`grace/input/`): `audio.py` (one sounddevice stream, Silero VAD, utterance
   segmenter), `wakeword.py` (openWakeWord), `stt/` (Moonshine, faster-whisper, Groq behind one
   `transcribe_stream` interface) and `voice.py` (wake word, follow-ups, push-to-talk, dictation;
   voice, STT and intent threads). `grace/legacy_voice.py` turns each final transcript into
   commands and replies: grammar, then follow-ups (`grace/intent/dialogue.py`), then the LLM
   router (`grace/intent/router.py`, providers in `providers.py`, tool schema generated from the
   registry in `tools.py`)
2. **Text-to-Speech Module**: Provides voice output
3. **Command grammar** (`grace/intent/grammar.py`): turns a whole utterance into an Intent; no
   keyword-contains matching
4. **Action registry** (`grace/actions/`): the only code that touches the OS (volume via pycaw,
   brightness via screen-brightness-control, mouse/keyboard, apps, power). Risky actions run only
   through the confirmation gate (`grace/actions/confirmation.py`)
5. **Screen Reader Module**: Captures and analyzes screen content
6. **Runtime** (`grace/__main__.py`, `grace/runtime.py`): Qt owns the main thread (overlay,
   `grace/ui/overlay.py`); the voice, STT and intent threads (`grace/input/voice.py`), the
   gesture camera loop, the action executor and the keyboard hook
   (`grace/input/hotkeys.py`) each run on their own thread and talk through the event bus
   (`grace/core/bus.py`) and shared state (`grace/core/state.py`). Every OS action, from voice
   or gesture, runs on the one executor thread, in order.

## Extending GRACE

You can extend GRACE by:

1. Adding an action: register a handler with a pydantic args model in `grace/actions/`
   (`registry.register("name", ArgsModel, risky=...)`), with a fake backend for tests
2. Adding phrases: fixed phrases and synonyms go in `grace/intent/synonyms.yaml`; patterns with
   arguments go in `RULES` in `grace/intent/grammar.py`. Add positive AND negative cases to
   `tests/test_grammar.py`
3. Adding specialized modules for domain-specific tasks

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Moonshine Voice (Useful Sensors), openWakeWord, Silero VAD and faster-whisper
- Google for Gemini AI API
- Tesseract OCR project
- All the open-source libraries used in this project 