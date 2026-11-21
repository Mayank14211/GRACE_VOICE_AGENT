# GRACE Voice Agent

GRACE (General-purpose Responsive Assistant for Computer Environment) is an AI-powered voice assistant that allows you to control your computer using voice commands. This prototype is designed to demonstrate how voice can replace traditional input methods like keyboard and mouse for human-computer interaction.

## Features

- **Speech-to-Text (STT)**: Convert your voice to text for hands-free typing
- **System Control**: Open applications, control windows, perform clicks, and more
- **Screen Reading**: Read and understand text on the screen
- **Code Analysis**: Analyze and debug code with AI assistance
- **Wake Word Detection**: Activate with customizable wake word (default: "listen")
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
- Internet connection (the legacy speech recognition uses Google web STT)
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

3. Check the environment (opens the mic and camera briefly; add `--no-hardware` to skip them):
```powershell
uv run python scripts/doctor.py
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
copy .env.example .env                    # API keys: GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY
copy settings.example.yaml settings.yaml  # hotkeys, thresholds, provider order, inference.threads
uv run python main.py --check-config      # shows which providers are enabled, then exits
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

Run the main application:

```bash
uv run python main.py
```

Run the standalone hand-gesture controller:

```bash
uv run python Hand_Gesture.py
```

You can customize the wake word (API keys are never passed on the command line; see
[Configuration](#configuration)):

```bash
uv run python main.py --wake-word "computer"
uv run python main.py --settings my-settings.yaml --env-file my.env
```

List available voices:
```bash
python main.py --list-voices
```

### Available Commands

After saying the wake word, you can use these commands. Commands must be the **whole** utterance:
"volume 40" works, "I said volume 40 yesterday" does nothing. Numbers can be spoken ("forty") or
digits, and "please" / "can you" in front are ignored. Anything that isn't a fixed command goes
to the Gemini assistant (see below) when `GEMINI_API_KEY` is set; otherwise GRACE says it didn't
understand.

**Follow-up commands:** say the wake word once, then keep going: "open notepad", "type Hello,
World!", "close this window". You can also say the command in the same breath as the wake word
("listen, open notepad"). GRACE keeps listening until you pause for `wakeword.follow_up_s` seconds
(default 10; `0` = wake word before every command), or until you say "stop listening" / "that's
all". It waits until it has finished speaking before it listens, so it doesn't hear itself.

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

### Gemini assistant (optional)

With `GEMINI_API_KEY` in `.env`, speech that isn't a fixed command goes to Gemini
(`gemini.model` in `settings.yaml`, default `gemini-3.5-flash-lite`):

| Ask | What happens | Gemini tier needed |
|---|---|---|
| "What's the capital of France?" / "And its population?" | Short spoken answer; remembers the conversation ("start over" forgets it) | Free |
| "Turn the volume down a bit and open youtube" | Tool calling: Gemini uses GRACE's actions (volume, apps, keys, typing...) | Free |
| "Search the web for today's weather in Delhi" | Google Search grounding; sources printed in the console | **Billing-enabled key** |
| "Generate an image of a cat in sunglasses" | Image saved to `Pictures\GRACE` and opened | **Billing-enabled key** |
| "Analyze the code" / "Debug this code" | Selected code is reviewed by Gemini | Free |

- Google gives **no free quota** for search grounding or image generation. On a free key GRACE
  says so instead of failing silently. Turn them off with `gemini.web_search: false` /
  `gemini.image_generation: false`.
- Gemini can **never** shut down, restart or sleep the computer. Those stay exact-phrase +
  confirmation only.
- Privacy: what you say to the assistant, and code you ask it to analyze, is sent to Google.
  Free-tier inputs may be used to improve Google's models. Action results sent back to Gemini
  are only the spoken message (e.g. "Copied."), never clipboard contents.

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

The code analysis feature uses Google's Gemini AI to analyze and debug code:

1. First, select the code you want to analyze with your mouse
2. Then use one of the following commands:

**Commands for Code Analysis:**
- "Analyze code" - Explains what your code does, its structure, and patterns
- "Debug code" - Identifies potential errors and suggests fixes

For both commands to work:
- `GEMINI_API_KEY` must be set in `.env`
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
- **"Gemini AI not available"**: Check `GEMINI_API_KEY` is set in `.env`
  (`uv run python main.py --check-config` shows whether the provider is enabled)
- **No analysis**: Ensure you've selected code before issuing the command
- **Network errors**: Check your internet connection

## Configuration

GRACE reads two files from the repository root. Both are git-ignored and both are optional.

| File | Holds | Template |
|---|---|---|
| `.env` | API keys only: `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` | `.env.example` |
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

1. **Speech Recognition Module**: Handles wake word detection and speech-to-text
2. **Text-to-Speech Module**: Provides voice output
3. **Command grammar** (`grace/intent/grammar.py`): turns a whole utterance into an Intent; no
   keyword-contains matching
4. **Action registry** (`grace/actions/`): the only code that touches the OS (volume via pycaw,
   brightness via screen-brightness-control, mouse/keyboard, apps, power). Risky actions run only
   through the confirmation gate (`grace/actions/confirmation.py`)
5. **Screen Reader Module**: Captures and analyzes screen content
6. **Main Controller** (`main.py`): wake word -> grammar -> registry

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

- OpenAI for Whisper API
- Google for Gemini AI API
- Tesseract OCR project
- All the open-source libraries used in this project 