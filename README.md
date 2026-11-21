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

5. Install Tesseract OCR:
   - **Windows**: Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt install tesseract-ocr`

6. Make sure to add Tesseract to your system PATH.

7. Verify Tesseract installation:
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

You can customize the wake word and API keys:

```bash
python main.py --wake-word "computer" --gemini-key "your-gemini-api-key"
```

List available voices:
```bash
python main.py --list-voices
```

### Available Commands

After saying the wake word, you can use these commands:

- **System Control**:
  - "Open [application]" - Opens specified application
  - "Open [website]" - Opens a website in your default browser
  - "Type [text]" - Types the specified text
  - "Click" / "Double click" / "Right click" - Performs mouse clicks
  - "Scroll up/down" - Scrolls the current window
  - "Copy" / "Paste" / "Select all" - Performs clipboard operations
  - "Undo" / "Redo" - Performs undo/redo operations
  - "Save" - Triggers save operation
  - "Close window/tab" - Closes current window or tab
  - "Screenshot" - Takes a screenshot
  - "Change brightness [1-100]" - Adjusts the screen brightness level
  - "Volume up/down" - Increases or decreases system volume
  - "Volume up/down [amount]" - Changes volume by specified amount
  - "Set volume [0-100]" - Sets the system volume to specified level
  - "Shut down" - Immediately shuts down your computer

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

To execute commands immediately:
1. Add "execute" at the end of a command: "open notepad execute"
2. Or start with "execute": "execute open notepad"

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
- A valid Gemini API key must be provided
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
- **"Gemini AI not available"**: Check your API key is valid and provided
- **No analysis**: Ensure you've selected code before issuing the command
- **Network errors**: Check your internet connection

## Architecture

GRACE Voice Agent is built with a modular architecture consisting of:

1. **Speech Recognition Module**: Handles wake word detection and speech-to-text
2. **Text-to-Speech Module**: Provides voice output
3. **System Control Module**: Interfaces with the operating system
4. **Screen Reader Module**: Captures and analyzes screen content
5. **Main Controller**: Orchestrates interactions between modules

## Extending GRACE

You can extend GRACE by:

1. Adding new commands in the `commands` dictionary in `GraceVoiceAgent` class
2. Creating command handlers with the naming convention `_handle_X_command`
3. Adding specialized modules for domain-specific tasks

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for Whisper API
- Google for Gemini AI API
- Tesseract OCR project
- All the open-source libraries used in this project 