# GRACE Voice Agent — Project Context

> This document captures a full technical snapshot of the project as of November 2026, intended as a reference for further improvement and development.

---

## 1. Project Summary

**GRACE** (General-purpose Responsive Assistant for Computer Environment) is a Python-based voice-controlled desktop assistant for Windows (with partial macOS/Linux support). It converts spoken commands into computer actions: launching applications, controlling the mouse and keyboard, reading the screen via OCR, analyzing code with Gemini AI, generating images, and adjusting system settings (volume, brightness).

A second, standalone module (`Hand_Gesture.py`) provides camera-based hand gesture control as a parallel interaction channel — currently not integrated with the main agent.

---

## 2. Repository Structure

```
GRACE_VOICEAGENT/
├── main.py                   # Primary entry point — GraceVoiceAgent class (production)
├── voice_agent.py            # Original monolithic prototype — legacy, not used by main.py
├── tts.py                    # Text-to-Speech module (pyttsx3)
├── speech_recognition_module.py  # Speech recognition (Google STT + Whisper API)
├── screen_reader.py          # Screen capture, OCR (Tesseract), Gemini AI
├── system_control.py         # App launching, URL opening, keyboard/mouse control
├── Hand_Gesture.py           # Standalone hand-gesture controller (MediaPipe + pycaw)
├── tesseract_check.py        # Diagnostic utility to verify Tesseract installation
├── test_screen_reader.py     # Manual test script for screen reader
├── requirements.txt          # Python dependencies (incomplete — see §6)
├── Gen_images/               # Output directory for AI-generated images
├── tempCodeRunnerFile (1).py # IDE artifact — should be deleted
└── screenshot_*.png          # Sample screenshots committed by mistake
```

---

## 3. Module-by-Module Analysis

### 3.1 `main.py` — `GraceVoiceAgent`

The core orchestrator. All modules are instantiated here and all command handlers live here.

**Initialization flow:**
1. Instantiates `TextToSpeech`, `SpeechRecognizer`, `ScreenReader`, `SystemControl`
2. Sets wake word (default: `"listen"`)
3. Registers ~25 command handlers in `self.commands` dict
4. Creates `Gen_images/` output directory

**Execution flow:**
```
start()
  └─ _start_wake_word_detection()      # background thread via SpeechRecognizer
        └─ _on_wake_word_detected()    # callback on detection
              ├─ listen_for_command()  # second microphone session
              └─ _process_command()   # string-match dispatch
```

**Command dispatch (`_process_command`):**
1. Check exact match in `self.commands`
2. Check if command _starts with_ a command key (prefix match)
3. Check if command _contains_ a command key (substring match)

The "execute" modifier strips itself and re-dispatches the remainder — handled both as a prefix (`"execute open chrome"`) and as a suffix (`"open chrome execute"`).

**Key handlers and their mechanics:**

| Handler | Mechanism |
|---|---|
| `_handle_open_command` | `SystemControl.open_application()` → fallback `open_url()` |
| `_handle_read_command` | `ScreenReader.read_screen_text()` or `get_selected_text()` |
| `_handle_analyze_command` | `ScreenReader.analyze_code()` via Gemini |
| `_handle_debug_command` | `ScreenReader.debug_code()` via Gemini |
| `_handle_generate_image_command` | Stability AI → HuggingFace fallback |
| `_handle_brightness_command` | WMI (Win) / osascript (mac) / sysfs (Linux) |
| `_handle_volume_up/down_command` | WMI get → keyboard key simulation fallback |
| `_handle_set_volume_command` | nircmd.exe → keyboard mute/unmute trick fallback |
| `_handle_shutdown_command` | `os.system('shutdown /s /t 10 /f')` on Windows |

---

### 3.2 `tts.py` — `TextToSpeech`

Wraps `pyttsx3`. Key design decisions:
- `speak()` is an alias for `speak_async()` (non-blocking by default)
- `speak_async()` runs on a daemon thread; interrupts any in-progress speech before starting
- Voice selection: prefers female voices; looks for "Zira" or "Samantha" by name; falls back to `voice_index` arg
- `wait_until_done()` joins the speaking thread

**Known limitation:** Calling `engine.runAndWait()` from multiple threads can raise `RuntimeError` — caught and silenced. No queue mechanism to serialise speech correctly.

---

### 3.3 `speech_recognition_module.py` — `SpeechRecognizer`

Supports three backends: `"google"` (default), `"whisper_api"` (OpenAI), `"sphinx"` (offline).

**Wake-word detection:**
- Runs in a background daemon thread (`_wake_word_listener`)
- Opens a new `sr.Microphone()` context on every iteration
- Uses `timeout=1`, `phrase_time_limit=3` for responsiveness
- When the wake word is found, fires `wake_word_callback()` on the **same** background thread

**Post-wake-word command listening:**
- `listen_for_command()` opens **another** microphone context with `phrase_time_limit=6`
- Because both calls share `self.recognizer`, there is a potential for state conflict if the callback fires again before the first command completes

**Post-processing fix:** Concatenated tokens like `"openexecute"` are split back to `"open execute"` for a fixed list of common commands.

---

### 3.4 `screen_reader.py` — `ScreenReader`

**Tesseract path auto-detection:** On Windows, checks four common install paths at construction time.

**Screen capture:** `PIL.ImageGrab.grab()` → fallback `pyautogui.screenshot()` → blank white image.

**OCR:** `pytesseract.image_to_string()` with retry using `--oem 3 --psm 6` config if first pass returns empty.

**Gemini integration:** Uses `gemini-pro` (deprecated) with structured prompts for `analyze_code()`, `debug_code()`, `summarize_text()`.

**Clipboard-based selection:** `get_selected_text()` saves clipboard state → Ctrl+C → reads clipboard → restores original.

**Temp file management:** Instance keeps a list `self.temp_files`; cleaned up in `__del__`. Risk: destructor may not run on interpreter shutdown.

---

### 3.5 `system_control.py` — `SystemControl`

**Application launching:** Hardcoded dict of ~13 apps → `subprocess.Popen(cmd, shell=True)` → fallback `os.system(f"start {cmd}")`.

**Website shortcuts:** Dict of 12 popular sites mapped to URLs; checked before app dict.

**URL opening:** Adds `http://` prefix if missing; uses `webbrowser.open()`.

**Typing:** `pyautogui.write()` for ASCII; clipboard paste fallback for non-ASCII.

**Keyboard shortcuts:** Dict of ~20 named shortcuts → `pyautogui.hotkey()`.

**Action history:** Rolling list (`self.action_history`) for debugging; accessible via `get_action_history(limit)`.

---

### 3.6 `Hand_Gesture.py` — `GestureController`

Standalone. Uses:
- **MediaPipe Hands** for 21-landmark hand tracking
- **pycaw** for direct Windows audio COM interface
- **screen-brightness-control** for brightness
- **win32gui/win32con** to keep the preview window on top

**Gesture-to-action mapping (max 3 fingers):**

| Fingers (Thumb, Idx, Mid, Ring, Pink) | Gesture | Action |
|---|---|---|
| `[0,1,0,0,0]` | MOVE | Move cursor |
| `[0,1,1,0,0]` | SCROLL_READY | Scroll up/down |
| `[1,1,0,0,0]` | LEFT_CLICK | Left click |
| `[1,0,1,0,0]` | RIGHT_CLICK | Right click |
| `[1,1,1,0,0]` | VOLUME | Adjust volume (hand up/down) |
| `[0,1,1,1,0]` | BRIGHTNESS | Adjust brightness (hand up/down) |

**Not connected to `GraceVoiceAgent` in any way.**

---

### 3.7 `voice_agent.py` — `VoiceAgent` (legacy)

The original prototype, written before the modular architecture was introduced. Has its own TTS engine, speech recognizer, and command dispatcher all in one class. Still contains a **hardcoded Gemini API key**. It is not imported or invoked by `main.py` and can be archived or removed.

---

## 4. API Keys & Security Issues (Critical)

API keys are **hardcoded as default values** in `main.py` and also appear directly in `voice_agent.py` and `screen_reader.py`:

| Key | File | Location |
|---|---|---|
| Gemini API key | `main.py` | Line 1219 (argparse default) |
| Stability AI key | `main.py` | Lines 60, 1223 |
| HuggingFace token | `main.py` | Lines 61, 1225 |
| Gemini API key | `voice_agent.py` | Line 36 |
| Gemini API key | `screen_reader.py` | Line 324 (example block) |

These keys are committed to git and visible in the repository history. All affected keys should be rotated immediately and moved to a `.env` file (loaded via `python-dotenv`) or OS-level environment variables.

---

## 5. Known Bugs & Defects

### 5.1 Command Dispatch Ambiguity
Substring matching in `_process_command` (`main.py:261`) is order-dependent and can fire the wrong handler. For example, "shut" matches the shutdown handler before "set volume" can be evaluated if the user says "shut the volume down". The commands dict iteration order (Python 3.7+ dict is insertion-ordered) determines which handler fires first.

### 5.2 Wake-Word + Command Race Condition
`_on_wake_word_detected` (`main.py:147`) is invoked on the background listener thread. It then calls `self.recognizer.listen_for_command()` which opens a new microphone session. If the wake-word listener restarts before the command capture finishes, two microphone captures compete. No mutex protects `self.recognizer`.

### 5.3 TTS Thread Interruption
`speak_async` joins the previous thread with `timeout=0.5` then starts a new one. If the engine is mid-sentence and cannot be stopped in 0.5s, the new speech may overlap or the `runAndWait()` in the old thread may raise a `RuntimeError` on the shared engine.

### 5.4 Deprecated Gemini Model
`"gemini-pro"` is deprecated (used in `screen_reader.py:51` and `voice_agent.py:37`). Calls may fail or be silently redirected. Should be `"gemini-1.5-flash"` or `"gemini-2.0-flash"`.

### 5.5 Windows Volume Set Fallback
`_set_windows_volume` (`main.py:860`) tries `nircmd.exe` (not bundled), then attempts a mute→unmute→keystroke sequence that cannot reliably land on a specific level. Volume up/down (`_adjust_windows_volume`) reads volume via a broken WMI class name (`MSFT_MicrosoftGraphAudioPolicy` does not exist) and silently falls back to key presses with `~2%` per press assumption.

### 5.6 `type_text` Double-Types Non-ASCII
`system_control.py:122` calls `pyautogui.write(text)` for all text, then also pastes via clipboard if `not text.isascii()`. For non-ASCII strings, the character is typed once by `write()` and again by the clipboard paste, resulting in duplicated output.

### 5.7 Temp File Leak in `ScreenReader`
`screen_reader.py:172` removes the path from `self.temp_files` after `os.unlink()`, but only if the file still exists. If `save_screenshot` fails, the path may never be added. The `__del__` cleanup is unreliable at interpreter shutdown.

### 5.8 `tempCodeRunnerFile (1).py` in Repo
An IDE artifact with no content is committed to the repository.

---

## 6. Dependencies Gap

`requirements.txt` lists 12 packages. The following are actively imported in the codebase but **missing from requirements.txt**:

| Package | Used in |
|---|---|
| `keyboard` | `main.py` (volume key simulation) |
| `requests` | `main.py` (image generation API calls) |
| `google-generativeai` | `screen_reader.py`, `voice_agent.py` |
| `screen-brightness-control` | `Hand_Gesture.py` |
| `mediapipe` | `Hand_Gesture.py` |
| `opencv-python` | `Hand_Gesture.py` |
| `pycaw` | `Hand_Gesture.py` |
| `pywin32` | `Hand_Gesture.py` (win32gui, win32con) |
| `python-dotenv` | Not yet used — needed for env-based key management |

Also, `Pillow==9.5.0` is listed but the codebase uses features available in Pillow 10+; version pinning should be reviewed.

---

## 7. Platform Support Matrix

| Feature | Windows | macOS | Linux |
|---|---|---|---|
| Core voice commands | ✓ | ✓ | ✓ |
| App launching | ✓ Full | ✓ Partial | ✓ Partial |
| Brightness control | ✓ WMI | ✓ osascript | ✓ sysfs |
| Volume control | ✓ Keyboard | ✓ osascript | ✓ amixer/pactl |
| Hand gesture control | ✓ | ✗ Not tested | ✗ Not tested |
| OCR screen reading | ✓ | ✓ | ✓ |
| Gemini code analysis | ✓ | ✓ | ✓ |
| Image generation | ✓ | ✓ | ✓ |
| Shutdown command | ✓ | ✓ | ✓ |

---

## 8. Improvement Roadmap

### 8.1 Security (High Priority)
- Move all API keys to a `.env` file with `python-dotenv`; never hardcode keys
- Add `.env` to `.gitignore`
- Rotate all currently exposed keys

### 8.2 Architecture
- **Integrate `Hand_Gesture.py`**: Add a `gesture` mode to `GraceVoiceAgent` that launches the gesture controller in a background thread and maps gestures to existing command handlers
- **Remove `voice_agent.py`**: It is fully superseded by the modular design; archive it or delete
- **Config file**: Add `config.yaml` or `.env` for wake word, voice index, speech rate, API keys

### 8.3 Command Parsing
- Replace string prefix/substring matching with an intent classifier (even a simple keyword → intent dict with priority weights) to eliminate ambiguity
- Consider integrating a lightweight NLP library (e.g., `rapidfuzz` for fuzzy matching, or a small intent model)

### 8.4 Speech Pipeline
- **Local wake-word detection**: Replace Google STT-based wake-word with [Porcupine](https://picovoice.ai/platform/porcupine/) or [Vosk](https://alphacephei.com/vosk/) to work offline and reduce latency
- **Fix microphone contention**: Use a single shared microphone session with a state machine rather than opening/closing microphone contexts on every listen
- **TTS queue**: Replace the single speaking thread with a queue-based approach to prevent overlap

### 8.5 AI Upgrades
- Update Gemini model to `gemini-1.5-flash` or `gemini-2.0-flash-exp` in `screen_reader.py`
- Add a **conversational context** mode: allow multi-turn dialogue with the AI instead of one-shot commands
- Add **screen understanding** using Gemini Vision (`gemini-1.5-flash` can take screenshots as input), replacing Tesseract for complex layouts

### 8.6 Volume & Brightness
- Use `pycaw` (already in `Hand_Gesture.py`) for precise Windows volume control in the main agent — eliminates the broken WMI + keyboard hack
- Use `screen-brightness-control` (already in `Hand_Gesture.py`) for brightness — eliminates WMI

### 8.7 Code Quality
- Fix `type_text` double-typing bug for non-ASCII
- Fix `requirements.txt` to include all dependencies
- Add logging (`logging` module) instead of bare `print` statements
- Delete `tempCodeRunnerFile (1).py` and committed screenshots
- Add a proper test suite (pytest) with mocked audio and system calls

### 8.8 UX / Accessibility
- **System tray icon**: Run GRACE in the background with a tray icon (using `pystray`) showing current mode
- **Visual feedback overlay**: Semi-transparent on-screen notification when a command is recognized
- **Command confirmation mode**: Optional mode that reads back the interpreted command before executing it
- **Custom wake word training**: Support multiple wake words (e.g., "GRACE", "Hey computer")

---

## 9. Data Flow Diagram

```
Microphone
    │
    ▼
SpeechRecognizer._wake_word_listener()      [background thread]
    │  wake_word detected
    ▼
GraceVoiceAgent._on_wake_word_detected()
    │
    ├─► listen_for_command()               [new microphone session]
    │       │
    │       ▼
    │   raw command string
    │
    ▼
_process_command(command)
    │
    ├─ exact match in self.commands → handler(args)
    ├─ prefix match              → handler(remaining)
    └─ substring match           → handler(remainder)
              │
              ▼
         Handler logic
              │
    ┌─────────┼──────────────────┐
    ▼         ▼                  ▼
SystemControl  ScreenReader   External APIs
(pyautogui,   (pytesseract,   (Gemini, Stability AI,
subprocess,    PIL, Gemini)    HuggingFace)
webbrowser)
              │
              ▼
         TextToSpeech.speak_async()        [daemon thread]
              │
              ▼
         pyttsx3 → speakers
```

---

## 10. Quick-Start Reference

```bash
# Default startup (wake word: "listen", uses hardcoded keys)
python main.py

# Custom wake word + explicit API keys (recommended approach until .env is set up)
python main.py --wake-word "grace" --gemini-key "YOUR_KEY" --stability-key "YOUR_KEY"

# List available TTS voices
python main.py --list-voices

# Select voice by index and speech rate
python main.py --voice-index 1 --speech-rate 150

# Run hand gesture controller (standalone)
python Hand_Gesture.py

# Verify Tesseract OCR installation
python tesseract_check.py
```

---

*Generated from full codebase analysis — `main.py`, `tts.py`, `speech_recognition_module.py`, `screen_reader.py`, `system_control.py`, `Hand_Gesture.py`, `voice_agent.py`, `tesseract_check.py`, `requirements.txt`, `README.md`.*
