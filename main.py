#!/usr/bin/env python3
"""
GRACE Voice Agent - legacy voice loop (replaced by `python -m grace` in S05).

M2: every command goes through grace.intent.grammar (whole-phrase grammar) and the action
registry (grace.actions). Risky actions (shutdown, restart, sleep) need the exact phrase AND a
"yes" / Ctrl+Alt+Y confirmation within 8 seconds.
"""

import argparse
import re
import sys
import threading
import time
from pathlib import Path

from grace.actions import create_registry
from grace.actions.confirmation import ConfirmationGate, Dispatcher
from grace.config import ConfigError, format_provider_report, load_settings, report_providers
from grace.intent.grammar import Grammar, default_grammar
from grace.intent.types import Intent
from grace.redaction import configure_logging, install_redaction

_EXECUTE_RE = re.compile(r"\s*(?:execute\b[\s,]*)?(?P<cmd>.*?)(?:[\s,]*\bexecute)?[\s.!]*", re.I | re.S)


def strip_execute_word(command: str) -> str:
    """Old wake-word habit: "execute open notepad" / "open notepad execute" -> "open notepad".

    Only a whole leading or trailing word "execute" is removed; nothing is matched inside.
    """
    m = _EXECUTE_RE.fullmatch(command or "")
    return m.group("cmd").strip() if m else (command or "").strip()


HELP_TEXT = (
    "Here are some things you can say:\n"
    "- Open notepad / open youtube / go to github dot com\n"
    "- Volume 40 / set the volume to 70 percent / volume up / a bit louder / mute / unmute\n"
    "- Brightness 60 / brightness down a lot / dim the screen / what is the brightness\n"
    "- Click, double click, right click / scroll down / scroll up a lot / scroll down 5\n"
    "- Type <text> / dictate ... stop dictation\n"
    "- Press enter / press control plus c / copy / paste / cut / undo / redo / save / select all\n"
    "- Close this window / close tab / minimize / maximize / switch window / show desktop\n"
    "- Take a screenshot / read the screen / read the selection / analyze code / debug code\n"
    "- Shut down the computer / restart the computer / put the computer to sleep\n"
    "  (these ask for confirmation: say yes or press Ctrl+Alt+Y within 8 seconds)\n"
    "- Exit / quit / help"
)


class GraceVoiceAgent:
    def __init__(self, gemini_api_key=None, openai_api_key=None, wake_word="jarvis", *,
                 confirm_hotkey="ctrl+alt+y", registry=None, grammar: Grammar | None = None,
                 tts=None, recognizer=None, screen_reader=None):
        """Initialize the voice agent.

        Hardware-backed parts (tts, recognizer, screen_reader, registry) can be injected; tests
        pass fakes. Anything not injected is created for real.
        """
        print("Initializing GRACE Voice Agent...")

        if tts is None:
            from tts import TextToSpeech
            tts = TextToSpeech()
        if recognizer is None:
            from speech_recognition_module import SpeechRecognizer
            recognizer = SpeechRecognizer(api="google", openai_api_key=openai_api_key)
        if screen_reader is None:
            from screen_reader import ScreenReader
            screen_reader = ScreenReader(gemini_api_key=gemini_api_key)
        self.tts = tts
        self.recognizer = recognizer
        self.screen_reader = screen_reader

        self.grammar = grammar or default_grammar()
        self.registry = registry if registry is not None else create_registry()
        self.confirm_hotkey = confirm_hotkey
        self.gate = ConfirmationGate(self.registry, say=self._say, confirm_hotkey=confirm_hotkey)
        self.dispatcher = Dispatcher(self.registry, self.gate, say=self._say)
        self._hotkey_handle = None

        self.wake_word = wake_word.lower()
        self.is_running = False
        self.mode = "command"  # "command" or "dictation"
        self.command_history = []

        # App-level intents (they don't touch the OS themselves); everything else -> dispatcher.
        self._app_handlers = {
            "app.help": self._handle_help_command,
            "app.exit": self._handle_exit_command,
            "app.dictation_start": self._handle_dictate_command,
            "app.dictation_stop": self._handle_stop_dictation,
            "app.read_screen": self._read_screen,
            "app.read_selection": self._read_selection,
            "app.analyze_code": lambda: self._analyze(selection=True),
            "app.analyze_screen": lambda: self._analyze(selection=False),
            "app.debug_code": self._debug_code,
        }

    # -- feedback (console print until the overlay lands in S05) --------------------------------
    def _say(self, message):
        print(f"GRACE: {message}")
        self.tts.speak_async(message)

    # -- lifecycle ------------------------------------------------------------------------------
    def start(self):
        """Start the voice agent"""
        self.is_running = True

        print(f"Starting GRACE Voice Agent (Wake word: '{self.wake_word}')")
        self._register_confirm_hotkey()
        self.tts.speak_async(f"GRACE Voice Agent activated. Say {self.wake_word} to control your computer.")

        try:
            self._start_wake_word_detection()
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nGRACE Voice Agent interrupted by user")
        finally:
            self._cleanup()

    def _register_confirm_hotkey(self):
        try:
            import keyboard
            # Run the confirmed action off the keyboard hook thread.
            self._hotkey_handle = keyboard.add_hotkey(
                self.confirm_hotkey,
                lambda: threading.Thread(target=self.gate.confirm, daemon=True).start(),
            )
        except Exception as e:  # keyboard missing / hook refused: voice "yes" still works
            print(f"Confirm hotkey {self.confirm_hotkey} unavailable: {e}")

    def _cleanup(self):
        """Clean up resources"""
        self.is_running = False
        self.gate.cancel(announce=False)
        if self._hotkey_handle is not None:
            try:
                import keyboard
                keyboard.remove_hotkey(self._hotkey_handle)
            except Exception:
                pass
        self.recognizer.stop_wake_word_detection()
        print("GRACE Voice Agent stopped")

    def _start_wake_word_detection(self):
        self.recognizer.start_wake_word_detection(
            wake_word=self.wake_word,
            callback=self._on_wake_word_detected
        )

    def _on_wake_word_detected(self):
        """Called when the wake word is detected: listen for one command and run it."""
        print(f"Wake word '{self.wake_word}' detected!")
        if self.mode == "dictation":
            self._listen_and_process_command()
            return
        command = self.recognizer.listen_for_command(phrase_time_limit=6)
        print(f"Heard after wake word: '{command}'")
        if not strip_execute_word(command or ""):
            self.tts.speak_async("How can I help you?")
            self._listen_and_process_command()
            return
        self._process_command(command)

    def _listen_and_process_command(self):
        if self.mode == "dictation":
            text = self.recognizer.listen_for_command(phrase_time_limit=10)
            print(f"Dictation heard: '{text}'")
            if text:
                self._process_dictation(text)
        else:
            command = self.recognizer.listen_for_command(phrase_time_limit=6)
            print(f"Command heard: '{command}'")
            if command:
                self._process_command(command)

    # -- command routing ------------------------------------------------------------------------
    def _process_command(self, command):
        """Grammar -> Intent -> app handler or dispatcher (registry + confirmation gate)."""
        print(f"Processing command: '{command}'")
        self.command_history.append(command)
        intent = self.grammar.parse(strip_execute_word(command))
        if intent is None:
            # The LLM router (S07) will take unmatched input; until then, say so.
            self._say("I'm not sure how to help with that.")
            return None
        return self._handle_intent(intent)

    def _handle_intent(self, intent: Intent):
        print(f"Intent: {intent.name} {intent.args}")
        handler = self._app_handlers.get(intent.name)
        if handler is not None:
            if self.gate.pending is not None:
                self.gate.cancel()
            handler()
            return None
        return self.dispatcher.handle(intent)

    def _process_dictation(self, text):
        """Dictation mode: type everything until the whole phrase "stop dictation"."""
        intent = self.grammar.parse(text)
        if intent is not None and intent.name == "app.dictation_stop":
            self._handle_stop_dictation()
            return
        result = self.registry.execute("input.type", {"text": text})
        if not result.ok:
            self._say(result.message)

    # -- app-level handlers ---------------------------------------------------------------------
    def _handle_dictate_command(self):
        self.mode = "dictation"
        self._say("Starting dictation mode. Say 'stop dictation' when you're done.")

    def _handle_stop_dictation(self):
        self.mode = "command"
        self._say("Dictation mode stopped")

    def _selected_text(self):
        result = self.registry.execute("clipboard.copy")
        return result.data.get("text", "") if result.ok else ""

    def _read_screen(self):
        try:
            self._say("Reading screen content. This may take a moment...")
            text = self.screen_reader.read_screen_text()
            if text and text.strip():
                limited_text = text[:1000] + ("..." if len(text) > 1000 else "")
                self.tts.speak_async(limited_text)
            else:
                self._say("I couldn't detect any text on the screen. Make sure Tesseract OCR is installed properly.")
        except Exception as e:
            print(f"Error in read command: {e}")
            self._say("There was an error processing the read command.")

    def _read_selection(self):
        text = self._selected_text()
        if text.strip():
            self.tts.speak_async(text[:1000] + ("..." if len(text) > 1000 else ""))
        else:
            self._say("No text is selected. Please select some text first.")

    def _analyze(self, selection):
        try:
            if selection:
                code = self._selected_text()
                if not code.strip():
                    self._say("No code is selected for analysis. Please select some code first.")
                    return
                self._say("Analyzing selected code. This may take a moment...")
            else:
                self._say("Analyzing screen content. This may take a moment...")
                code = self.screen_reader.read_screen_text()
                if not (code and code.strip()):
                    self._say("I couldn't detect any content to analyze on the screen.")
                    return
            analysis = self.screen_reader.analyze_code(code)
            print(f"Analysis result: {analysis[:100]}...")
            self.tts.speak_async(analysis)
        except Exception as e:
            print(f"Error in analyze command: {e}")
            self._say("There was an error processing the analyze command.")

    def _debug_code(self):
        try:
            code = self._selected_text()
            if not code.strip():
                self._say("No code is selected for debugging. Please select some code first.")
                return
            self._say("Debugging selected code. This may take a moment...")
            debug_result = self.screen_reader.debug_code(code)
            print(f"Debug analysis result: {debug_result[:100]}...")
            self.tts.speak_async(debug_result)
        except Exception as e:
            print(f"Error in debug command: {e}")
            self._say("There was an error processing the debug command.")

    def _handle_exit_command(self):
        self.tts.speak_async("Shutting down GRACE Voice Agent. Goodbye!")
        time.sleep(1.5)  # let the goodbye play before the loop exits
        self.is_running = False

    def _handle_help_command(self):
        print(HELP_TEXT)
        self.tts.speak(HELP_TEXT)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='GRACE Voice Agent. API keys are read from .env only (see .env.example).')
    parser.add_argument('--settings', type=Path, default=None,
                        help='settings file (default: settings.yaml in the repo root)')
    parser.add_argument('--env-file', type=Path, default=None,
                        help='.env file with API keys (default: .env in the repo root)')
    parser.add_argument('--check-config', action='store_true',
                        help='load settings and .env, report providers, then exit')
    parser.add_argument('--wake-word', type=str, default="listen",
                        help='Wake word to activate the agent')
    parser.add_argument('--voice-index', type=int, default=None,
                        help='Index of voice to use (run main.py --list-voices to see options)')
    parser.add_argument('--speech-rate', type=int, default=170,
                        help='Speech rate (words per minute)')
    parser.add_argument('--list-voices', action='store_true',
                        help='List available voices and exit')

    args = parser.parse_args()

    configure_logging()
    try:
        settings = load_settings(args.settings, args.env_file)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(2)
    install_redaction(settings.secret_values())
    statuses = report_providers(settings)
    print("LLM providers:\n" + format_provider_report(statuses))
    if args.check_config:
        return

    if args.list_voices:
        from tts import TextToSpeech
        tts = TextToSpeech()
        voices = tts.list_available_voices()
        print(f"Found {len(voices)} voices:")
        for voice in voices:
            print(f"Voice {voice['index']}: {voice['name']} (Gender: {voice['gender'] or 'Unknown'})")
            tts.change_voice(voice['index'])
            tts.speak_async(f"This is the voice of {voice['name']}.")
            time.sleep(2.5)
        return

    agent = GraceVoiceAgent(
        gemini_api_key=settings.api_key("gemini") if "gemini" in settings.enabled_providers() else None,
        openai_api_key=None,  # legacy Whisper API path; replaced by Groq / local STT in M5
        wake_word=args.wake_word,
        confirm_hotkey=settings.hotkeys.confirm,
    )

    if args.voice_index is not None:
        agent.tts.change_voice(args.voice_index)
    if args.speech_rate != 170:
        agent.tts.change_rate(args.speech_rate)

    agent.start()

if __name__ == "__main__":
    main()
