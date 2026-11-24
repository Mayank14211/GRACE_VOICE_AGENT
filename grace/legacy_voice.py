"""Voice agent: what GRACE does with a transcribed utterance (grammar, Gemini, TTS replies).

Moved here from main.py in S05. Since S06 it no longer listens: grace/input/voice.py owns the
microphone, wake word, follow-up window and push-to-talk, and calls ``handle_utterance`` on the
intent thread with each final transcript. S07 replaces the Gemini-only fallback with the router.

Every command goes through grace.intent.grammar (whole-phrase grammar) and the action registry.
In the runtime the registry is an ``ExecutorRegistry``, so voice actions run on the same executor
thread as gestures. Risky actions (shutdown, restart, sleep) need the exact phrase AND a "yes" /
Ctrl+Alt+Y confirmation within 8 seconds.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from grace.actions import create_registry
from grace.actions.confirmation import ConfirmationGate, Dispatcher
from grace.intent.grammar import Grammar, default_grammar
from grace.intent.types import Intent

# Punctuation is only dropped together with a trailing "execute": "type Hello!" keeps its "!".
_EXECUTE_RE = re.compile(
    r"\s*(?:execute\b[\s,]*)?(?P<cmd>.*?)(?:[\s,]*\bexecute[\s.!]*)?\s*", re.I | re.S
)


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
    "- Anything else goes to Gemini (needs GEMINI_API_KEY): questions, 'search the web for ...',\n"
    "  'generate an image of ...', or several steps at once ('mute and open youtube')\n"
    "- After the wake word (a short tone), keep talking: follow-up commands need no wake word\n"
    "  until you pause. 'Stop listening' ends it; 'start over' makes Gemini forget.\n"
    "- Hold Ctrl+Alt+Space to talk without the wake word. Ctrl+Alt+M mic, Ctrl+Alt+G gestures,\n"
    "  Ctrl+Alt+Q stops everything.\n"
    "- Exit / quit / help"
)

ANALYZE_INSTRUCTION = (
    "You review code for a developer who is listening, not reading. In under 80 words of plain "
    "spoken English (no markdown, no code blocks), say what the code does and the most important "
    "problem or improvement."
)
DEBUG_INSTRUCTION = (
    "You debug code for a developer who is listening, not reading. In under 80 words of plain "
    "spoken English (no markdown, no code blocks), name the most likely bug and how to fix it. "
    "If there is no obvious bug, say so."
)


def _noop(*_: Any) -> None:
    pass


class GraceVoiceAgent:
    def __init__(
        self,
        *,
        confirm_hotkey: str = "ctrl+alt+y",
        registry: Any = None,
        grammar: Grammar | None = None,
        tts: Any = None,
        screen_reader: Any = None,
        assistant: Any = None,
        wake_word: str = "listen",
        on_heard: Callable[[str], None] = _noop,
        on_reply: Callable[[str], None] = _noop,
        on_mode: Callable[[str], None] = _noop,
        on_end_session: Callable[[], None] = _noop,
        on_exit: Callable[[], None] | None = None,
    ) -> None:
        """Hardware-backed parts (tts, screen_reader, registry) can be injected; tests pass fakes.
        Anything not injected is created for real. ``assistant`` (GeminiAssistant) answers
        whatever the grammar doesn't match; None disables that. ``on_end_session`` is called for
        "stop listening" (the voice pipeline closes its follow-up window).
        """
        if tts is None:
            from tts import TextToSpeech

            tts = TextToSpeech()
        if screen_reader is None:
            from screen_reader import ScreenReader

            # OCR only: its old Gemini path (google-generativeai, 'gemini-pro') no longer works.
            screen_reader = ScreenReader(gemini_api_key=None)
        self.tts = tts
        self.screen_reader = screen_reader

        self.grammar = grammar or default_grammar()
        self.registry = registry if registry is not None else create_registry()
        self.confirm_hotkey = confirm_hotkey
        self.gate = ConfirmationGate(self.registry, say=self._say, confirm_hotkey=confirm_hotkey)
        self.dispatcher = Dispatcher(self.registry, self.gate, say=self._say)
        self.assistant = assistant
        self._on_heard = on_heard
        self._on_reply = on_reply
        self._on_mode = on_mode
        self._on_end_session = on_end_session
        self._on_exit = on_exit

        self.wake_word = wake_word.lower()
        self.is_running = True
        self.mode = "command"  # "command" or "dictation"
        self._dictated = False  # something was typed since dictation started
        self.command_history: list[str] = []

        # App-level intents (they don't touch the OS themselves); everything else -> dispatcher.
        self._app_handlers: dict[str, Callable[[], None]] = {
            "app.help": self._handle_help_command,
            "app.exit": self._handle_exit_command,
            "app.dictation_start": self._handle_dictate_command,
            "app.dictation_stop": self._handle_stop_dictation,
            "app.read_screen": self._read_screen,
            "app.read_selection": self._read_selection,
            "app.analyze_code": lambda: self._analyze(selection=True),
            "app.analyze_screen": lambda: self._analyze(selection=False),
            "app.debug_code": self._debug_code,
            "app.end_session": self._end_session,
            "app.reset_conversation": self._reset_conversation,
        }

    # -- feedback -------------------------------------------------------------------------------
    def _say(self, message: str) -> None:
        print(f"GRACE: {message}")
        self._on_reply(message)
        self.tts.speak_async(message)

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self._on_mode(mode)

    # -- input --------------------------------------------------------------------------------
    def handle_utterance(self, text: str) -> None:
        """One final transcript from the voice pipeline (intent thread)."""
        if not self.is_running or not strip_execute_word(text):
            return
        self._handle_text(text)

    def _handle_text(self, text: str) -> None:
        print(f"Heard: '{text}'")
        self._on_heard(text)
        if self.mode == "dictation":
            self._process_dictation(text)
        else:
            self._process_command(text)

    def _end_session(self) -> None:
        self._on_end_session()
        self._say("Okay. Say the wake word when you need me.")

    def _reset_conversation(self) -> None:
        if self.assistant is not None:
            self.assistant.reset()
        self._say("Okay, starting a new conversation.")

    # -- command routing ------------------------------------------------------------------------
    def _process_command(self, command: str) -> Any:
        """Grammar -> Intent -> app handler or dispatcher (registry + confirmation gate)."""
        print(f"Processing command: '{command}'")
        self.command_history.append(command)
        text = strip_execute_word(command)
        intent = self.grammar.parse(text)
        if intent is None:
            return self._ask_assistant(text)
        return self._handle_intent(intent)

    def _ask_assistant(self, text: str) -> Any:
        """Unmatched speech -> Gemini (answers, web search, images, tool calls)."""
        if self.gate.pending is not None:
            self.gate.cancel()  # anything else abandons a pending risky action
        if self.assistant is None:
            self._say("I'm not sure how to help with that.")
            return None
        reply = self.assistant.ask(text)
        for title, url in reply.sources:
            print(f"  source: {title} <{url}>")
        for path in reply.images:
            print(f"  image: {path}")
        self._say(reply.text)
        return reply

    def _handle_intent(self, intent: Intent) -> Any:
        print(f"Intent: {intent.name} {intent.args}")
        handler = self._app_handlers.get(intent.name)
        if handler is not None:
            if self.gate.pending is not None:
                self.gate.cancel()
            handler()
            return None
        return self.dispatcher.handle(intent)

    def _process_dictation(self, text: str) -> None:
        """Dictation mode: type each finished sentence until the whole phrase "stop dictation".

        Sentences after the first get a leading space so they don't run together.
        """
        intent = self.grammar.parse(text)
        if intent is not None and intent.name == "app.dictation_stop":
            self._handle_stop_dictation()
            return
        typed = f" {text}" if self._dictated else text
        result = self.registry.execute("input.type", {"text": typed})
        if result.ok:
            self._dictated = True
        else:
            self._say(result.message)

    # -- app-level handlers ---------------------------------------------------------------------
    def _handle_dictate_command(self) -> None:
        self._dictated = False
        self._set_mode("dictation")
        self._say("Starting dictation mode. Say 'stop dictation' when you're done.")

    def _handle_stop_dictation(self) -> None:
        self._set_mode("command")
        self._say("Dictation mode stopped")

    def cancel_dictation(self) -> None:
        """The mic was turned off (or panic) during dictation: leave dictation mode quietly."""
        if self.mode == "dictation":
            self._set_mode("command")
            print("Dictation stopped (mic off).")

    def _selected_text(self) -> str:
        result = self.registry.execute("clipboard.copy")
        return result.data.get("text", "") if result.ok else ""

    def _read_screen(self) -> None:
        try:
            self._say("Reading screen content. This may take a moment...")
            text = self.screen_reader.read_screen_text()
            if text and text.strip():
                self.tts.speak_async(text[:1000] + ("..." if len(text) > 1000 else ""))
            else:
                self._say(
                    "I couldn't detect any text on the screen. "
                    "Make sure Tesseract OCR is installed properly."
                )
        except Exception as e:  # noqa: BLE001
            print(f"Error in read command: {e}")
            self._say("There was an error processing the read command.")

    def _read_selection(self) -> None:
        text = self._selected_text()
        if text.strip():
            self.tts.speak_async(text[:1000] + ("..." if len(text) > 1000 else ""))
        else:
            self._say("No text is selected. Please select some text first.")

    def _code_review(self, code: str, instruction: str) -> str:
        """Selected code / screen text -> Gemini (user-initiated; the text leaves the machine)."""
        if self.assistant is None:
            return "Code analysis needs a GEMINI_API_KEY in the .env file."
        return self.assistant.complete(f"```\n{code[:20000]}\n```", instruction=instruction)

    def _analyze(self, selection: bool) -> None:
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
            analysis = self._code_review(code, ANALYZE_INSTRUCTION)
            print(f"Analysis result: {analysis[:100]}...")
            self.tts.speak_async(analysis)
        except Exception as e:  # noqa: BLE001
            print(f"Error in analyze command: {e}")
            self._say("There was an error processing the analyze command.")

    def _debug_code(self) -> None:
        try:
            code = self._selected_text()
            if not code.strip():
                self._say("No code is selected for debugging. Please select some code first.")
                return
            self._say("Debugging selected code. This may take a moment...")
            debug_result = self._code_review(code, DEBUG_INSTRUCTION)
            print(f"Debug analysis result: {debug_result[:100]}...")
            self.tts.speak_async(debug_result)
        except Exception as e:  # noqa: BLE001
            print(f"Error in debug command: {e}")
            self._say("There was an error processing the debug command.")

    def _handle_exit_command(self) -> None:
        self.tts.speak_async("Shutting down GRACE. Goodbye!")
        time.sleep(1.5)  # let the goodbye play before the app exits
        self.is_running = False
        self._session_open = False
        if self._on_exit is not None:
            self._on_exit()

    def _handle_help_command(self) -> None:
        print(HELP_TEXT)
        self.tts.speak(HELP_TEXT)


def build_assistant(settings: Any, registry: Any) -> Any:
    """GeminiAssistant from settings, or None when Gemini has no key / is disabled."""
    if "gemini" not in settings.enabled_providers():
        return None
    from grace.intent.assistant import GeminiAssistant
    from grace.intent.providers import GeminiClient

    g = settings.gemini
    client = GeminiClient(
        settings.api_key("gemini"), timeout_s=g.timeout_s, max_retries=settings.router.max_retries
    )
    return GeminiAssistant(
        client,
        registry,
        model=g.model,
        image_model=g.image_model,
        web_search=g.web_search,
        image_generation=g.image_generation,
        thinking_level=g.thinking_level,
        image_timeout_s=g.image_timeout_s,
        memory_reset_s=g.memory_reset_s,
    )
