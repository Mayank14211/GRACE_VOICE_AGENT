"""Voice agent: what GRACE does with a transcribed utterance (grammar, router, TTS replies).

Moved here from main.py in S05. Since S06 it no longer listens: grace/input/voice.py owns the
microphone, wake word, follow-up window and push-to-talk, and calls ``handle_utterance`` on the
intent thread with each final transcript.

    transcript -> grammar.parse()            fixed commands (never an LLM)
               -> resolve_followup()         "a bit more", "undo that", "now the brightness"
               -> router.route()             LLM provider chain (S07) -> validated Intents
               -> app handler | dispatcher   (registry, confirmation gate for risky intents)

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
from grace.intent.dialogue import READ_BEFORE_SET, Dialogue, is_undo_phrase, resolve_followup
from grace.intent.grammar import Grammar, default_grammar
from grace.intent.router import spoken
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
    "- Follow-ups: 'a bit more', 'less', 'undo that', 'now the brightness', 'make it 30', 'again'\n"
    "- Anything else goes to the assistant (Groq, then Gemini, then OpenRouter): questions,\n"
    "  several steps at once ('mute and open youtube'), 'search the web for ...' (Tavily key),\n"
    "  'draw a picture of ...' (Cloudflare keys)\n"
    "- After the wake word (a short tone), keep talking: follow-up commands need no wake word\n"
    "  until you pause. 'Stop listening' ends it; 'start over' makes the assistant forget.\n"
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
        router: Any = None,
        extras: Any = None,
        dialogue: Dialogue | None = None,
        wake_word: str = "listen",
        on_heard: Callable[[str], None] = _noop,
        on_reply: Callable[[str], None] = _noop,
        on_mode: Callable[[str], None] = _noop,
        on_end_session: Callable[[], None] = _noop,
        on_exit: Callable[[], None] | None = None,
    ) -> None:
        """Hardware-backed parts (tts, screen_reader, registry) can be injected; tests pass fakes.
        Anything not injected is created for real. ``router`` (grace.intent.router.Router)
        handles whatever the grammar and follow-ups don't match; None disables that.
        ``extras`` (grace.intent.assistant.Extras) runs web search (Tavily), images (Cloudflare
        FLUX) and code analysis (router without Gemini); None disables them.
        ``on_end_session`` is called for "stop listening" (the voice pipeline closes its
        follow-up window).
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
        self.router = router
        self.extras = extras
        self.dialogue = dialogue if dialogue is not None else Dialogue()
        self._replies: list[str] | None = None  # what GRACE says during the current command
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
        # App intents with arguments (from the router).
        self._arg_handlers: dict[str, Callable[[dict[str, Any]], None]] = {
            "app.answer": lambda args: self._say(str(args.get("text", ""))),
            "app.web_search": self._web_search,
            "app.generate_image": self._generate_image,
        }

    # -- feedback -------------------------------------------------------------------------------
    def _say(self, message: str) -> None:
        if self._replies is not None:
            self._replies.append(message)
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
        self.dialogue.reset()
        self._say("Okay, starting a new conversation.")

    # -- command routing ------------------------------------------------------------------------
    def _process_command(self, command: str) -> Any:
        """Grammar -> follow-up -> router; the exchange is remembered for the next request."""
        print(f"Processing command: '{command}'")
        self.command_history.append(command)
        text = strip_execute_word(command)
        self._replies = []
        try:
            return self._route(text)
        finally:
            replies, self._replies = self._replies, None
            self.dialogue.add(text, " ".join(replies))

    def _route(self, text: str) -> Any:
        intent = self.grammar.parse(text)
        normalized = self.grammar.normalize(text)
        # "undo that" means "reverse the last change" when there is one; otherwise Ctrl+Z.
        if intent is None or is_undo_phrase(normalized):
            followup = resolve_followup(
                normalized,
                self.dialogue.last(),
                is_risky=self.registry.is_risky,
                steps=self.grammar.steps,
                scroll=self.grammar.scroll,
            )
            intent = followup or intent
        if intent is None:
            return self._ask_router(text)
        return self._handle_intent(intent)

    def _ask_router(self, text: str) -> Any:
        """Unmatched speech -> LLM router -> validated intents (risky ones via the gate)."""
        if self.gate.pending is not None:
            self.gate.cancel()  # anything else abandons a pending risky action
        if self.router is None:
            self._say("I'm not sure how to help with that.")
            return None
        result = self.router.route(text, self.dialogue)
        if not result.intents:
            self._say(result.reply)
            return result
        # A risky intent waits for confirmation, and any later command would cancel it, so it
        # runs last (and there is at most one).
        safe = [i for i in result.intents if not self.registry.is_risky(i.name)]
        risky = [i for i in result.intents if self.registry.is_risky(i.name)][:1]
        for intent in safe + risky:
            self._handle_intent(intent)
        return result

    def _handle_intent(self, intent: Intent) -> Any:
        print(f"Intent: {intent.name} {intent.args}")
        handler = self._app_handlers.get(intent.name)
        arg_handler = self._arg_handlers.get(intent.name)
        if handler is not None or arg_handler is not None:
            if self.gate.pending is not None:
                self.gate.cancel()
            if arg_handler is not None:
                arg_handler(dict(intent.args))
            elif handler is not None:
                handler()
            return None
        before = self._read_before(intent)
        result = self.dispatcher.handle(intent)  # also app.confirm / app.cancel
        followable = intent.name in self.registry and not self.registry.is_risky(intent.name)
        if result is not None and result.ok and followable:
            self.dialogue.record(intent, before)
        return result

    def _read_before(self, intent: Intent) -> dict[str, Any]:
        """The value a `.set` replaces, so "undo that" can put it back."""
        reader = READ_BEFORE_SET.get(intent.name)
        if reader is None or reader not in self.registry:
            return {}
        result = self.registry.execute(reader)
        level = result.data.get("level") if result.ok else None
        return {"level": level} if level is not None else {}

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
        """Selected code / screen text -> the LLM chain WITHOUT Gemini (user-initiated; the
        text leaves the machine, but never to Gemini)."""
        if self.extras is None:
            return "Code analysis needs a Groq or OpenRouter key in the .env file."
        return self.extras.complete(f"```\n{code[:20000]}\n```", instruction=instruction)

    def _web_search(self, args: dict[str, Any]) -> None:
        if self.extras is None:
            self._say("Web search needs a TAVILY_API_KEY in the .env file.")
            return
        found = self.extras.web_search(str(args.get("query", "")))
        for title, url in found.sources:
            print(f"  source: {title} <{url}>")
        self._say(spoken(found.text) or found.text)

    def _generate_image(self, args: dict[str, Any]) -> None:
        if self.extras is None:
            self._say("Image generation needs the Cloudflare keys in the .env file.")
            return
        made = self.extras.generate_image(str(args.get("prompt", "")))
        for path in made.images:
            print(f"  image: {path}")
        self._say(made.text)

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
