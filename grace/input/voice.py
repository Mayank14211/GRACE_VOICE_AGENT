"""Voice worker thread: wraps the legacy voice agent (old STT) for the S05 runtime.

One thread owns the microphone, so wake-word passes, conversations and push-to-talk never open
two captures at once (the old code's race, docs/CONTEXT.md 5.2):

    loop:  push-to-talk requested?  -> one command, no wake word
           mic off?                 -> sleep until woken (mic on, push-to-talk, stop)
           else                     -> one short wake-word pass (~1 s if silent)
                                       wake word heard -> conversation (follow-ups, no wake word)

Other threads only call ``wake``, ``request_ptt``, ``release_ptt``, ``cancel`` and ``stop``;
they never block. ``cancel`` (mic off, panic stop) bumps a generation counter: whatever the
current listen returns is dropped and the conversation ends.

Limits of the old STT, fixed by S06 (sounddevice stream + VAD): a listen that has started can't be
interrupted, so the microphone is only released when that pass ends (about 1 s when silent, up to
~6 s plus recognition when someone is talking); push-to-talk capture ends on a pause, not on key
release. The state and the overlay change immediately either way.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from typing import Any, Protocol

log = logging.getLogger(__name__)


def split_wake_word(text: str | None, wake_word: str) -> str | None:
    """Return what was said after the wake word, or None if the wake word wasn't said.

    The wake word must be a whole word ("listen", not "listening"); case is ignored.
    "Listen, open notepad" -> "open notepad"; "listen" -> "".
    """
    if not text or not wake_word:
        return None
    m = re.search(rf"\b{re.escape(wake_word)}\b[\s,.!?:]*(?P<rest>.*)", text, re.I | re.S)
    return m.group("rest").strip() if m else None


class WakePhraseListener(Protocol):
    def listen_for_wake_phrase(self) -> str | None: ...


class VoiceAgent(Protocol):
    wake_word: str
    is_running: bool
    may_listen: Callable[[], bool]

    def _on_wake_word_detected(self, rest: str = "") -> None: ...

    def push_to_talk(self, heard: str | None = None) -> None: ...


class VoiceWorker:
    def __init__(
        self,
        agent: Any,
        recognizer: WakePhraseListener,
        *,
        mic_on: Callable[[], bool],
        name: str = "grace-voice",
    ) -> None:
        self.agent = agent
        self._recognizer = recognizer
        self._mic_on = mic_on
        self._name = name
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._ptt_requested = False
        self._in_ptt = False
        self._gen = 0  # bumped by cancel()
        self._session_gen = 0  # generation the current conversation / push-to-talk started in
        self._thread: threading.Thread | None = None
        self.wake_passes = 0
        agent.may_listen = self.may_listen

    # ------------------------------------------------------------------ control (any thread)
    def wake(self) -> None:
        self._wake.set()

    def request_ptt(self) -> None:
        if getattr(self.agent, "_session_open", False):
            return  # already in a conversation: it is listening anyway
        with self._lock:
            self._ptt_requested = True
        self._wake.set()

    def release_ptt(self) -> None:
        """Key released. The old STT ends the capture on a pause, not on release (S06)."""

    def cancel(self) -> None:
        """Mic off / panic: drop what the current listen returns and end any conversation."""
        with self._lock:
            self._gen += 1
            self._ptt_requested = False
        self._wake.set()

    def may_listen(self) -> bool:
        with self._lock:
            current = self._session_gen == self._gen
            in_ptt = self._in_ptt
        return not self._stop.is_set() and current and (in_ptt or self._mic_on())

    def start(self) -> VoiceWorker:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self.cancel()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                # A legacy listen can't be interrupted; the daemon thread ends with the process.
                log.info("voice worker still inside a listen; leaving it to finish")
            self._thread = None

    # ------------------------------------------------------------------ worker thread
    def _take_ptt(self) -> bool:
        with self._lock:
            requested, self._ptt_requested = self._ptt_requested, False
            if requested:
                self._session_gen = self._gen
                self._in_ptt = True
        return requested

    def _run_ptt(self, heard: str | None) -> None:
        try:
            self.agent.push_to_talk(heard)
        finally:
            with self._lock:
                self._in_ptt = False

    def step(self) -> None:
        """One iteration (tests drive this directly)."""
        if self._take_ptt():
            self._run_ptt(None)
            return
        if not self._mic_on():
            self._wake.wait()
            self._wake.clear()
            return
        with self._lock:
            gen = self._gen
        text = self._recognizer.listen_for_wake_phrase()
        self.wake_passes += 1
        if self._stop.is_set():
            return
        if self._take_ptt():
            # The key went down during this pass: what was heard is the command.
            rest = split_wake_word(text, self.agent.wake_word)
            self._run_ptt(rest if rest is not None else text)
            return
        with self._lock:
            if gen != self._gen:
                return  # mic toggled / panic during the pass
            self._session_gen = gen
        if not self._mic_on():
            return
        rest = split_wake_word(text, self.agent.wake_word)
        if rest is not None:
            self.agent._on_wake_word_detected(rest)

    def _run(self) -> None:
        self.agent.is_running = True
        try:
            while not self._stop.is_set():
                try:
                    self.step()
                except Exception:  # noqa: BLE001 - keep listening after a bad pass
                    log.exception("voice worker error")
                    self._stop.wait(1.0)
        finally:
            self.agent.is_running = False
