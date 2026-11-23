"""Voice input: microphone -> VAD -> wake word / push-to-talk -> STT -> the agent.

Threads (docs/PLAN.md section 4), all talking through queues and callbacks, never blocking
each other:

    grace-voice   reads 32 ms frames from the ONE mic stream; Silero VAD, wake word, segmenter
                  (ListenPipeline.feed). Opens the mic only while it is on or push-to-talk is held.
    grace-stt     transcribes one utterance at a time while it is being spoken (SttWorker):
                  partial text -> overlay, final text -> ListenPipeline.handle_final
    grace-intent  runs each command through the agent (grammar, Gemini, actions), so a slow
                  answer never stops GRACE from listening (SerialWorker)

What the user experiences (``wakeword`` settings):

    idle      waiting for the wake word. With a wake-word model, only openWakeWord runs. Without
              one, each utterance is transcribed and must start with ``runtime.wake_word``
              ("listen, open notepad"). With the wake word off, GRACE is never idle.
    active    after the wake word: every utterance is a command, no wake word needed, until
              ``follow_up_s`` pass without one, or "stop listening". Dictation mode stays active
              until "stop dictation" or the mic is turned off.
    push-to-talk  hold the key: everything said until release is one command (mic on or off).

``cancel`` (mic off, panic) drops the utterance being captured and any text still on its way:
a generation counter makes late results harmless. ``Runtime`` only calls ``wake``,
``request_ptt``, ``release_ptt``, ``cancel`` and ``stop`` (grace.runtime.VoiceControl).
"""

from __future__ import annotations

import itertools
import logging
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from grace.input.audio import FRAME_S, AudioSource, SegmentEnd, Segmenter, VadModel
from grace.input.stt.base import Final, Partial, SttEngine, SttError
from grace.input.wakeword import WakeWordDetector

log = logging.getLogger(__name__)

Kind = Literal["command", "dictation", "ptt", "wake_check"]
WakeMode = Literal["model", "text", "off"]

FIRST_COMMAND_S = 6.0  # after a wake word, time to start the command even if follow_up_s is 0
ECHO_TAIL_S = 0.25  # ignore the mic this long after GRACE stops talking


# --------------------------------------------------------------------------- pure text helpers
def split_wake_word(text: str | None, wake_word: str) -> str | None:
    """Return what was said after the wake word, or None if the wake word wasn't said.

    The wake word must be a whole word ("listen", not "listening"); case is ignored.
    "Listen, open notepad" -> "open notepad"; "listen" -> "".
    """
    if not text or not wake_word:
        return None
    m = re.search(rf"\b{re.escape(wake_word)}\b[\s,.!?:]*(?P<rest>.*)", text, re.I | re.S)
    return m.group("rest").strip() if m else None


def strip_leading_wake_word(text: str, wake_word: str) -> str:
    """Drop a wake word said at the very start ("Hey Grace, open notepad" -> "open notepad").

    Only the start: "open Grace's folder" is left alone.
    """
    if not wake_word:
        return text.strip()
    m = re.match(
        rf"\s*(?:(?:hey|hi|ok|okay)[\s,]+)?{re.escape(wake_word)}\b[\s,.!?:]*(?P<rest>.*)",
        text,
        re.I | re.S,
    )
    return m.group("rest").strip() if m else text.strip()


# --------------------------------------------------------------------------- utterances
_END = object()
_CANCEL = object()
_ids = itertools.count(1)


@dataclass(eq=False)
class Utterance:
    """Audio of one utterance on its way to STT. Frames go in on the voice thread; the STT
    thread reads ``chunks()`` until the end (or a cancel)."""

    kind: Kind
    gen: int
    id: int = field(default_factory=lambda: next(_ids))
    started: float = 0.0  # perf_counter
    eos_t: float = 0.0  # perf_counter when the end of speech was decided (or the key released)
    hangover_s: float = 0.0  # silence between the last speech and the end decision
    end_reason: str = ""
    cancelled: bool = False
    frames: int = 0
    _q: queue.Queue[Any] = field(default_factory=queue.Queue, repr=False)

    def put(self, frame: np.ndarray) -> None:
        self.frames += 1
        self._q.put(frame)

    def end(self, t: float, reason: str, hangover_s: float = 0.0) -> None:
        self.eos_t, self.end_reason, self.hangover_s = t, reason, hangover_s
        self._q.put(_END)

    def cancel(self) -> None:
        self.cancelled = True
        self._q.put(_CANCEL)

    def chunks(self) -> Iterator[np.ndarray]:
        while True:
            item = self._q.get()
            if item is _END or item is _CANCEL:
                return
            yield item

    @property
    def duration_s(self) -> float:
        return self.frames * FRAME_S


# --------------------------------------------------------------------------- the pipeline
class ListenPipeline:
    """Decides what each frame and each transcript means. No threads, no hardware: the voice
    thread calls ``feed``; the STT thread calls ``handle_final`` / ``done``; any thread may call
    the control methods. One lock keeps them consistent (each call holds it for ~1 ms)."""

    def __init__(
        self,
        *,
        vad: VadModel,
        segmenter: Segmenter,
        submit: Callable[[Utterance], None],
        wake: WakeWordDetector | None = None,
        wake_mode: WakeMode = "text",
        wake_word: str = "listen",
        follow_up_s: float = 10.0,
        command_max_s: float = 12.0,
        dictation_max_s: float = 30.0,
        ptt_pre_roll_ms: int = 300,
        is_dictation: Callable[[], bool] = lambda: False,
        is_muted: Callable[[], bool] = lambda: False,
        on_listening: Callable[[bool], None] = lambda on: None,
        on_wake: Callable[[float | None], None] = lambda score: None,
        on_dictation_cancelled: Callable[[], None] = lambda: None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if wake_mode == "model" and wake is None:
            raise ValueError("wake_mode 'model' needs a detector")
        self.vad = vad
        self.seg = segmenter
        self._submit = submit
        self.wake = wake
        self.wake_mode = wake_mode
        self.wake_word = wake_word.lower()
        self.follow_up_s = follow_up_s
        self.command_max_s = command_max_s
        self.dictation_max_s = dictation_max_s
        self._is_dictation = is_dictation
        self._is_muted = is_muted
        self._on_listening = on_listening
        self._on_wake = on_wake
        self._on_dictation_cancelled = on_dictation_cancelled
        self._clock = clock
        self._lock = threading.RLock()
        self._recent: deque[np.ndarray] = deque(
            maxlen=max(0, round(ptt_pre_roll_ms / 1000 / FRAME_S))
        )
        self.gen = 0
        self.phase: Literal["idle", "active"] = self._rest_phase()
        self.deadline = 0.0  # perf_counter when the follow-up window closes
        self.ptt_held = False
        self._open: Utterance | None = None  # utterance being captured
        self._pending = 0  # submitted, not yet done
        self._muted_until = 0.0
        self._listening = False

    # ------------------------------------------------------------------ helpers (lock held)
    def _rest_phase(self) -> Literal["idle", "active"]:
        return "active" if self.wake_mode == "off" else "idle"

    def _publish_listening(self) -> None:
        on = (
            self.ptt_held
            or (self.phase == "active" and self.wake_mode != "off")
            or (self.wake_mode == "off" and self._open is not None)
        )
        if on != self._listening:
            self._listening = on
            self._on_listening(on)

    def _start(self, kind: Kind, audio: tuple[np.ndarray, ...] | list[np.ndarray]) -> Utterance:
        utt = Utterance(kind, self.gen, started=self._clock())
        for frame in audio:
            utt.put(frame)
        self._open = utt
        self._pending += 1
        self._submit(utt)
        return utt

    def _cancel_open(self) -> None:
        if self._open is not None:
            self._open.cancel()
            self._open = None

    def _activate(self, now: float) -> None:
        self.phase = "active"
        self.deadline = now + max(self.follow_up_s, FIRST_COMMAND_S)
        self._publish_listening()

    def _after_command(self, now: float) -> None:
        if self.wake_mode == "off" or self.phase != "active":
            return
        if self.follow_up_s <= 0 and not self._is_dictation():
            self.phase = "idle"
        else:
            self.deadline = now + self.follow_up_s
        self._publish_listening()

    # ------------------------------------------------------------------ voice thread
    def feed(self, frame: np.ndarray) -> None:
        with self._lock:
            now = self._clock()
            self._recent.append(frame)
            if self._open is not None and self._open.kind == "ptt":
                self._open.put(frame)
                if self._open.duration_s >= self.dictation_max_s:
                    self._open.end(now, "max_length")
                    self._open = None
                return
            if self._is_muted():
                self._muted_until = now + ECHO_TAIL_S
            muted = now < self._muted_until
            if self.phase == "idle" and self.wake_mode == "model":
                if not muted:
                    assert self.wake is not None
                    score = self.wake.process(frame)
                    if score is not None:
                        self.wake.reset()
                        self.seg.reset()
                        self.vad.reset()
                        self._activate(now)
                        self._on_wake(score)
                self.tick(now)
                return
            if muted and not self.seg.in_speech:
                self.seg.reset()
                self.vad.reset()
                self.tick(now)
                return
            prob = self.vad.prob(frame)
            dictation = self.phase == "active" and self._is_dictation()
            if not self.seg.in_speech:
                self.seg.max_s = self.dictation_max_s if dictation else self.command_max_s
            step = self.seg.push(frame, prob)
            if step.start:
                kind: Kind = (
                    "wake_check"
                    if self.phase == "idle"
                    else "dictation"
                    if dictation
                    else "command"
                )
                self._start(kind, step.audio)
                self._publish_listening()
            elif self._open is not None:
                for f in step.audio:
                    self._open.put(f)
            if step.end is not None:
                self._close(step.end, now)
            self.tick(now)

    def _close(self, end: SegmentEnd, now: float) -> None:
        utt, self._open = self._open, None
        if utt is None:
            return
        if end.too_short:
            utt.cancel()
        else:
            utt.end(now, end.reason, end.t - end.speech_end_t)
        self._publish_listening()

    def tick(self, now: float | None = None) -> None:
        """Close the follow-up window once it has passed (called every frame, and when idle)."""
        with self._lock:
            now = self._clock() if now is None else now
            if (
                self.phase == "active"
                and self.wake_mode != "off"
                and now >= self.deadline
                and self._open is None
                and self._pending == 0
                and not self.ptt_held
                and not self._is_dictation()
            ):
                self.phase = "idle"
                self._publish_listening()

    def wants_audio(self) -> bool:
        """Keep the mic open even with the mic toggle off (push-to-talk)."""
        with self._lock:
            return self.ptt_held or (self._open is not None and self._open.kind == "ptt")

    def mic_opened(self) -> None:
        with self._lock:
            self.vad.reset()
            self.seg.reset()
            if self.wake is not None:
                self.wake.reset()
            self._recent.clear()
            if self.wake_mode == "off":
                self.phase = "active"

    # ------------------------------------------------------------------ control (any thread)
    def ptt_down(self) -> None:
        with self._lock:
            if self.ptt_held:
                return
            self.ptt_held = True
            if self._open is not None and self._open.kind != "ptt":
                # The key wins: what was being captured is re-captured from the recent audio.
                self._cancel_open()
            self.seg.reset()
            self._start("ptt", list(self._recent))
            self._publish_listening()

    def ptt_up(self) -> None:
        with self._lock:
            if not self.ptt_held:
                return
            self.ptt_held = False
            if self._open is not None and self._open.kind == "ptt":
                self._open.end(self._clock(), "released")
                self._open = None
            self.vad.reset()
            self.seg.reset()
            self._publish_listening()

    def cancel(self) -> None:
        """Mic off / panic: drop everything in flight and go back to waiting."""
        with self._lock:
            self.gen += 1
            self._cancel_open()
            self.ptt_held = False
            self.seg.reset()
            self.phase = self._rest_phase()
            was_dictating = self._is_dictation()
            self._publish_listening()
        if was_dictating:
            self._on_dictation_cancelled()

    def end_session(self) -> None:
        """'stop listening': close the follow-up window now."""
        with self._lock:
            if self.wake_mode != "off":
                self.phase = "idle"
            self._publish_listening()

    # ------------------------------------------------------------------ STT thread
    def is_current(self, utt: Utterance) -> bool:
        with self._lock:
            return utt.gen == self.gen and not utt.cancelled

    def handle_final(self, utt: Utterance, text: str) -> str | None:
        """What to run for this transcript (None = nothing)."""
        with self._lock:
            if utt.gen != self.gen or utt.cancelled:
                return None
            text = text.strip()
            now = self._clock()
            if utt.kind == "wake_check":
                rest = split_wake_word(text, self.wake_word)
                if rest is None:
                    return None
                self._activate(now)
                self._on_wake(None)
                if rest:
                    self._after_command(now)
                return rest or None
            command = (
                text if utt.kind == "dictation" else strip_leading_wake_word(text, self.wake_word)
            )
            if command:
                self._after_command(now)
            return command or None

    def done(self, utt: Utterance) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)
            self.tick()


# --------------------------------------------------------------------------- workers
class SttWorker:
    """Transcribes utterances one at a time on its own thread (``run`` is the synchronous core).

    ``engine_for(dictation)`` picks the engine per utterance (grace.input.stt.SttRouter).
    """

    def __init__(
        self,
        engine_for: Callable[[bool], SttEngine | None],
        pipeline_ref: Callable[[], ListenPipeline],
        *,
        on_partial: Callable[[Utterance, str], None] = lambda u, t: None,
        on_command: Callable[[Utterance, str], None] = lambda u, t: None,
        on_final: Callable[[Utterance, Final], None] = lambda u, f: None,
        on_unavailable: Callable[[str], None] = lambda why: None,
        name: str = "grace-stt",
    ) -> None:
        self._engine_for = engine_for
        self._pipeline = pipeline_ref
        self._on_partial = on_partial
        self._on_command = on_command
        self._on_final = on_final
        self._on_unavailable = on_unavailable
        self._jobs: queue.Queue[Utterance | None] = queue.Queue()
        self._name = name
        self._thread: threading.Thread | None = None

    def submit(self, utt: Utterance) -> None:
        self._jobs.put(utt)

    def run(self, utt: Utterance) -> None:
        pipeline = self._pipeline()
        try:
            engine = self._engine_for(utt.kind == "dictation")
            if engine is None:
                for _ in utt.chunks():
                    pass
                if not utt.cancelled:
                    self._on_unavailable("no speech-to-text engine is available")
                return
            for event in engine.transcribe_stream(utt.chunks()):
                if isinstance(event, Partial):
                    if utt.kind != "wake_check" and pipeline.is_current(utt):
                        self._on_partial(utt, event.text)
                    continue
                if utt.cancelled:
                    return
                self._on_final(utt, event)
                command = pipeline.handle_final(utt, event.text)
                if command:
                    self._on_command(utt, command)
        except SttError as exc:
            log.warning("speech to text failed: %s", exc)
            if not utt.cancelled:
                self._on_unavailable(str(exc))
        except Exception:  # noqa: BLE001 - one bad utterance must not stop listening
            log.exception("speech to text crashed on utterance %d", utt.id)
        finally:
            pipeline.done(utt)

    def _run(self) -> None:
        while (utt := self._jobs.get()) is not None:
            self.run(utt)

    def start(self) -> SttWorker:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        if self._thread is None:
            return
        self._jobs.put(None)
        self._thread.join(timeout)
        self._thread = None


class SerialWorker:
    """Runs submitted callables one after another on one thread (the agent / intent thread)."""

    def __init__(self, name: str = "grace-intent") -> None:
        self._jobs: queue.Queue[Callable[[], Any] | None] = queue.Queue()
        self._name = name
        self._thread: threading.Thread | None = None

    def submit(self, fn: Callable[[], Any]) -> None:
        self._jobs.put(fn)

    def _run(self) -> None:
        while (fn := self._jobs.get()) is not None:
            try:
                fn()
            except Exception:  # noqa: BLE001
                log.exception("command handling failed")

    def start(self) -> SerialWorker:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._jobs.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            log.info("a command is still running; leaving it to finish")
        self._thread = None


class VoiceWorker:
    """The voice thread: owns the microphone stream and feeds the pipeline.

    Implements grace.runtime.VoiceControl. The mic is open while ``mic_on()`` is true or
    push-to-talk needs it, and closed otherwise (privacy, and the mic light goes off).
    """

    def __init__(
        self,
        pipeline: ListenPipeline,
        source: AudioSource,
        *,
        mic_on: Callable[[], bool],
        stt: SttWorker | None = None,
        intents: SerialWorker | None = None,
        on_mic_error: Callable[[str], None] = lambda why: None,
        name: str = "grace-voice",
        retry_s: float = 3.0,
    ) -> None:
        self.pipeline = pipeline
        self.source = source
        self.stt = stt
        self.intents = intents
        self._mic_on = mic_on
        self._on_mic_error = on_mic_error
        self._name = name
        self._retry_s = retry_s
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mic_error: str | None = None
        self._retry_at = 0.0

    # ------------------------------------------------------------------ VoiceControl
    def wake(self) -> None:
        self._wake.set()

    def request_ptt(self) -> None:
        self.pipeline.ptt_down()
        self._wake.set()

    def release_ptt(self) -> None:
        self.pipeline.ptt_up()
        self._wake.set()

    def cancel(self) -> None:
        self.pipeline.cancel()
        self._wake.set()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> VoiceWorker:
        if self.stt is not None:
            self.stt.start()
        if self.intents is not None:
            self.intents.start()
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self.pipeline.cancel()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        self.source.stop()
        if self.stt is not None:
            self.stt.stop()
        if self.intents is not None:
            self.intents.stop()

    # ------------------------------------------------------------------ voice thread
    def step(self, read_timeout: float = 0.1) -> None:
        """One iteration (tests drive this directly)."""
        want = self._mic_on() or self.pipeline.wants_audio()
        if want and not self.source.is_open and time.monotonic() >= self._retry_at:
            try:
                self.source.start()
            except Exception as exc:  # noqa: BLE001 - no mic, device busy, unplugged
                why = f"{type(exc).__name__}: {exc}"
                self._retry_at = time.monotonic() + self._retry_s
                if why != self._mic_error:
                    self._mic_error = why
                    log.warning("microphone unavailable: %s", why)
                    self._on_mic_error(why)
                return
            self._mic_error = None
            self.pipeline.mic_opened()
        elif not want and self.source.is_open:
            self.source.stop()
        if not self.source.is_open:
            self._wake.wait(0.5 if not want else read_timeout)
            self._wake.clear()
            self.pipeline.tick()
            return
        frame = self.source.read(read_timeout)
        if frame is None:
            self.pipeline.tick()
            return
        self.pipeline.feed(frame)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:  # noqa: BLE001 - keep listening after a bad frame
                log.exception("voice worker error")
                self._stop.wait(0.5)


# --------------------------------------------------------------------------- assembly
class VoiceUnavailable(RuntimeError):
    """Voice input can't start (model files missing, no STT engine). Safe to show the user."""


@dataclass(frozen=True)
class VoiceHooks:
    """What the voice stack tells the rest of the app (all called from worker threads)."""

    command: Callable[[str], None]  # final command text -> the agent (on the intent thread)
    partial: Callable[[str], None] = lambda text: None  # overlay
    listening: Callable[[bool], None] = lambda on: None  # status chip
    wake: Callable[[float | None], None] = lambda score: None  # earcon, metrics
    stt_result: Callable[[Utterance, Final], None] = lambda u, f: None  # metrics
    problem: Callable[[str], None] = lambda why: None  # mic or STT unavailable
    is_dictation: Callable[[], bool] = lambda: False
    is_muted: Callable[[], bool] = lambda: False  # GRACE is talking
    dictation_cancelled: Callable[[], None] = lambda: None


def build_voice_worker(
    settings: Any, hooks: VoiceHooks, *, mic_on: Callable[[], bool], wake_word: str
) -> tuple[VoiceWorker, list[str]]:
    """The real voice stack from settings. Returns (worker, lines describing it).

    Raises VoiceUnavailable when Silero VAD or every STT engine is missing.
    """
    from grace.input.audio import MicSource, SileroVad
    from grace.input.stt import build_stt_router
    from grace.input.wakeword import load_wake_word
    from grace.models import MODELS_DIR, SILERO_VAD, resolve

    vad_path = MODELS_DIR / SILERO_VAD
    if not vad_path.is_file():
        raise VoiceUnavailable("Silero VAD model missing: run scripts/fetch_models.py")
    audio, ww = settings.audio, settings.wakeword
    router = build_stt_router(settings)
    stt_desc = router.preload()
    if router.engine_for(dictation=False) is None:
        raise VoiceUnavailable(stt_desc)
    detector, wake_desc = load_wake_word(
        ww.enabled,
        resolve(ww.model_path),
        models_dir=MODELS_DIR,
        threshold=ww.threshold,
        patience=ww.patience,
    )
    mode: WakeMode = "model" if detector is not None else ("text" if ww.enabled else "off")
    if mode == "text":
        wake_desc = f"say '{wake_word}' ({wake_desc})"
    elif mode == "off":
        wake_desc = "off: everything said while the mic is on is a command"

    intents = SerialWorker()
    pipeline_ref: list[ListenPipeline] = []

    def on_command(utt: Utterance, text: str) -> None:
        def run() -> None:
            if utt.gen == pipeline_ref[0].gen:  # mic off / panic since: drop it
                hooks.command(text)

        intents.submit(run)

    stt = SttWorker(
        router.engine_for,
        lambda: pipeline_ref[0],
        on_partial=(lambda u, t: hooks.partial(t)) if settings.stt.partials else lambda u, t: None,
        on_command=on_command,
        on_final=hooks.stt_result,
        on_unavailable=hooks.problem,
    )
    pipeline = ListenPipeline(
        vad=SileroVad(vad_path),
        segmenter=Segmenter(
            threshold=audio.vad_threshold,
            end_of_speech_ms=audio.end_of_speech_ms,
            max_s=audio.max_utterance_s,
            pre_roll_ms=audio.pre_roll_ms,
            min_speech_ms=audio.min_speech_ms,
        ),
        submit=stt.submit,
        wake=detector,
        wake_mode=mode,
        wake_word=wake_word,
        follow_up_s=ww.follow_up_s,
        command_max_s=audio.max_utterance_s,
        dictation_max_s=audio.dictation_max_s,
        ptt_pre_roll_ms=audio.pre_roll_ms,
        is_dictation=hooks.is_dictation,
        is_muted=hooks.is_muted,
        on_listening=hooks.listening,
        on_wake=hooks.wake,
        on_dictation_cancelled=hooks.dictation_cancelled,
    )
    pipeline_ref.append(pipeline)
    worker = VoiceWorker(
        pipeline,
        MicSource(audio.input_device),
        mic_on=mic_on,
        stt=stt,
        intents=intents,
        on_mic_error=lambda why: hooks.problem(f"Microphone unavailable: {why}"),
    )
    lines = [
        f"speech to text: {stt_desc} (language {settings.language})",
        f"wake word: {wake_desc}",
        f"end of speech after {audio.end_of_speech_ms} ms of silence",
    ]
    return worker, lines
