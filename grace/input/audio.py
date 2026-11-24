"""Audio input: one 16 kHz mono microphone stream, Silero VAD, utterance segmentation.

    source = MicSource(device=None).start()        # sounddevice; 512-sample (32 ms) frames
    vad = SileroVad(models_dir / "vad/silero_vad.onnx")
    seg = Segmenter(threshold=0.5, end_of_speech_ms=500, max_s=12.0)
    while (frame := source.read(0.1)) is not None:
        step = seg.push(frame, vad.prob(frame))
        step.start / step.audio / step.end         # utterance opened / frames for it / closed

One stream feeds everything (wake word, VAD, STT): grace/input/voice.py reads it on the voice
thread and routes frames. ``ArraySource`` replays a WAV or array the same way (tests, eval).

The segmenter is pure (probabilities in, decisions out) and follows Silero's own rule: speech
starts on a frame >= threshold; it ends once frames stay below ``threshold - 0.15`` for
``end_of_speech_ms`` (a frame >= threshold in between restarts that count), or at ``max_s``.
Up to ``pre_roll_ms`` of audio from before the start goes with the utterance so the first syllable
is not cut. An utterance with less than ``min_speech_ms`` of speech is marked ``too_short``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME = 512  # samples per frame: what Silero VAD v5+ expects at 16 kHz (32 ms)
FRAME_S = FRAME / SAMPLE_RATE


# --------------------------------------------------------------------------- sources
class AudioSource(Protocol):
    @property
    def is_open(self) -> bool: ...

    def start(self) -> AudioSource: ...

    def stop(self) -> None: ...

    def read(self, timeout: float) -> np.ndarray | None:
        """Next float32 frame of ``FRAME`` samples in [-1, 1], or None if none came in time."""
        ...


class MicSource:
    """The microphone as a sounddevice InputStream; frames queue up for the voice thread.

    The PortAudio callback only copies the frame into a bounded queue. If the reader falls behind
    by more than ``max_queue_s`` the oldest frames are dropped (counted in ``dropped``), so a
    stalled consumer never grows memory or delays live audio forever.
    """

    def __init__(self, device: int | str | None = None, *, max_queue_s: float = 10.0) -> None:
        self.device = device
        self._max_frames = max(1, int(max_queue_s / FRAME_S))
        self._q: queue.Queue[np.ndarray] = queue.Queue(self._max_frames)
        self._stream: Any = None
        self._lock = threading.Lock()
        self.dropped = 0
        self.status_flags = 0

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def start(self) -> MicSource:
        import sounddevice as sd

        with self._lock:
            if self._stream is not None:
                return self
            self._q = queue.Queue(self._max_frames)
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME,
                device=self.device,
                callback=self._callback,
            )
            stream.start()
            self._stream = stream
        log.info("microphone open (%s)", self.device if self.device is not None else "default")
        return self

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self.status_flags += 1
        frame = indata[:, 0].copy()
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self.dropped += 1
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass

    def read(self, timeout: float) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:  # noqa: BLE001 - device unplugged mid-stream
            log.debug("closing the microphone failed", exc_info=True)
        log.info("microphone closed")


class ArraySource:
    """Replays samples as ``FRAME``-sized frames (tests, eval). ``realtime`` paces like a mic."""

    def __init__(
        self,
        samples: np.ndarray,
        *,
        realtime: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._frames = list(frames(samples))
        self._i = 0
        self._open = False
        self.realtime = realtime
        self._sleep = sleep

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def exhausted(self) -> bool:
        return self._i >= len(self._frames)

    def start(self) -> ArraySource:
        self._open = True
        return self

    def stop(self) -> None:
        self._open = False

    def read(self, timeout: float = 0.0) -> np.ndarray | None:
        if self.exhausted:
            return None
        if self.realtime:
            self._sleep(FRAME_S)
        frame = self._frames[self._i]
        self._i += 1
        return frame


def frames(samples: np.ndarray, size: int = FRAME) -> Iterator[np.ndarray]:
    """``samples`` as float32 frames of ``size``; the last one is zero-padded."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    for i in range(0, len(samples), size):
        frame = samples[i : i + size]
        if len(frame) < size:
            frame = np.pad(frame, (0, size - len(frame)))
        yield frame


def load_wav(path: str | Path) -> np.ndarray:
    """A WAV file as float32 mono at 16 kHz (resampled if needed)."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // g, int(sr) // g)
    return np.asarray(mono, dtype=np.float32)


def to_int16(frame: np.ndarray) -> np.ndarray:
    return (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)


# --------------------------------------------------------------------------- VAD
class VadModel(Protocol):
    def prob(self, frame: np.ndarray) -> float:
        """Speech probability (0-1) of one ``FRAME``-sample frame."""
        ...

    def reset(self) -> None: ...


class SileroVad:
    """Silero VAD v5/v6 ONNX on onnxruntime (CPU, 1 thread: a frame costs well under 1 ms).

    The model takes 64 samples of context + 512 new samples, a [2, 1, 128] state and the rate.
    """

    CONTEXT = 64

    def __init__(self, path: str | Path) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        if frame.shape[-1] != FRAME:
            raise ValueError(f"Silero VAD needs {FRAME}-sample frames, got {frame.shape[-1]}")
        x = np.concatenate([self._context, frame.reshape(1, -1).astype(np.float32)], axis=1)
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[:, -self.CONTEXT :]
        return float(np.asarray(out).reshape(-1)[0])


# --------------------------------------------------------------------------- segmentation
@dataclass(frozen=True)
class SegmentEnd:
    reason: str  # "silence" | "max_length"
    t: float  # stream time (s) when the end was decided
    speech_end_t: float  # stream time (s) at the end of the last frame with speech
    duration_s: float  # from the start trigger to the end
    voiced_ms: float  # frames at or above the threshold
    too_short: bool  # voiced_ms < min_speech_ms: a click or cough, not a command


@dataclass(frozen=True)
class Step:
    start: bool = False  # an utterance opened on this frame
    audio: tuple[np.ndarray, ...] = ()  # frames that belong to the open utterance
    end: SegmentEnd | None = None  # the utterance closed on this frame


_IDLE = Step()


class Segmenter:
    def __init__(
        self,
        *,
        threshold: float = 0.5,
        end_of_speech_ms: int = 500,
        max_s: float = 12.0,
        pre_roll_ms: int = 300,
        min_speech_ms: int = 200,
    ) -> None:
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.01)
        self.end_of_speech_ms = end_of_speech_ms
        self.max_s = max_s  # may change between utterances (command vs dictation)
        self.min_speech_ms = min_speech_ms
        pre_roll_frames = max(0, round(pre_roll_ms / 1000 / FRAME_S))
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_frames)
        self._n = 0  # frames seen
        self.reset()

    @property
    def in_speech(self) -> bool:
        return self._start_t is not None

    @property
    def t(self) -> float:
        return self._n * FRAME_S

    def reset(self) -> None:
        """Forget any open utterance and the pre-roll (after a wake word, mute, cancel)."""
        self._start_t: float | None = None
        self._silence_since: float | None = None
        self._voiced = 0
        self._last_voiced_t = 0.0
        self._pre_roll.clear()

    def push(self, frame: np.ndarray, prob: float) -> Step:
        self._n += 1
        now = self.t  # end of this frame
        voiced = prob >= self.threshold
        if self._start_t is None:
            if not voiced:
                self._pre_roll.append(frame)
                return _IDLE
            audio = (*self._pre_roll, frame)
            self._pre_roll.clear()
            self._start_t = now - FRAME_S
            self._silence_since = None
            self._voiced = 1
            self._last_voiced_t = now
            return Step(start=True, audio=audio)

        if voiced:
            self._voiced += 1
            self._last_voiced_t = now
            self._silence_since = None
        elif prob < self.neg_threshold and self._silence_since is None:
            self._silence_since = now - FRAME_S

        reason = None
        if (
            self._silence_since is not None
            and (now - self._silence_since) * 1000 >= self.end_of_speech_ms
        ):
            reason = "silence"
        elif now - self._start_t >= self.max_s:
            reason = "max_length"
        if reason is None:
            return Step(audio=(frame,))
        voiced_ms = self._voiced * FRAME_S * 1000
        end = SegmentEnd(
            reason=reason,
            t=now,
            speech_end_t=self._last_voiced_t,
            duration_s=now - self._start_t,
            voiced_ms=voiced_ms,
            too_short=voiced_ms < self.min_speech_ms,
        )
        self._start_t = None
        self._silence_since = None
        return Step(audio=(frame,), end=end)


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    reason: str
    voiced_ms: float


def segment(samples: np.ndarray, vad: VadModel, segmenter: Segmenter) -> list[Segment]:
    """Utterances in ``samples`` (tests, eval). Too-short blips are left out. An utterance still
    open at the end of the audio closes there with reason "eof"."""
    vad.reset()
    out: list[Segment] = []
    start = 0.0
    for frame in frames(samples):
        step = segmenter.push(frame, vad.prob(frame))
        if step.start:
            start = segmenter.t - FRAME_S
        if step.end is not None and not step.end.too_short:
            out.append(Segment(start, step.end.t, step.end.reason, step.end.voiced_ms))
    if segmenter.in_speech:
        out.append(Segment(start, segmenter.t, "eof", 0.0))
        segmenter.reset()
    return out
