"""grace/input/audio.py: VAD segmentation on WAV fixtures, the pure segmenter, sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grace.input.audio import (
    FRAME,
    FRAME_S,
    SAMPLE_RATE,
    ArraySource,
    MicSource,
    Segmenter,
    SileroVad,
    frames,
    load_wav,
    segment,
    to_int16,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "audio"


@pytest.fixture(scope="module")
def vad() -> SileroVad:
    return SileroVad(FIX / "silero_vad.onnx")


# --------------------------------------------------------------------------- real Silero on WAVs
def test_silence_has_no_utterance(vad: SileroVad) -> None:
    assert segment(load_wav(FIX / "silence.wav"), vad, Segmenter()) == []


def test_one_command_is_one_utterance(vad: SileroVad) -> None:
    (seg,) = segment(load_wav(FIX / "one_command.wav"), vad, Segmenter())
    # Speech is placed at 0.80 s; the end comes end_of_speech_ms after the last word.
    assert seg.start_s == pytest.approx(0.8, abs=0.15)
    assert seg.reason == "silence"
    assert 0.5 <= seg.voiced_ms / 1000 <= 1.5


def test_two_commands_are_two_utterances(vad: SileroVad) -> None:
    first, second = segment(load_wav(FIX / "two_commands.wav"), vad, Segmenter())
    assert first.end_s < second.start_s
    assert second.start_s - first.end_s > 0.5  # the 1.5 s pause minus the 0.5 s end-of-speech


def test_a_long_end_of_speech_merges_the_two_commands(vad: SileroVad) -> None:
    (seg,) = segment(load_wav(FIX / "two_commands.wav"), vad, Segmenter(end_of_speech_ms=2000))
    assert seg.reason in {"silence", "eof"}


def test_max_length_cuts_an_utterance(vad: SileroVad) -> None:
    segs = segment(load_wav(FIX / "one_command.wav"), vad, Segmenter(max_s=0.4, min_speech_ms=0))
    assert segs[0].reason == "max_length"
    assert segs[0].end_s - segs[0].start_s == pytest.approx(0.4, abs=FRAME_S)


def test_silero_rejects_a_wrong_frame_size(vad: SileroVad) -> None:
    with pytest.raises(ValueError):
        vad.prob(np.zeros(FRAME // 2, dtype=np.float32))


# --------------------------------------------------------------------------- pure segmenter
def run(seg: Segmenter, probs: list[float]):
    """Push one frame per probability; frames carry their index so tests can see which went."""
    steps = []
    for i, p in enumerate(probs):
        steps.append(seg.push(np.full(FRAME, i, dtype=np.float32), p))
    return steps


def frame_ids(step) -> list[int]:
    return [int(f[0]) for f in step.audio]


def test_start_carries_the_pre_roll() -> None:
    seg = Segmenter(pre_roll_ms=100)  # 100 ms = 3 frames
    steps = run(seg, [0.0] * 5 + [0.9])
    assert steps[5].start and frame_ids(steps[5]) == [2, 3, 4, 5]
    assert seg.in_speech


def test_end_after_end_of_speech_silence() -> None:
    seg = Segmenter(end_of_speech_ms=96, min_speech_ms=0)  # 3 frames
    steps = run(seg, [0.9, 0.9, 0.1, 0.1, 0.1])
    assert steps[-1].end is not None and steps[-1].end.reason == "silence"
    assert steps[-1].end.speech_end_t == pytest.approx(2 * FRAME_S)
    assert all(s.end is None for s in steps[:-1])
    assert not seg.in_speech


def test_speech_in_between_restarts_the_silence_count() -> None:
    seg = Segmenter(end_of_speech_ms=96, min_speech_ms=0)
    steps = run(seg, [0.9, 0.1, 0.1, 0.9, 0.1, 0.1])
    assert all(s.end is None for s in steps)


def test_uncertain_frames_do_not_restart_the_silence_count() -> None:
    # 0.4 is between neg_threshold (0.35) and threshold (0.5): it neither ends nor restarts.
    seg = Segmenter(end_of_speech_ms=96, min_speech_ms=0)
    steps = run(seg, [0.9, 0.1, 0.4, 0.4])
    assert steps[-1].end is not None


def test_too_short_blip_is_flagged() -> None:
    seg = Segmenter(end_of_speech_ms=64, min_speech_ms=200)
    steps = run(seg, [0.9, 0.1, 0.1])
    assert steps[-1].end is not None and steps[-1].end.too_short


def test_max_length_can_change_between_utterances() -> None:
    seg = Segmenter(max_s=0.1, min_speech_ms=0)
    assert run(seg, [0.9] * 4)[-1].end.reason == "max_length"
    seg.max_s = 1.0
    assert all(s.end is None for s in run(seg, [0.9] * 10))


def test_reset_drops_an_open_utterance_and_the_pre_roll() -> None:
    seg = Segmenter(pre_roll_ms=100)
    run(seg, [0.0, 0.0, 0.9])
    seg.reset()
    assert not seg.in_speech
    step = seg.push(np.full(FRAME, 7, dtype=np.float32), 0.9)
    assert frame_ids(step) == [7]


# --------------------------------------------------------------------------- sources and helpers
def test_frames_pad_the_last_one() -> None:
    out = list(frames(np.ones(FRAME + 10, dtype=np.float32)))
    assert len(out) == 2 and out[1][9] == 1.0 and out[1][10] == 0.0


def test_array_source_replays_frames_and_ends() -> None:
    sleeps: list[float] = []
    src = ArraySource(np.zeros(FRAME * 2), realtime=True, sleep=sleeps.append).start()
    assert src.is_open and src.read() is not None and src.read() is not None
    assert src.read() is None and src.exhausted
    assert sleeps == [FRAME_S, FRAME_S]


def test_mic_callback_drops_the_oldest_frame_when_full() -> None:
    mic = MicSource(max_queue_s=2 * FRAME_S)  # room for 2 frames
    for i in range(3):
        mic._callback(np.full((FRAME, 1), i, dtype=np.float32), FRAME, None, None)
    assert mic.dropped == 1
    assert [mic.read(0)[0] for _ in range(2)] == [1.0, 2.0]
    assert mic.read(0) is None


def test_load_wav_resamples_to_16k(tmp_path: Path) -> None:
    import soundfile as sf

    t = np.arange(22050) / 22050
    sf.write(tmp_path / "a.wav", np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1), 22050)
    audio = load_wav(tmp_path / "a.wav")
    assert audio.dtype == np.float32 and abs(len(audio) - SAMPLE_RATE) <= 1


def test_to_int16_clips() -> None:
    assert to_int16(np.array([2.0, -2.0, 0.5])).tolist() == [32767, -32767, 16383]
