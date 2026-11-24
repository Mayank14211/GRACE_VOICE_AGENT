"""grace/input/voice.py: wake word, follow-ups, push-to-talk, dictation, cancel, mic ownership.

The pipeline runs without threads or hardware: a fake VAD calls loud frames speech, a fake wake
detector fires on a marker frame, a manual clock advances 32 ms per frame, and each STT job runs
synchronously once its utterance has ended (FakeStt returns scripted text).
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from grace.input.audio import FRAME, FRAME_S, Segmenter
from grace.input.stt.fake import FakeStt
from grace.input.voice import (
    ListenPipeline,
    SerialWorker,
    SttWorker,
    VoiceWorker,
    split_wake_word,
    strip_leading_wake_word,
)
from grace.models import moonshine_dir

SPEECH = np.full(FRAME, 0.3, dtype=np.float32)
SILENCE = np.zeros(FRAME, dtype=np.float32)
WAKE = np.full(FRAME, 0.77, dtype=np.float32)


class FakeVad:
    def __init__(self) -> None:
        self.resets = 0

    def prob(self, frame: np.ndarray) -> float:
        return 0.9 if float(np.abs(frame).mean()) > 0.05 else 0.0

    def reset(self) -> None:
        self.resets += 1


class FakeWake:
    name = "fake_wake"

    def __init__(self) -> None:
        self.frames = 0

    def process(self, frame: np.ndarray) -> float | None:
        self.frames += 1
        return 0.9 if frame[0] == np.float32(0.77) else None

    def reset(self) -> None:
        pass


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Harness:
    def __init__(
        self,
        texts: list[str] | None = None,
        *,
        wake_mode: str = "text",
        follow_up_s: float = 10.0,
        command_max_s: float = 12.0,
        engine: FakeStt | None = None,
    ) -> None:
        self.clock = Clock()
        self.jobs: list = []
        self.commands: list[str] = []
        self.partials: list[str] = []
        self.listening: list[bool] = []
        self.wakes: list[float | None] = []
        self.problems: list[str] = []
        self.dictation = False
        self.muted = False
        self.dictation_cancels = 0
        self.vad = FakeVad()
        self.wake = FakeWake()
        self.engine = engine if engine is not None else FakeStt(texts or [], partial_every=3)
        self.p = ListenPipeline(
            vad=self.vad,
            segmenter=Segmenter(end_of_speech_ms=96, min_speech_ms=64, pre_roll_ms=64),
            submit=self.jobs.append,
            wake=self.wake if wake_mode == "model" else None,
            wake_mode=wake_mode,
            wake_word="listen",
            follow_up_s=follow_up_s,
            command_max_s=command_max_s,
            dictation_max_s=30.0,
            ptt_pre_roll_ms=96,
            is_dictation=lambda: self.dictation,
            is_muted=lambda: self.muted,
            on_listening=self.listening.append,
            on_wake=self.wakes.append,
            on_dictation_cancelled=self._cancelled_dictation,
            clock=self.clock,
        )
        self.stt = SttWorker(
            lambda dictation: self.engine,
            lambda: self.p,
            on_partial=lambda u, t: self.partials.append(t),
            on_command=lambda u, t: self.commands.append(t),
            on_unavailable=self.problems.append,
        )

    def _cancelled_dictation(self) -> None:
        self.dictation_cancels += 1

    def feed(self, frame: np.ndarray, n: int = 1) -> None:
        for _ in range(n):
            self.clock.now += FRAME_S
            self.p.feed(frame)

    def run_stt(self) -> None:
        while self.jobs:
            self.stt.run(self.jobs.pop(0))

    def say(self, frames: int = 10) -> None:
        """One utterance: speech, then enough silence to end it, then STT."""
        self.feed(SPEECH, frames)
        self.feed(SILENCE, 5)
        self.run_stt()

    def wait(self, seconds: float) -> None:
        self.feed(SILENCE, int(seconds / FRAME_S) + 1)


# --------------------------------------------------------------------------- pure text helpers
def test_split_wake_word_is_pure() -> None:
    assert split_wake_word("Listen, open notepad", "listen") == "open notepad"
    assert split_wake_word("listen", "listen") == ""
    assert split_wake_word("I was listening to music", "listen") is None
    assert split_wake_word("hey LISTEN type Hi", "listen") == "type Hi"
    assert split_wake_word(None, "listen") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hey Grace, open notepad.", "open notepad."),
        ("grace open notepad", "open notepad"),
        ("OK grace volume 40", "volume 40"),
        ("open Grace's folder", "open Grace's folder"),
        ("open notepad", "open notepad"),
    ],
)
def test_strip_leading_wake_word(text: str, expected: str) -> None:
    assert strip_leading_wake_word(text, "grace") == expected


# --------------------------------------------------------------------------- spoken wake word
def test_speech_without_the_wake_word_does_nothing() -> None:
    h = Harness(["what a nice day"])
    h.say()
    assert h.commands == [] and h.p.phase == "idle" and h.listening == []


def test_wake_word_and_command_in_one_breath() -> None:
    h = Harness(["Listen, open notepad", "volume 30"])
    h.say()
    assert h.commands == ["open notepad"] and h.wakes == [None]
    assert h.p.phase == "active" and h.listening == [True]
    h.say()  # a follow-up needs no wake word
    assert h.commands == ["open notepad", "volume 30"]


def test_a_command_started_before_the_wake_word_transcript_arrives_is_still_a_command() -> None:
    """Regression: "listen" ... short pause ... "volume 30". The second utterance starts while the
    first is still being transcribed, so it is captured as a wake check; it must still run."""
    h = Harness(["listen", "volume 30"])
    h.feed(SPEECH, 10)
    h.feed(SILENCE, 5)  # first utterance ends; its transcript is still pending
    h.feed(SPEECH, 10)
    h.feed(SILENCE, 5)  # second utterance ends too
    assert [j.kind for j in h.jobs] == ["wake_check", "wake_check"] and h.p.phase == "idle"
    h.run_stt()
    assert h.commands == ["volume 30"] and h.p.phase == "active"


def test_a_stray_second_utterance_without_the_wake_word_is_still_ignored() -> None:
    h = Harness(["what a nice day", "volume 30"])
    h.feed(SPEECH, 10)
    h.feed(SILENCE, 5)
    h.feed(SPEECH, 10)
    h.feed(SILENCE, 5)
    h.run_stt()
    assert h.commands == [] and h.p.phase == "idle"


def test_wake_word_alone_opens_the_window() -> None:
    h = Harness(["listen", "volume 30"])
    h.say()
    assert h.commands == [] and h.p.phase == "active"
    h.say()
    assert h.commands == ["volume 30"]


def test_the_window_closes_after_follow_up_s() -> None:
    h = Harness(["listen volume 30", "volume 40"], follow_up_s=2.0)
    h.say()
    h.wait(2.1)
    assert h.p.phase == "idle" and h.listening == [True, False]
    h.say()  # no wake word: nothing
    assert h.commands == ["volume 30"]


def test_a_pause_shorter_than_the_window_keeps_it_open() -> None:
    h = Harness(["listen volume 30", "volume 40"], follow_up_s=2.0)
    h.say()
    h.wait(1.5)
    h.say()
    assert h.commands == ["volume 30", "volume 40"]


def test_follow_up_zero_means_one_command_per_wake_word() -> None:
    h = Harness(["listen open notepad", "type hi"], follow_up_s=0)
    h.say()
    assert h.p.phase == "idle"
    h.say()
    assert h.commands == ["open notepad"]


def test_follow_up_zero_still_waits_for_the_first_command() -> None:
    h = Harness(["listen", "open notepad"], follow_up_s=0)
    h.say()
    h.wait(3.0)  # under FIRST_COMMAND_S
    h.say()
    assert h.commands == ["open notepad"] and h.p.phase == "idle"


def test_stop_listening_closes_the_window() -> None:
    h = Harness(["listen volume 30", "volume 40"])
    h.say()
    h.p.end_session()
    h.say()
    assert h.commands == ["volume 30"] and h.listening == [True, False]


def test_partials_are_shown_for_commands_but_not_for_wake_checks() -> None:
    h = Harness(["random chatter", "listen", "open notepad"])
    h.say()
    assert h.partials == []  # chatter while idle is never shown
    h.say()
    h.say()
    assert h.partials and h.commands == ["open notepad"]


def test_the_window_stays_open_while_a_command_is_transcribed() -> None:
    h = Harness(["listen volume 30", "volume 40"], follow_up_s=1.0)
    h.say()
    h.feed(SPEECH, 10)
    h.feed(SILENCE, 5)  # utterance ended, STT not run yet
    h.wait(2.0)
    assert h.p.phase == "active"
    h.run_stt()
    assert h.commands == ["volume 30", "volume 40"]


# --------------------------------------------------------------------------- wake-word model
def test_model_mode_runs_only_the_detector_while_idle() -> None:
    h = Harness(["volume 30"], wake_mode="model")
    h.feed(SPEECH, 10)
    h.feed(SILENCE, 5)
    assert h.jobs == [] and h.vad.resets == 0 and h.wake.frames == 15
    h.feed(WAKE)
    assert h.wakes == [0.9] and h.p.phase == "active" and h.listening == [True]
    h.say()
    assert h.commands == ["volume 30"]


def test_model_mode_drops_a_leftover_wake_word() -> None:
    h = Harness(["listen, volume 30"], wake_mode="model")
    h.feed(WAKE)
    h.say()
    assert h.commands == ["volume 30"]


def test_no_wake_word_mode_takes_every_utterance() -> None:
    h = Harness(["volume 30", "volume 40"], wake_mode="off")
    h.say()
    h.wait(30)
    h.say()
    assert h.commands == ["volume 30", "volume 40"] and h.p.phase == "active"


# --------------------------------------------------------------------------- cancel (mic off)
def test_cancel_during_an_utterance_drops_it() -> None:
    h = Harness(["listen volume 30", "volume 40"])
    h.say()
    h.feed(SPEECH, 10)
    h.p.cancel()
    h.feed(SILENCE, 5)
    h.run_stt()
    assert h.commands == ["volume 30"] and h.p.phase == "idle"
    assert h.listening == [True, False]


def test_cancel_after_the_utterance_ended_drops_the_late_text() -> None:
    h = Harness(["listen volume 30"])
    h.feed(SPEECH, 10)
    h.feed(SILENCE, 5)
    h.p.cancel()  # mic off while STT is still working
    h.run_stt()
    assert h.commands == []


def test_cancel_ends_dictation() -> None:
    h = Harness(["listen start dictation"])
    h.say()
    h.dictation = True
    h.p.cancel()
    assert h.dictation_cancels == 1


# --------------------------------------------------------------------------- push-to-talk
def test_push_to_talk_needs_no_wake_word_and_ends_on_release() -> None:
    h = Harness(["volume 25"])
    h.feed(SILENCE, 5)
    h.p.ptt_down()
    assert h.p.wants_audio() and h.listening == [True]
    h.feed(SPEECH, 10)
    h.feed(SILENCE, 20)  # pauses don't end it: only the release does
    assert len(h.jobs) == 1
    h.p.ptt_up()
    h.run_stt()
    assert h.commands == ["volume 25"] and not h.p.wants_audio()
    assert h.listening == [True, False]
    (utt_audio,) = h.engine.utterances
    assert len(utt_audio) == (3 + 30) * FRAME  # 96 ms pre-roll + everything while held


def test_push_to_talk_strips_a_wake_word_said_anyway() -> None:
    h = Harness(["listen, volume 35"])
    h.p.ptt_down()
    h.feed(SPEECH, 10)
    h.p.ptt_up()
    h.run_stt()
    assert h.commands == ["volume 35"]


def test_push_to_talk_takes_over_an_utterance_in_progress() -> None:
    # The cancelled capture is still transcribed (its text is thrown away), then the key's.
    h = Harness(["listen volume 99", "volume 15"])
    h.feed(SPEECH, 2)  # VAD opened a wake check
    h.p.ptt_down()
    h.feed(SPEECH, 8)
    h.p.ptt_up()
    h.run_stt()
    assert h.commands == ["volume 15"]


def test_push_to_talk_silence_does_nothing() -> None:
    h = Harness([""])
    h.p.ptt_down()
    h.feed(SILENCE, 10)
    h.p.ptt_up()
    h.run_stt()
    assert h.commands == []


def test_cancel_releases_push_to_talk() -> None:
    h = Harness(["volume 25"])
    h.p.ptt_down()
    h.feed(SPEECH, 5)
    h.p.cancel()
    h.run_stt()
    assert h.commands == [] and not h.p.ptt_held and not h.p.wants_audio()


# --------------------------------------------------------------------------- dictation, limits
def test_dictation_utterances_and_window() -> None:
    h = Harness(["listen", "Dear team,", "stop dictation"], follow_up_s=1.0)
    h.say()
    h.dictation = True
    h.wait(5.0)  # dictation keeps the window open
    assert h.p.phase == "active"
    h.say()
    assert h.jobs == [] and h.commands == ["Dear team,"]
    assert h.p.seg.max_s == 30.0


def test_dictation_keeps_the_leading_word() -> None:
    h = Harness(["listen", "Listen to this, team."])
    h.say()
    h.dictation = True
    h.say()
    assert h.commands == ["Listen to this, team."]


def test_a_command_is_cut_at_max_length() -> None:
    h = Harness(["listen", "a very long command", "and the rest"], command_max_s=0.5)
    h.say()
    h.feed(SPEECH, 20)  # 0.64 s: cut at 0.5 s; the speech after it is a new utterance
    h.feed(SILENCE, 5)
    h.run_stt()
    assert h.commands == ["a very long command", "and the rest"]
    assert len(h.engine.utterances[1]) <= int(0.5 / FRAME_S + 3) * FRAME


def test_a_short_blip_is_ignored() -> None:
    h = Harness(["listen", "click"])
    h.say()
    h.feed(SPEECH, 1)  # 32 ms: under min_speech_ms
    h.feed(SILENCE, 5)
    h.run_stt()
    assert h.commands == []


def test_grace_talking_is_not_heard() -> None:
    h = Harness(["listen volume 30"])
    h.muted = True
    h.feed(SPEECH, 10)
    h.muted = False
    h.feed(SPEECH, 3)  # echo tail (0.25 s) still ignored
    h.feed(SILENCE, 5)
    assert h.jobs == []
    h.say()
    assert h.commands == ["volume 30"]


def test_no_engine_reports_a_problem_and_frees_the_window() -> None:
    h = Harness(["listen volume 30"])
    h.stt = SttWorker(lambda d: None, lambda: h.p, on_unavailable=h.problems.append)
    h.say()
    assert h.problems == ["no speech-to-text engine is available"] and h.p._pending == 0


def test_an_stt_error_is_reported_not_raised() -> None:
    h = Harness(engine=FakeStt(fail="model broke"))
    h.say()
    assert h.problems == ["model broke"] and h.p._pending == 0


# --------------------------------------------------------------------------- the voice thread
class FakeSource:
    def __init__(self, frames: list[np.ndarray] | None = None, fail: str | None = None) -> None:
        self.frames = list(frames or [])
        self.fail = fail
        self.opened = 0
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def start(self) -> FakeSource:
        if self.fail:
            raise OSError(self.fail)
        self.opened += 1
        self._open = True
        return self

    def stop(self) -> None:
        self._open = False

    def read(self, timeout: float) -> np.ndarray | None:
        return self.frames.pop(0) if self.frames else None


def worker_for(h: Harness, source: FakeSource, mic: list[bool]) -> tuple[VoiceWorker, list[str]]:
    errors: list[str] = []
    w = VoiceWorker(h.p, source, mic_on=lambda: mic[0], on_mic_error=errors.append, retry_s=60)
    return w, errors


def test_mic_opens_only_while_on() -> None:
    h = Harness()
    mic = [False]
    source = FakeSource([SPEECH] * 3)
    w, _ = worker_for(h, source, mic)
    w.wake()
    w.step()
    assert source.opened == 0
    mic[0] = True
    w.step()
    assert source.is_open and len(source.frames) == 2
    mic[0] = False
    w.wake()
    w.step()
    assert not source.is_open


def test_push_to_talk_opens_the_mic_while_it_is_off() -> None:
    h = Harness(["volume 25"])
    source = FakeSource([SPEECH] * 5)
    w, _ = worker_for(h, source, [False])
    w.request_ptt()
    for _ in range(6):
        w.step()
    w.release_ptt()
    w.step()
    assert not source.is_open
    h.run_stt()
    assert h.commands == ["volume 25"]


def test_push_to_talk_never_prepends_audio_from_before_the_mic_closed() -> None:
    h = Harness(["volume 25"])
    mic = [True]
    loud_then_quiet = [SPEECH] * 3  # heard while the mic was on, then it is switched off
    source = FakeSource(loud_then_quiet)
    w, _ = worker_for(h, source, mic)
    for _ in range(3):
        w.step()
    mic[0] = False
    w.wake()
    w.step()  # closes the mic
    w.request_ptt()  # minutes later, with the mic off
    source.frames = [SILENCE] * 4
    for _ in range(5):
        w.step()
    w.release_ptt()
    w.step()
    h.run_stt()
    (audio,) = h.engine.utterances[-1:]
    assert len(audio) == 4 * FRAME and not audio.any()  # only what came after the key


def test_a_missing_mic_is_reported_once_and_retried_later() -> None:
    h = Harness()
    source = FakeSource(fail="no input device")
    w, errors = worker_for(h, source, [True])
    w.step()
    w.step()
    assert errors == ["OSError: no input device"] and not source.is_open


@pytest.mark.timeout(10)
def test_threads_run_and_stop() -> None:
    h = Harness(["listen volume 30"])
    done = threading.Event()
    stt = SttWorker(
        lambda d: h.engine,
        lambda: h.p,
        on_command=lambda u, t: (h.commands.append(t), done.set()),
    )
    h.p._submit = stt.submit
    frames = [SPEECH] * 10 + [SILENCE] * 10
    source = FakeSource(frames)
    w = VoiceWorker(h.p, source, mic_on=lambda: True, stt=stt, intents=SerialWorker())
    w.start()
    assert done.wait(5)
    w.stop()
    assert h.commands == ["volume 30"] and not source.is_open


def test_serial_worker_survives_a_failing_job() -> None:
    worker = SerialWorker().start()
    ran = threading.Event()
    worker.submit(lambda: 1 / 0)
    worker.submit(ran.set)
    assert ran.wait(2)
    worker.stop()


# --------------------------------------------------------------------------- real models
@pytest.mark.skipif(
    not moonshine_dir("small-streaming").is_dir(), reason="run scripts/fetch_models.py"
)
def test_real_vad_and_moonshine_find_both_commands() -> None:
    from pathlib import Path

    from grace.input.audio import SileroVad, frames, load_wav
    from grace.input.stt.moonshine import MoonshineStt

    fix = Path(__file__).resolve().parent / "fixtures" / "audio"
    jobs: list = []
    commands: list[str] = []
    p = ListenPipeline(
        vad=SileroVad(fix / "silero_vad.onnx"),
        segmenter=Segmenter(),
        submit=jobs.append,
        wake_mode="off",
    )
    engine = MoonshineStt(moonshine_dir("small-streaming"), 4)
    stt = SttWorker(lambda d: engine, lambda: p, on_command=lambda u, t: commands.append(t))
    for frame in frames(load_wav(fix / "two_commands.wav")):
        p.feed(frame)
    assert len(jobs) == 2
    for job in jobs:
        stt.run(job)
    assert [c.lower().strip(".") for c in commands] == ["volume up", "take a screenshot"]


# --------------------------------------------------------------------------- metrics hooks
def _utterance_and_final(kind: str = "command"):
    from grace.input.stt.base import Final
    from grace.input.voice import Utterance

    utt = Utterance(kind, 0, started=1.0)  # type: ignore[arg-type]
    utt.eos_t, utt.hangover_s, utt.end_reason = 10.0, 0.5, "silence"
    final = Final("open notepad", "moonshine", 1.5, 8.0, 10.0, 10.12, partials=3)
    return utt, final


def test_stt_metrics_row_goes_through_a_real_metrics_log(tmp_path) -> None:
    """Regression: the row used the field name ``kind``, which MetricsLog.write already takes as
    the event name ("got multiple values for argument 'kind'"), so no voice command ever ran."""
    import json

    from grace.input.voice import stt_event_fields
    from grace.metrics.log import MetricsLog

    utt, final = _utterance_and_final()
    fields = stt_event_fields(utt, final)
    assert "kind" not in fields
    log = MetricsLog(tmp_path / "s.jsonl")
    log.write("stt", **fields)  # must not raise
    log.close()
    (row,) = [json.loads(line) for line in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert row["kind"] == "stt" and row["utterance"] == "command" and row["engine"] == "moonshine"
    assert row["eos_to_final_ms"] == pytest.approx(120.0, abs=0.1)
    assert row["speech_end_to_final_ms"] == pytest.approx(620.0, abs=0.1)
    assert "open notepad" not in json.dumps(row)  # never the transcript


def test_no_metrics_call_uses_the_reserved_kind_field() -> None:
    import ast
    from pathlib import Path

    offenders = []
    for path in (Path(__file__).resolve().parent.parent / "grace").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "write"
                and "metrics" in ast.unparse(node.func.value)
                and any(k.arg == "kind" for k in node.keywords)
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders


def test_a_failing_metrics_hook_does_not_drop_the_command() -> None:
    h = Harness(["listen volume 30"])

    def broken(utt, final) -> None:
        raise TypeError("metrics blew up")

    h.stt._on_final = broken
    h.stt._on_partial = broken
    h.say()
    assert h.commands == ["volume 30"] and h.p._pending == 0
