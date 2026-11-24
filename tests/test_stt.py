"""grace/input/stt: selection logic, router fallbacks, the engine contract, Groq without network."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np
import pytest

from grace.input.audio import frames, load_wav
from grace.input.stt import SttRouter, build_stt_router, select_engines
from grace.input.stt.base import (
    BatchEngine,
    FallbackEngine,
    Final,
    Partial,
    SttError,
    chain,
    transcribe,
)
from grace.input.stt.fake import FakeStt
from grace.input.stt.groq import GROQ_TRANSCRIBE_URL, GroqStt
from grace.models import MODELS_DIR, moonshine_dir

FIX = Path(__file__).resolve().parent / "fixtures" / "audio"
FAKE_KEY = "gsk" + "_" + "T3st" * 12  # built at runtime: the secret scan never sees a literal


# --------------------------------------------------------------------------- selection (pure)
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"language": "en", "dictation": False}, ["moonshine"]),
        ({"language": "en", "dictation": True}, ["moonshine"]),
        ({"language": "en", "dictation": False, "dictation_accurate": True}, ["moonshine"]),
        (
            {"language": "en", "dictation": True, "dictation_accurate": True, "groq_ready": True,
             "whisper_installed": True},
            ["groq", "whisper_local", "moonshine"],
        ),
        (
            {"language": "en", "dictation": True, "dictation_accurate": True},
            ["moonshine"],  # offline, no extra: English still works
        ),
        (
            {"language": "hi", "dictation": False, "groq_ready": True, "whisper_installed": True},
            ["groq", "whisper_local"],
        ),
        ({"language": "hi", "dictation": False, "whisper_installed": True}, ["whisper_local"]),
        ({"language": "hi", "dictation": False, "groq_ready": True}, ["groq"]),
        ({"language": "hi", "dictation": False}, []),  # Hindi offline needs the extra
        ({"language": "auto", "dictation": False, "whisper_installed": True}, ["whisper_local"]),
        ({"language": "hi", "dictation": False, "engine": "moonshine"}, ["moonshine"]),
        (
            {"language": "en", "dictation": False, "engine": "groq", "whisper_installed": True},
            ["groq", "whisper_local", "moonshine"],
        ),
        ({"language": "en", "dictation": False, "engine": "whisper_local"}, ["moonshine"]),
    ],
)  # fmt: skip
def test_select_engines(kwargs: dict, expected: list[str]) -> None:
    assert select_engines(**kwargs) == expected


# --------------------------------------------------------------------------- router
class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def router(engines: dict[str, FakeStt], **kw) -> SttRouter:
    factories: dict[str, Callable] = {name: (lambda e=e: e) for name, e in engines.items()}
    return SttRouter(factories, **kw)


def chunks(n: int = 4) -> list[np.ndarray]:
    return [np.full(512, 0.1, dtype=np.float32) for _ in range(n)]


def test_router_uses_moonshine_for_english() -> None:
    r = router({"moonshine": FakeStt(["hi"], name="moonshine")})
    engine = r.engine_for(dictation=False)
    assert engine is not None and engine.name == "moonshine"
    assert r.preload() == "moonshine"


def test_router_skips_an_engine_that_cannot_load() -> None:
    r = router(
        {
            "whisper_local": FakeStt(name="whisper_local", fail="not installed"),
            "moonshine": FakeStt(["ok"], name="moonshine"),
        },
        dictation_accurate=True,
        whisper_installed=True,
    )
    engine = r.engine_for(dictation=True)
    assert engine is not None and engine.name == "moonshine"
    assert r.unavailable() == {"whisper_local": "not installed"}


def test_router_reports_when_nothing_can_transcribe() -> None:
    r = router({"moonshine": FakeStt(name="moonshine", fail="model missing")})
    assert r.engine_for() is None
    assert "model missing" in r.preload()


def test_groq_failure_falls_back_and_cools_down() -> None:
    clock = Clock()
    groq = FakeStt(name="groq")
    whisper = FakeStt(["namaste", "dobara"], name="whisper_local")
    r = SttRouter(
        {"groq": lambda: groq, "whisper_local": lambda: whisper},
        language="hi",
        groq_key_set=True,
        whisper_installed=True,
        cooldown_s=60,
        clock=clock,
    )
    engine = r.engine_for()
    assert engine is not None and engine.name == "groq>whisper_local"
    groq.fail = "HTTP 503"  # loaded fine; now the request fails
    final, _ = transcribe(engine, chunks())
    assert (final.text, final.engine, final.fallback_from) == ("namaste", "whisper_local", "groq")
    assert r.names_for(False) == ["whisper_local"]  # groq skipped while cooling down
    clock.now = 61
    assert r.names_for(False) == ["groq", "whisper_local"]


def test_build_router_from_settings_loads_nothing() -> None:
    from grace.config import Settings

    r = build_stt_router(Settings())
    assert r.names_for(False) == ["moonshine"] and r.unavailable() == {}
    assert r._engines == {}  # lazy


# --------------------------------------------------------------------------- engine contract
class EchoBatch(BatchEngine):
    name = "echo"

    def _transcribe(self, audio: np.ndarray) -> str:
        return f"{len(audio)} samples"


def groq_ok(texts: list[str]) -> GroqStt:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": texts.pop(0) if texts else ""})

    return GroqStt(FAKE_KEY, client=httpx.Client(transport=httpx.MockTransport(handler)))


ENGINES: dict[str, Callable[[], object]] = {
    "fake": lambda: FakeStt(["hello there"], partial_every=2),
    "batch": EchoBatch,
    "groq": lambda: groq_ok(["hello there"]),
    "fallback": lambda: FallbackEngine(FakeStt(fail="down"), FakeStt(["hello there"])),
}


@pytest.mark.parametrize("make", ENGINES.values(), ids=ENGINES.keys())
def test_engine_contract(make: Callable[[], object]) -> None:
    engine = make()
    engine.load()
    feed = chunks(6)
    events = list(engine.transcribe_stream(iter(feed)))
    finals = [e for e in events if isinstance(e, Final)]
    assert len(finals) == 1 and events[-1] is finals[0]
    assert all(isinstance(e, Partial) for e in events[:-1])
    final = finals[0]
    assert final.audio_s == pytest.approx(6 * 512 / 16000)
    assert 0 < final.t_first_audio <= final.t_end_of_audio <= final.t_final
    assert final.latency_ms >= 0 and final.text


@pytest.mark.parametrize("make", ENGINES.values(), ids=ENGINES.keys())
def test_engine_contract_empty_audio(make: Callable[[], object]) -> None:
    engine = make()
    final, _ = transcribe(engine, iter([]))
    assert final.audio_s == 0


def test_fallback_replays_all_audio_and_reports_the_failure() -> None:
    failures: list[str] = []
    primary = FakeStt(fail="offline", name="groq")
    fallback = FakeStt(["ok"], name="whisper_local")
    engine = chain([primary, fallback], lambda name, err: failures.append(f"{name}: {err}"))
    final, _ = transcribe(engine, iter(chunks(5)))
    assert len(fallback.utterances[0]) == 5 * 512
    assert final.fallback_from == "groq" and failures == ["groq: offline"]


def test_chain_of_one_is_the_engine() -> None:
    fake = FakeStt()
    assert chain([fake]) is fake
    with pytest.raises(ValueError):
        chain([])


# --------------------------------------------------------------------------- groq (no network)
def groq_with(handler: Callable[[httpx.Request], httpx.Response], **kw) -> GroqStt:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GroqStt(FAKE_KEY, client=client, sleep=lambda s: None, **kw)


def test_groq_sends_audio_model_and_language() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": " namaste "})

    final, _ = transcribe(groq_with(handler, language="hi"), iter(chunks()))
    assert final.text == "namaste"
    (req,) = seen
    assert str(req.url) == GROQ_TRANSCRIBE_URL
    assert req.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = req.read()
    assert b"whisper-large-v3-turbo" in body and b'name="language"' in body and b"RIFF" in body


def test_groq_retries_server_errors_then_succeeds() -> None:
    codes = [503, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(codes.pop(0), json={"text": "ok"})

    final, _ = transcribe(groq_with(handler, max_retries=1), iter(chunks()))
    assert final.text == "ok" and codes == []


@pytest.mark.parametrize(("status", "calls"), [(429, 1), (401, 1), (500, 2)])
def test_groq_errors_raise_without_the_key(status: int, calls: int) -> None:
    count = []

    def handler(request: httpx.Request) -> httpx.Response:
        count.append(1)
        return httpx.Response(status, json={"error": "x"})

    with pytest.raises(SttError) as err:
        transcribe(groq_with(handler, max_retries=1), iter(chunks()))
    assert len(count) == calls and FAKE_KEY not in str(err.value)


def test_groq_network_error_is_retried_then_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(SttError, match="ConnectError"):
        transcribe(groq_with(handler, max_retries=2), iter(chunks()))


def test_groq_without_a_key_cannot_load() -> None:
    with pytest.raises(SttError, match="GROQ_API_KEY"):
        GroqStt(None).load()


# --------------------------------------------------------------------------- local engines
def test_whisper_local_without_the_package_is_an_stt_error(monkeypatch) -> None:
    import sys

    from grace.input.stt.whisper_local import WhisperLocalStt

    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(SttError, match="stt-whisper"):
        WhisperLocalStt(download_root=MODELS_DIR / "whisper").load()


def test_moonshine_without_its_model_is_an_stt_error(tmp_path: Path) -> None:
    from grace.input.stt.moonshine import MoonshineStt

    with pytest.raises(SttError, match="fetch_models"):
        MoonshineStt(tmp_path / "missing", 4).load()


@pytest.mark.skipif(
    not moonshine_dir("small-streaming").is_dir(), reason="run scripts/fetch_models.py"
)
def test_moonshine_transcribes_a_command_with_partials() -> None:
    from grace.input.stt.moonshine import MoonshineStt

    engine = MoonshineStt(moonshine_dir("small-streaming"), 4, update_interval=0.3)
    final, partials = transcribe(engine, frames(load_wav(FIX / "one_command.wav")))
    assert "notepad" in final.text.lower()
    assert partials and final.partials == len(partials)
    assert final.latency_ms < 1000  # ~100 ms on the base machine; generous for slower CPUs


def _whisper_ready() -> bool:
    from grace.input.stt import whisper_installed

    return whisper_installed() and (MODELS_DIR / "whisper").is_dir()


@pytest.mark.skipif(not _whisper_ready(), reason="needs --extra stt-whisper and fetch --whisper")
def test_whisper_local_transcribes_a_command() -> None:
    from grace.input.stt.whisper_local import WhisperLocalStt

    engine = WhisperLocalStt("base", threads=4, language="en", download_root=MODELS_DIR / "whisper")
    final, partials = transcribe(engine, frames(load_wav(FIX / "one_command.wav")))
    assert "notepad" in final.text.lower() and partials == []
