"""grace/input/stt: selection logic, router fallbacks, the engine contract, Groq without network."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np
import pytest

from grace.core.quota import Limits, QuotaGovernor
from grace.input.audio import frames, load_wav
from grace.input.stt import (
    VOICE_CLOUD,
    VOICE_LOCAL,
    VOICE_OFFLINE,
    SttRouter,
    build_stt_router,
    moonshine_installed,
    select_engines,
    voice_status,
)
from grace.input.stt.base import (
    BatchEngine,
    FallbackEngine,
    Final,
    Partial,
    SttError,
    UtteranceCancelled,
    chain,
    transcribe,
)
from grace.input.stt.fake import FakeStt
from grace.input.stt.groq import GROQ_TRANSCRIBE_URL, GroqStt
from grace.models import MODELS_DIR, moonshine_dir

FIX = Path(__file__).resolve().parent / "fixtures" / "audio"
FAKE_KEY = "gsk" + "_" + "T3st" * 12  # built at runtime: the secret scan never sees a literal


# --------------------------------------------------------------------------- selection (pure)
# Cloud-first (S06b). G = groq usable (key set, governor allows, not cooling down); the extras
# are moonshine ([stt-local]) and whisper_local ([stt-whisper]).
ALL = {"groq_ready": True, "whisper_installed": True, "moonshine_installed": True}
NO_EXTRAS = {"whisper_installed": False, "moonshine_installed": False}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # commands: groq, then moonshine, then whisper_local
        ({"language": "en", "dictation": False, **ALL}, ["groq", "moonshine", "whisper_local"]),
        ({"language": "en", "dictation": False, "groq_ready": True}, ["groq", "moonshine"]),
        # dictation and Hindi / auto: groq, then whisper_local, then moonshine for English only
        ({"language": "en", "dictation": True, **ALL}, ["groq", "whisper_local", "moonshine"]),
        ({"language": "hi", "dictation": False, **ALL}, ["groq", "whisper_local"]),
        ({"language": "auto", "dictation": True, **ALL}, ["groq", "whisper_local"]),
        ({"language": "hi", "dictation": False, "groq_ready": True}, ["groq"]),
        # key missing, cooling down or governor exhausted all arrive as groq_ready=False
        ({"language": "en", "dictation": False}, ["moonshine"]),
        ({"language": "en", "dictation": True, "whisper_installed": True},
         ["whisper_local", "moonshine"]),
        ({"language": "hi", "dictation": False, "whisper_installed": True}, ["whisper_local"]),
        # [stt-local] not installed
        ({"language": "en", "dictation": False, "groq_ready": True, **NO_EXTRAS}, ["groq"]),
        ({"language": "en", "dictation": False, "groq_ready": True, "whisper_installed": True,
          "moonshine_installed": False}, ["groq", "whisper_local"]),
        # nothing available: no key / offline and no extra
        ({"language": "en", "dictation": False, **NO_EXTRAS}, []),
        ({"language": "hi", "dictation": False}, []),  # Hindi offline needs [stt-whisper]
        # demo mode: moonshine first (no network on stage), then groq
        ({"language": "en", "dictation": False, "demo_mode": True, **ALL},
         ["moonshine", "groq", "whisper_local"]),
        ({"language": "en", "dictation": True, "demo_mode": True, "groq_ready": True},
         ["moonshine", "groq"]),
        ({"language": "en", "dictation": False, "demo_mode": True, "groq_ready": True,
          "moonshine_installed": False}, ["groq"]),
        ({"language": "hi", "dictation": False, "demo_mode": True, **ALL},
         ["groq", "whisper_local"]),  # Moonshine is English only
        # wake-word checks prefer what stays on the machine
        ({"language": "en", "dictation": False, "prefer_local": True, **ALL},
         ["moonshine", "whisper_local", "groq"]),
        ({"language": "en", "dictation": False, "prefer_local": True, "groq_ready": True,
          **NO_EXTRAS}, ["groq"]),
        # stt.engine set explicitly
        ({"language": "hi", "dictation": False, "engine": "moonshine"}, ["moonshine"]),
        ({"language": "en", "dictation": False, "engine": "moonshine", **NO_EXTRAS}, []),
        ({"language": "en", "dictation": False, "engine": "groq", **ALL},
         ["groq", "whisper_local", "moonshine"]),
        ({"language": "en", "dictation": False, "engine": "groq", "whisper_installed": True},
         ["whisper_local", "moonshine"]),  # explicit groq still respects cooldown and quota
        ({"language": "en", "dictation": False, "engine": "whisper_local"}, ["moonshine"]),
        # dictation_accurate no longer changes anything
        ({"language": "en", "dictation": True, "dictation_accurate": True, **ALL},
         ["groq", "whisper_local", "moonshine"]),
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


def test_router_uses_moonshine_for_english_without_groq() -> None:
    r = router({"moonshine": FakeStt(["hi"], name="moonshine")})
    engine = r.engine_for(dictation=False)
    assert engine is not None and engine.name == "moonshine"
    assert r.preload() == "moonshine" and r.status == VOICE_LOCAL


def test_router_is_cloud_first_with_a_key() -> None:
    engines = {"groq": FakeStt(["hi"], name="groq"), "moonshine": FakeStt(name="moonshine")}
    r = router(engines, groq_key_set=True)
    assert r.preload() == "groq>moonshine" and r.status == VOICE_CLOUD


def test_router_demo_mode_prefers_moonshine() -> None:
    engines = {"groq": FakeStt(name="groq"), "moonshine": FakeStt(name="moonshine")}
    r = router(engines, groq_key_set=True, demo_mode=True)
    assert r.names_for(False) == ["moonshine", "groq"]


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
    desc = r.preload()
    assert desc.startswith(VOICE_OFFLINE) and "model missing" in desc
    assert r.status == VOICE_OFFLINE and not r.can_recover()


def test_router_without_the_extra_never_builds_moonshine() -> None:
    built: list[str] = []
    r = SttRouter(
        {"moonshine": lambda: built.append("moonshine") or FakeStt(name="moonshine")},
        moonshine_installed=False,
    )
    assert r.engine_for() is None and built == []
    assert any("stt-local" in why for why in r.reasons())


def test_router_skips_groq_while_the_governor_says_no() -> None:
    quota = QuotaGovernor({"groq:w": Limits(rpd=2)}, state_path=None, clock=lambda: 1.8e9)
    r = router(
        {"groq": FakeStt(name="groq"), "moonshine": FakeStt(name="moonshine")},
        groq_key_set=True,
        quota=quota,
        quota_key="groq:w",
    )
    assert r.names_for(False) == ["groq", "moonshine"]
    quota.record("groq:w", requests=2)
    assert r.names_for(False) == ["moonshine"]
    assert "groq: free speech-to-text limit reached" in r.reasons()
    assert r.can_recover()  # it comes back when the day resets


def test_router_checks_room_for_one_minimum_billed_request() -> None:
    quota = QuotaGovernor({"groq:w": Limits(audio_s_per_hour=25)}, state_path=None,
                          clock=lambda: 1.8e9)  # fmt: skip
    r = router({"groq": FakeStt(name="groq")}, groq_key_set=True, quota=quota,
               quota_key="groq:w", min_billed_s=10)  # fmt: skip
    quota.record("groq:w", audio_s=10)
    assert r.groq_ready()  # 10 + 10 <= 25
    quota.record("groq:w", audio_s=10)
    assert not r.groq_ready()  # 20 + 10 > 25


def test_a_lone_groq_failure_cools_it_down() -> None:
    """Regression (S06b): with Groq the only engine, its error never reached the cooldown, so
    every utterance waited for Groq's timeout again."""
    clock = Clock()
    groq = FakeStt(name="groq", fail="ConnectError")
    r = SttRouter({"groq": lambda: groq}, groq_key_set=True, moonshine_installed=False,
                  clock=clock)  # fmt: skip
    groq.fail = None
    engine = r.engine_for()
    groq.fail = "ConnectError"
    with pytest.raises(SttError):
        transcribe(engine, chunks())
    assert r.engine_for() is None and r.can_recover()
    clock.now = 61
    assert r.names_for(False) == ["groq"]


def test_status_lines_change_only_when_the_mode_changes() -> None:
    seen: list[str] = []
    r = router({"groq": FakeStt(name="groq")}, groq_key_set=True, on_status=seen.append)
    r.preload()  # sets the first line without announcing it
    assert r.report("groq") is None
    assert r.report("moonshine") == VOICE_LOCAL
    assert r.report("moonshine") is None
    assert r.report(None, "offline") == VOICE_OFFLINE
    assert r.report("groq") == VOICE_CLOUD
    assert seen == [VOICE_LOCAL, VOICE_OFFLINE, VOICE_CLOUD]


@pytest.mark.parametrize(
    ("engine", "kw", "line"),
    [
        ("groq", {}, VOICE_CLOUD),
        ("groq>moonshine", {}, VOICE_CLOUD),
        ("moonshine", {}, VOICE_LOCAL),
        ("whisper_local", {}, VOICE_LOCAL),
        ("moonshine", {"demo_mode": True}, "Voice: local mode (demo)"),
        ("moonshine", {"setting": "moonshine"}, "Voice: local mode"),
        (None, {}, VOICE_OFFLINE),
    ],
)
def test_voice_status_is_pure(engine: str | None, kw: dict, line: str) -> None:
    assert voice_status(engine, **kw) == line


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

    r = build_stt_router(Settings())  # no key in a bare Settings()
    expected = ["moonshine"] if moonshine_installed() else []
    assert r.names_for(False) == expected and r.unavailable() == {}
    assert r._engines == {}  # lazy


def test_build_router_is_cloud_first_and_demo_mode_prefers_local(tmp_path: Path) -> None:
    from grace.config import load_settings

    env = tmp_path / ".env"
    env.write_text(f"GROQ_API_KEY={FAKE_KEY}\n")
    s = load_settings(tmp_path / "none.yaml", env, environ={})
    r = build_stt_router(s)
    assert r.names_for(False)[0] == "groq" and r.quota_key == "groq:whisper-large-v3-turbo"
    cfg = tmp_path / "s.yaml"
    cfg.write_text("demo_mode: true\n")
    demo = build_stt_router(load_settings(cfg, env, environ={}))
    assert demo.demo_mode and demo.groq_key_set  # S07: demo mode no longer turns Groq off
    if moonshine_installed():
        assert demo.names_for(False) == ["moonshine", "groq"]


def test_default_install_imports_nothing_from_moonshine() -> None:
    """The STT, voice and runtime modules (and building the router) never import moonshine."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import grace.input.stt, grace.input.voice, grace.runtime, grace.doctor\n"
        "from grace.config import Settings\n"
        "from grace.input.stt import build_stt_router\n"
        "build_stt_router(Settings()).names_for(False)\n"
        "bad = sorted(m for m in sys.modules if m.startswith('moonshine'))\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                          timeout=60)  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr


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


def test_the_last_engine_of_a_chain_reports_its_failure_too() -> None:
    failures: list[str] = []
    engine = chain([FakeStt(fail="a", name="groq"), FakeStt(fail="b", name="moonshine")],
                   lambda name, err: failures.append(name))  # fmt: skip
    with pytest.raises(SttError):
        transcribe(engine, iter(chunks()))
    assert failures == ["groq", "moonshine"]


def test_a_cancel_passes_through_engines_and_fallbacks() -> None:
    """UtteranceCancelled is not an SttError: no fallback replays a cancelled utterance."""
    fallback = FakeStt(["never"], name="groq", uploads=True)

    def cancelled():
        yield chunks(1)[0]
        raise UtteranceCancelled("mic off")

    engine = chain([FakeStt(fail="down", name="moonshine"), fallback], lambda n, e: None)
    assert engine.uploads
    with pytest.raises(UtteranceCancelled):
        transcribe(engine, cancelled())
    assert fallback.utterances == []


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


def test_groq_counts_billed_audio_seconds_in_the_governor() -> None:
    quota = QuotaGovernor({"groq:w": Limits(rpd=10)}, state_path=None, clock=lambda: 1.8e9)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "ok"})

    engine = groq_with(handler, quota=quota, quota_key="groq:w", min_billed_s=10.0)
    assert engine.uploads
    transcribe(engine, iter(chunks()))  # 0.128 s of audio, billed as 10 s
    long = [np.zeros(16000, dtype=np.float32)] * 12  # 12 s
    transcribe(engine, iter(long))
    used = quota.used("groq:w")
    assert used["rpd"] == 2 and used["asd"] == pytest.approx(22.0)


def test_groq_refuses_to_send_when_the_quota_is_spent() -> None:
    sent: list[int] = []
    quota = QuotaGovernor({"groq:w": Limits(audio_s_per_day=15)}, state_path=None,
                          clock=lambda: 1.8e9)  # fmt: skip
    quota.record("groq:w", audio_s=10)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(1)
        return httpx.Response(200, json={"text": "ok"})

    with pytest.raises(SttError, match="limit"):
        transcribe(groq_with(handler, quota=quota, quota_key="groq:w"), iter(chunks()))
    assert sent == []


def test_a_429_is_not_counted() -> None:
    quota = QuotaGovernor({}, state_path=None, clock=lambda: 1.8e9)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate"})

    with pytest.raises(SttError):
        transcribe(groq_with(handler, quota=quota, quota_key="groq:w"), iter(chunks()))
    assert quota.used("groq:w")["rpd"] == 0


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
    not (moonshine_installed() and moonshine_dir("small-streaming").is_dir()),
    reason="needs uv sync --extra stt-local and scripts/fetch_models.py --stt-local",
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
