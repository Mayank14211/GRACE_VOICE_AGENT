"""Quota governor and provider cooldowns (grace/core). Pure, no network."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from grace.config import load_settings
from grace.core.cooldown import Cooldowns
from grace.core.quota import Limits, NullQuota, QuotaGovernor, build_governor

T0 = 1_800_000_000.0  # 2027-01-15 08:00 UTC


class Clock:
    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_requests_per_minute_reset_with_the_minute() -> None:
    clock = Clock(T0)
    q = QuotaGovernor({"groq:m": Limits(rpm=2)}, clock=clock)
    for _ in range(2):
        assert q.allow("groq:m")
        q.record("groq:m")
    assert not q.allow("groq:m")
    clock.now += 60
    assert q.allow("groq:m")


def test_daily_tokens_block_once_spent() -> None:
    clock = Clock()
    q = QuotaGovernor({"groq:m": Limits(tpd=1000)}, clock=clock)
    q.record("groq:m", tokens=999)
    assert q.allow("groq:m")  # token use is only known afterwards
    q.record("groq:m", tokens=1)
    assert not q.allow("groq:m")
    clock.now += 86_400
    assert q.allow("groq:m")


def test_units_per_day_and_month() -> None:
    clock = Clock()
    q = QuotaGovernor({"img": Limits(units_per_day=100), "search": Limits(units_per_month=2)},
                      clock=clock)  # fmt: skip
    q.record("img", units=57.6)
    assert q.allow("img", units=40) and not q.allow("img", units=57.6)
    q.record("search", units=1)
    q.record("search", units=1)
    assert not q.allow("search", units=1)
    clock.now += 40 * 86_400  # next month
    assert q.allow("search", units=1)


def test_unknown_keys_are_unlimited_but_still_counted() -> None:
    q = QuotaGovernor({}, clock=Clock())
    q.record("x:y", tokens=5)
    assert q.allow("x:y") and q.used("x:y")["tpd"] == 5
    assert NullQuota().allow("anything")


def test_warns_once_at_80_percent(caplog: pytest.LogCaptureFixture) -> None:
    warnings: list[tuple[str, str, float]] = []
    q = QuotaGovernor({"groq:m": Limits(rpd=10)}, clock=Clock(),
                      on_warning=lambda k, c, f: warnings.append((k, c, f)))  # fmt: skip
    with caplog.at_level(logging.WARNING, logger="grace.core.quota"):
        for _ in range(9):
            q.record("groq:m")
    assert warnings == [("groq:m", "rpd", 0.8)]
    assert sum("80%" in m for m in caplog.messages) == 1


def test_day_and_month_counts_survive_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "quota.json"
    clock = Clock()
    first = QuotaGovernor({"g": Limits(rpd=3, rpm=100)}, state_path=path, clock=clock)
    for _ in range(3):
        first.record("g")
    again = QuotaGovernor({"g": Limits(rpd=3, rpm=100)}, state_path=path, clock=clock)
    assert not again.allow("g") and again.used("g")["rpm"] == 0  # minutes are not persisted
    path.write_text("not json")
    assert QuotaGovernor({"g": Limits(rpd=3)}, state_path=path, clock=clock).allow("g")


# --------------------------------------------------------------------------- audio seconds (S06b)
WHISPER = "groq:whisper-large-v3-turbo"
FREE_WHISPER = Limits(rpm=20, rpd=2000, audio_s_per_hour=7200, audio_s_per_day=28_800)


def test_audio_seconds_per_hour_reset_with_the_hour() -> None:
    clock = Clock()
    q = QuotaGovernor({WHISPER: FREE_WHISPER}, clock=clock)
    for _ in range(3):  # 3 x 20 requests of 120 s = 7,200 s: the hour is spent
        for _ in range(20):
            assert q.allow(WHISPER, audio_s=10)
            q.record(WHISPER, audio_s=120)
        clock.now += 61  # stay under 20 requests per minute
    assert not q.allow(WHISPER, audio_s=10)
    assert q.used(WHISPER)["ash"] == 7200
    clock.now = T0 + 3600  # next UTC hour (T0 is on the hour)
    assert q.allow(WHISPER, audio_s=10)


def test_audio_seconds_per_day_and_requests_per_minute() -> None:
    clock = Clock()
    q = QuotaGovernor({WHISPER: Limits(rpm=2, audio_s_per_day=25)}, clock=clock)
    q.record(WHISPER, audio_s=10)
    assert q.allow(WHISPER, audio_s=10)
    q.record(WHISPER, audio_s=10)
    assert not q.allow(WHISPER, audio_s=10)  # rpm 2 reached
    clock.now += 60
    assert not q.allow(WHISPER, audio_s=10)  # 20 + 10 > 25 audio s today
    assert q.allow(WHISPER, audio_s=5)
    clock.now += 86_400
    assert q.allow(WHISPER, audio_s=10)  # daily reset at 00:00 UTC


def test_audio_seconds_warn_once_at_80_percent() -> None:
    warnings: list[tuple[str, str, float]] = []
    q = QuotaGovernor({WHISPER: Limits(audio_s_per_day=100)}, clock=Clock(),
                      on_warning=lambda k, c, f: warnings.append((k, c, round(f, 2))))  # fmt: skip
    for _ in range(9):
        q.record(WHISPER, audio_s=10)
    assert warnings == [(WHISPER, "asd", 0.8)]


def test_audio_hour_and_day_survive_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "quota.json"
    clock = Clock()
    QuotaGovernor({WHISPER: FREE_WHISPER}, state_path=path, clock=clock).record(
        WHISPER, audio_s=7200
    )
    again = QuotaGovernor({WHISPER: FREE_WHISPER}, state_path=path, clock=clock)
    used = again.used(WHISPER)
    assert (used["ash"], used["asd"], used["rpd"], used["asm"]) == (7200, 7200, 1, 0)
    assert not again.allow(WHISPER, audio_s=10)


def test_stt_and_router_counters_share_one_store(tmp_path: Path) -> None:
    path = tmp_path / "quota.json"
    q = QuotaGovernor({}, state_path=path, clock=Clock())
    q.record("groq:openai/gpt-oss-20b", tokens=50)
    q.record(WHISPER, audio_s=10)
    again = QuotaGovernor({}, state_path=path, clock=Clock())
    assert again.used("groq:openai/gpt-oss-20b")["tpd"] == 50 and again.used(WHISPER)["asd"] == 10


def test_old_quota_files_still_load(tmp_path: Path) -> None:
    """A logs/quota.json written by S07 (no audio counters) keeps its counts."""
    path = tmp_path / "quota.json"
    path.write_text('{"g": {"rpd": ["2027-01-15", 3]}}')
    q = QuotaGovernor({"g": Limits(rpd=3)}, state_path=path, clock=Clock())
    assert not q.allow("g") and q.used("g")["asd"] == 0


def test_build_governor_from_settings(tmp_path: Path) -> None:
    s = load_settings(tmp_path / "none.yaml", tmp_path / ".env", environ={})
    q = build_governor(s, state_path=None)
    assert q.limits["groq:openai/gpt-oss-20b"] == Limits(rpm=30, rpd=1000, tpm=8000, tpd=200_000)
    assert q.limits["openrouter:google/gemma-4-26b-a4b-it:free"].rpd == 50
    assert q.limits["cloudflare:@cf/black-forest-labs/flux-1-schnell"].units_per_day == 10_000
    assert q.limits["tavily:search"].units_per_month == 1000
    assert q.limits[WHISPER] == FREE_WHISPER


def test_cooldowns() -> None:
    clock = Clock(0.0)
    c = Cooldowns(10.0, clock)
    assert not c.active("groq")
    c.trip("groq")
    assert c.active("groq") and c.remaining("groq") == pytest.approx(10.0)
    clock.now += 10.0
    assert not c.active("groq") and c.remaining("groq") == 0.0
    c.trip("gemini", seconds=3)
    c.clear("gemini")
    assert not c.active("gemini")
