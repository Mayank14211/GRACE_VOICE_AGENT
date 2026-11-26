"""LLM router: provider chain, cooldowns, quota, validation + retry, history. No network."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from grace.actions import create_fake_registry
from grace.config import load_settings
from grace.core.cooldown import Cooldowns
from grace.core.quota import Limits, QuotaGovernor
from grace.intent.dialogue import Dialogue
from grace.intent.fakes import FakeChatProvider, calls, says, tool
from grace.intent.providers import (
    ChatReply,
    ProviderAuthError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
)
from grace.intent.router import (
    NO_PROVIDER,
    REPHRASE,
    SYSTEM_PROMPT,
    UNAVAILABLE,
    Router,
    build_router,
    spoken,
)
from grace.intent.tools import build_toolset
from grace.intent.types import Intent


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Rows:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def write(self, kind: str, **fields) -> None:
        self.rows.append({"kind": kind, **fields})

    def close(self) -> None:
        pass


@pytest.fixture
def toolset():
    return build_toolset(create_fake_registry()[0], search=True, images=True)


@pytest.fixture
def clock():
    return Clock()


def make(toolset, clock, *providers, **kw) -> Router:
    return Router(providers, toolset, clock=clock, now=lambda: datetime(2026, 11, 26, 14, 5), **kw)


VOLUME_40 = tool("system_control", {"action": "volume_set", "level": 40})


# --------------------------------------------------------------------------- success
def test_success_returns_validated_intents(toolset, clock) -> None:
    groq = FakeChatProvider("groq", calls(VOLUME_40))
    result = make(toolset, clock, groq).route("make it quieter, like forty")
    assert result.ok and result.provider == "groq" and not result.retried
    assert result.intents == [
        Intent("volume.set", {"level": 40}, "make it quieter, like forty", "router")
    ]


def test_multiple_tool_calls_keep_order_and_are_capped(toolset, clock) -> None:
    many = calls(
        tool("system_control", {"action": "volume_mute", "mute": True}),
        tool("apps_windows", {"action": "apps_open", "name": "youtube"}),
        tool("answer_user", {"text": "Done."}),
        tool("mouse", {"action": "input_click"}),
    )
    result = make(toolset, clock, FakeChatProvider("groq", many)).route("mute and open youtube")
    assert [i.name for i in result.intents] == ["volume.mute", "apps.open", "app.answer"]


def test_plain_text_is_a_spoken_answer(toolset, clock) -> None:
    reply = ChatReply(content="**Paris** is the capital. It is in France. It has the Louvre.")
    result = make(toolset, clock, FakeChatProvider("openrouter", reply)).route("capital of france")
    answer = {"text": "Paris is the capital. It is in France."}
    assert result.intents == [Intent("app.answer", answer, "capital of france", "router")]


def test_answers_are_at_most_two_sentences(toolset, clock) -> None:
    long = says("One. Two! Three? Four.")
    result = make(toolset, clock, FakeChatProvider("groq", long)).route("count")
    assert result.intents[0].args == {"text": "One. Two!"}


@pytest.mark.parametrize(
    ("raw", "out"),
    [
        ("Hello there.", "Hello there."),
        ("- one\n- two", "one two"),
        ("# Title\nSome `code` here. More. Even more.", "Title Some code here. More."),
        ("", ""),
    ],
)
def test_spoken(raw: str, out: str) -> None:
    assert spoken(raw) == out


# --------------------------------------------------------------------------- failures
def test_429_cools_down_and_falls_to_the_next(toolset, clock) -> None:
    groq = FakeChatProvider("groq", QuotaExceededError("busy", billing_required=False))
    gemini = FakeChatProvider("gemini", says("Hi."), says("Hi again."))
    router = make(toolset, clock, groq, gemini)
    result = router.route("hello")
    assert result.provider == "gemini" and result.ok
    assert result.attempts == [{"provider": "groq", "error": "QuotaExceededError"}]
    assert router.cooldowns.active("groq")
    # Within the cooldown groq is not even asked.
    second = router.route("hello again")
    assert second.provider == "gemini" and len(groq.requests) == 1
    assert second.attempts == [{"provider": "groq", "error": "cooldown"}]


def test_cooldown_expires_after_60_seconds(toolset, clock) -> None:
    groq = FakeChatProvider("groq", ProviderTimeoutError("slow"), says("Back."))
    backup = FakeChatProvider("openrouter", says("Local."))
    router = make(toolset, clock, groq, backup, cooldown_s=60.0)
    assert router.route("a").provider == "openrouter"
    clock.now += 59.9
    assert router.cooldowns.active("groq")
    clock.now += 0.2
    assert not router.cooldowns.active("groq")
    assert router.route("b").provider == "groq"


@pytest.mark.parametrize(
    "error",
    [
        ProviderTimeoutError("slow"),
        ProviderUnavailableError("down"),
        ProviderAuthError("bad key"),
        QuotaExceededError("quota", billing_required=True),
    ],
)
def test_any_provider_error_moves_on(toolset, clock, error) -> None:
    router = make(
        toolset, clock, FakeChatProvider("groq", error), FakeChatProvider("openrouter", says("ok"))
    )
    result = router.route("hi")
    assert result.provider == "openrouter" and router.cooldowns.active("groq")


def test_all_providers_failing_is_spoken_not_raised(toolset, clock) -> None:
    router = make(
        toolset,
        clock,
        FakeChatProvider("groq", QuotaExceededError("q", billing_required=False)),
        FakeChatProvider("gemini", ProviderTimeoutError("slow")),
        FakeChatProvider("openrouter", ProviderUnavailableError("not running")),
    )
    result = router.route("hi")
    assert not result.ok and result.reply == UNAVAILABLE and result.intents == []
    assert [a["provider"] for a in result.attempts] == ["groq", "gemini", "openrouter"]
    again = router.route("hi")  # all cooling down: nothing is called
    assert again.reply == UNAVAILABLE and {a["error"] for a in again.attempts} == {"cooldown"}


def test_no_providers(toolset, clock) -> None:
    result = make(toolset, clock).route("hi")
    assert not result.ok and result.reply == NO_PROVIDER


# --------------------------------------------------------------------------- validation
def test_malformed_json_gets_one_retry_with_the_error(toolset, clock) -> None:
    groq = FakeChatProvider(
        "groq",
        calls(tool("system_control", '{"action": "volume_set", "level": ')),
        calls(VOLUME_40),
    )
    result = make(toolset, clock, groq).route("volume forty please")
    assert result.ok and result.retried and result.intents[0].args == {"level": 40}
    correction = groq.requests[1]["messages"][-1]["content"]
    assert "not valid JSON" in correction
    # The retry keeps the same prefix (system prompt, history, user message).
    assert groq.requests[1]["messages"][:2] == groq.requests[0]["messages"]


def test_unknown_tool_retried_then_next_provider(toolset, clock) -> None:
    bad = calls(tool("launch_rockets", {}))
    groq = FakeChatProvider("groq", bad, bad)
    backup = FakeChatProvider("openrouter", calls(VOLUME_40))
    router = make(toolset, clock, groq, backup)
    result = router.route("volume forty")
    assert result.ok and result.provider == "openrouter" and len(groq.requests) == 2
    assert not router.cooldowns.active("groq")  # bad output is not a provider outage
    # The next provider starts clean: same history, no correction messages from groq.
    assert backup.requests[0]["messages"] == groq.requests[0]["messages"]


def test_invalid_everywhere_asks_to_rephrase(toolset, clock) -> None:
    bad = calls(tool("launch_rockets", {}))
    groq = FakeChatProvider("groq", bad, bad)
    backup = FakeChatProvider("openrouter", calls(tool("system_control", "{oops")), bad)
    result = make(toolset, clock, groq, backup).route("do the thing")
    assert not result.ok and result.reply == REPHRASE and result.retried
    assert [a["provider"] for a in result.attempts] == ["groq"] * 2 + ["openrouter"] * 2


def test_invalid_arguments_are_retried(toolset, clock) -> None:
    groq = FakeChatProvider(
        "groq", calls(tool("system_control", {"action": "volume_set"})), calls(VOLUME_40)
    )
    result = make(toolset, clock, groq).route("volume")
    assert result.ok and "invalid level" in groq.requests[1]["messages"][-1]["content"]


def test_empty_reply_is_invalid(toolset, clock) -> None:
    groq = FakeChatProvider("groq", ChatReply(), ChatReply())
    assert make(toolset, clock, groq).route("hm").reply == REPHRASE


def test_malformed_json_twice_falls_to_the_next_provider(toolset, clock) -> None:
    broken = calls(tool("system_control", '{"action": "volume_set", "level": '))
    groq = FakeChatProvider("groq", broken, broken)
    gemini = FakeChatProvider("gemini", calls(VOLUME_40))
    result = make(toolset, clock, groq, gemini).route("volume forty")
    assert result.provider == "gemini" and result.intents[0].args == {"level": 40}


def test_a_failure_during_the_retry_moves_to_the_next_provider(toolset, clock) -> None:
    groq = FakeChatProvider("groq", calls(tool("nope", {})), ProviderTimeoutError("slow"))
    backup = FakeChatProvider("openrouter", says("Hello."))
    result = make(toolset, clock, groq, backup).route("hi")
    assert result.provider == "openrouter" and result.ok


def test_risky_actions_are_never_tools_and_invented_ones_fail(clock) -> None:
    ts = build_toolset(create_fake_registry()[0], search=True, images=True)
    assert not {"system.shutdown", "system.restart", "system.sleep"} & ts.intents()
    shutdown = calls(tool("system_control", {"action": "system_shutdown"}))
    groq = FakeChatProvider("groq", shutdown, says("Say 'shut down the computer' to do that."))
    result = make(ts, clock, groq).route("turn everything off for tonight")
    assert [i.name for i in result.intents] == ["app.answer"]  # never system.shutdown


# --------------------------------------------------------------------------- prompt
def test_system_prompt_and_tools_are_byte_identical_across_calls(toolset, clock) -> None:
    groq = FakeChatProvider("groq", says("a"), says("b"), says("c"))
    router = make(toolset, clock, groq)
    dialogue = Dialogue(clock=clock)
    for text in ("first", "second", "third"):
        router.route(text, dialogue)
        dialogue.add(text, "ok")
    firsts = [json.dumps(r["messages"][0], sort_keys=True) for r in groq.requests]
    tools = [json.dumps(r["tools"], sort_keys=True) for r in groq.requests]
    assert len(set(firsts)) == 1 and len(set(tools)) == 1
    assert groq.requests[0]["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert all(r["tools"] is groq.requests[0]["tools"] for r in groq.requests)


def test_history_and_last_action_are_sent(toolset, clock) -> None:
    groq = FakeChatProvider("groq", says("Done."))
    router = make(toolset, clock, groq)
    dialogue = Dialogue(clock=clock)
    dialogue.add("volume 40", "Volume set to 40 percent.")
    dialogue.record(Intent("volume.set", {"level": 40}))
    router.route("and the other one", dialogue)
    messages = groq.requests[0]["messages"]
    assert messages[1:3] == [
        {"role": "user", "content": "volume 40"},
        {"role": "assistant", "content": "Volume set to 40 percent."},
    ]
    assert messages[-1]["content"] == (
        "Time: Thursday 26 November 2026, 14:05\n"
        "Last action: system_control(volume_set, level=40)\n"
        "User: and the other one"
    )


def test_metrics_and_provider_callback(toolset, clock) -> None:
    rows, shown = Rows(), []
    groq = FakeChatProvider("groq", QuotaExceededError("q", billing_required=False))
    backup = FakeChatProvider("openrouter", calls(VOLUME_40))
    router = make(toolset, clock, groq, backup, metrics=rows, on_provider=shown.append)
    router.route("my secret plan: volume forty")
    [row] = rows.rows
    assert row["kind"] == "router" and row["provider"] == "openrouter" and row["ok"] is True
    assert row["intents"] == ["volume.set"] and "latency_ms" in row
    assert "secret plan" not in json.dumps(row)  # utterances are never logged
    assert shown == ["openrouter"]
    make(toolset, clock, metrics=rows, on_provider=shown.append).route("x")
    assert shown[-1] == "offline"


# --------------------------------------------------------------------------- fallback + history
def test_fallback_keeps_the_history(toolset, clock) -> None:
    groq = FakeChatProvider(
        "groq", says("Canberra."), QuotaExceededError("q", billing_required=False)
    )
    gemini = FakeChatProvider("gemini", says("About 470,000 people."))
    router = make(toolset, clock, groq, gemini)
    dialogue = Dialogue(clock=clock)
    first = router.route("what is the capital of australia", dialogue)
    dialogue.add("what is the capital of australia", first.intents[0].args["text"])
    second = router.route("and how many people live there", dialogue)
    assert second.provider == "gemini"
    sent = gemini.requests[0]["messages"]
    assert sent[1:3] == [
        {"role": "user", "content": "what is the capital of australia"},
        {"role": "assistant", "content": "Canberra."},
    ]
    assert sent == groq.requests[1]["messages"]  # exactly what groq got before it failed


def test_five_forced_429s_all_fall_back_without_losing_history(toolset, clock) -> None:
    groq = FakeChatProvider("groq", *[QuotaExceededError("q", billing_required=False)] * 5)
    gemini = FakeChatProvider("gemini", *[says(f"Answer {i}.") for i in range(5)])
    router = make(toolset, clock, groq, gemini, cooldown_s=0.0)  # no cooldown: groq asked each time
    dialogue = Dialogue(clock=clock)
    for i in range(5):
        result = router.route(f"question {i}", dialogue)
        assert result.provider == "gemini" and result.ok
        assert result.attempts == [{"provider": "groq", "error": "QuotaExceededError"}]
        dialogue.add(f"question {i}", result.intents[0].args["text"])
    last = gemini.requests[-1]["messages"]
    assert [m["content"] for m in last[1:-1]] == [
        x for i in range(4) for x in (f"question {i}", f"Answer {i}.")
    ]
    assert len(groq.requests) == 5


# --------------------------------------------------------------------------- quota
def test_quota_skips_an_exhausted_provider(toolset, clock, tmp_path) -> None:
    quota = QuotaGovernor(
        {"groq:": Limits(rpd=1)}, state_path=tmp_path / "q.json", clock=lambda: 1_800_000_000
    )
    groq = FakeChatProvider("groq", says("One."))
    gemini = FakeChatProvider("gemini", says("Two."))
    router = make(toolset, clock, groq, gemini, quota=quota)
    assert router.route("a").provider == "groq"
    second = router.route("b")
    assert second.provider == "gemini"
    assert second.attempts == [{"provider": "groq", "error": "quota"}]


def test_tokens_are_counted_per_model(toolset, clock, tmp_path) -> None:
    quota = QuotaGovernor({}, state_path=None, clock=lambda: 1_800_000_000)
    reply = ChatReply(tool_calls=calls(VOLUME_40).tool_calls, usage={"total_tokens": 1234})
    groq = FakeChatProvider("groq", reply)
    make(toolset, clock, groq, quota=quota).route("volume 40")
    used = quota.used("groq:")
    assert used["rpm"] == 1 and used["rpd"] == 1 and used["tpd"] == 1234


# --------------------------------------------------------------------------- code analysis
def test_complete_never_sends_screen_text_to_gemini(toolset, clock) -> None:
    gemini = FakeChatProvider("gemini", ChatReply(content="unused"))
    groq = FakeChatProvider("groq", ProviderTimeoutError("slow"))
    openrouter = FakeChatProvider("openrouter", ChatReply(content="Looks fine."))
    router = make(toolset, clock, gemini, groq, openrouter)
    text = router.complete("```secret code```", instruction="review", exclude=frozenset({"gemini"}))
    assert text == "Looks fine." and gemini.requests == []
    req = openrouter.requests[0]
    assert req["tools"] == [] and req["messages"][0] == {"role": "system", "content": "review"}
    only_gemini = make(toolset, clock, FakeChatProvider("gemini"))
    assert "Gemini never sees screen text" in only_gemini.complete(
        "x", exclude=frozenset({"gemini"})
    )


def test_cooldowns_unit() -> None:
    clock = Clock()
    c = Cooldowns(10.0, clock)
    assert not c.active("groq")
    c.trip("groq")
    assert c.active("groq") and c.remaining("groq") == pytest.approx(10.0)
    clock.now += 10.0
    assert not c.active("groq") and c.remaining("groq") == 0.0


def test_build_router_from_settings(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=" + "gsk" + "_" + "A1b2C3d4" * 7 + "\n")
    cfg = tmp_path / "s.yaml"
    cfg.write_text("demo_mode: true\n")  # demo mode no longer pins the router
    router = build_router(load_settings(cfg, env, environ={}), create_fake_registry()[0])
    assert router.provider_names == ["groq"]
    assert "system.shutdown" not in router.toolset.intents()
    assert "web_search" not in router.toolset.names  # no search executor configured
    with_tools = build_router(
        load_settings(cfg, env, environ={}), create_fake_registry()[0], search=True, images=True
    )
    assert {"web_search", "generate_image"} <= set(with_tools.toolset.names)
