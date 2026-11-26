"""LLM router: provider chain, cooldowns, validation + one retry, dialogue context. No network."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from grace.actions import create_fake_registry
from grace.config import load_settings
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
    Cooldowns,
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
    return build_toolset(create_fake_registry()[0], extras=True)


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
    result = make(toolset, clock, FakeChatProvider("local", reply)).route("capital of france")
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
    local = FakeChatProvider("local", says("Local."))
    router = make(toolset, clock, groq, local, cooldown_s=60.0)
    assert router.route("a").provider == "local"
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
        toolset, clock, FakeChatProvider("groq", error), FakeChatProvider("local", says("ok"))
    )
    result = router.route("hi")
    assert result.provider == "local" and router.cooldowns.active("groq")


def test_all_providers_failing_is_spoken_not_raised(toolset, clock) -> None:
    router = make(
        toolset,
        clock,
        FakeChatProvider("groq", QuotaExceededError("q", billing_required=False)),
        FakeChatProvider("gemini", ProviderTimeoutError("slow")),
        FakeChatProvider("local", ProviderUnavailableError("not running")),
    )
    result = router.route("hi")
    assert not result.ok and result.reply == UNAVAILABLE and result.intents == []
    assert [a["provider"] for a in result.attempts] == ["groq", "gemini", "local"]
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


def test_unknown_tool_retried_then_rephrase(toolset, clock) -> None:
    bad = calls(tool("launch_rockets", {}))
    groq = FakeChatProvider("groq", bad, bad)
    local = FakeChatProvider("local")  # never asked: bad output is not a provider failure
    router = make(toolset, clock, groq, local)
    result = router.route("do the thing")
    assert not result.ok and result.reply == REPHRASE and result.retried
    assert len(groq.requests) == 2 and local.requests == []
    assert not router.cooldowns.active("groq")


def test_invalid_arguments_are_retried(toolset, clock) -> None:
    groq = FakeChatProvider(
        "groq", calls(tool("system_control", {"action": "volume_set"})), calls(VOLUME_40)
    )
    result = make(toolset, clock, groq).route("volume")
    assert result.ok and "invalid level" in groq.requests[1]["messages"][-1]["content"]


def test_empty_reply_is_invalid(toolset, clock) -> None:
    groq = FakeChatProvider("groq", ChatReply(), ChatReply())
    assert make(toolset, clock, groq).route("hm").reply == REPHRASE


def test_a_failure_during_the_retry_moves_to_the_next_provider(toolset, clock) -> None:
    groq = FakeChatProvider("groq", calls(tool("nope", {})), ProviderTimeoutError("slow"))
    local = FakeChatProvider("local", says("Hello."))
    result = make(toolset, clock, groq, local).route("hi")
    assert result.provider == "local" and result.ok


def test_risky_intents_are_returned_for_the_gate(clock) -> None:
    risky = build_toolset(create_fake_registry()[0], allow_risky=True)
    groq = FakeChatProvider("groq", calls(tool("system_control", {"action": "system_shutdown"})))
    result = make(risky, clock, groq).route("turn everything off for tonight")
    assert [i.name for i in result.intents] == ["system.shutdown"]  # the agent asks to confirm
    safe = build_toolset(create_fake_registry()[0])
    groq = FakeChatProvider(
        "groq", calls(tool("system_control", {"action": "system_shutdown"})), says("I can't.")
    )
    assert make(safe, clock, groq).route("shut it all down").intents[0].name == "app.answer"


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
    local = FakeChatProvider("local", calls(VOLUME_40))
    router = make(toolset, clock, groq, local, metrics=rows, on_provider=shown.append)
    router.route("my secret plan: volume forty")
    [row] = rows.rows
    assert row["kind"] == "router" and row["provider"] == "local" and row["ok"] is True
    assert row["intents"] == ["volume.set"] and "latency_ms" in row
    assert "secret plan" not in json.dumps(row)  # utterances are never logged
    assert shown == ["local"]
    make(toolset, clock, metrics=rows, on_provider=shown.append).route("x")
    assert shown[-1] == "offline"


# --------------------------------------------------------------------------- warm-up
def test_warm_up_sends_the_same_prefix_with_a_long_timeout(toolset, clock) -> None:
    local = FakeChatProvider("local", says("."), timeout_s=15.0)
    router = make(toolset, clock, local)
    assert router.warm_up(timeout_s=120.0) is True
    req = local.requests[0]
    assert req["messages"][0]["content"] == SYSTEM_PROMPT and req["tools"] == toolset.tools
    assert req["max_tokens"] == 1 and req["timeout_s"] == 120.0
    assert (
        make(toolset, clock, FakeChatProvider("local", ProviderTimeoutError("x"))).warm_up()
        is False
    )
    assert make(toolset, clock).warm_up() is False


def test_cooldowns_unit() -> None:
    clock = Clock()
    c = Cooldowns(10.0, clock)
    assert not c.active("groq")
    c.trip("groq")
    assert c.active("groq") and c.remaining("groq") == pytest.approx(10.0)
    clock.now += 10.0
    assert not c.active("groq") and c.remaining("groq") == 0.0


def test_build_router_from_settings(tmp_path: Path) -> None:
    cfg = tmp_path / "s.yaml"
    cfg.write_text("demo_mode: true\nrouter: {allow_risky: true}\n")
    router = build_router(
        load_settings(cfg, tmp_path / ".env", environ={}), create_fake_registry()[0]
    )
    assert router.provider_names == ["local"]
    assert "system.shutdown" in router.toolset.intents()
    assert "web_search" not in router.toolset.names


# --------------------------------------------------------------------------- multi-step
NOTEPAD = tool("apps_windows", {"action": "apps_open", "name": "notepad"}, id_="c2")


def test_more_asks_for_the_next_step(toolset, clock) -> None:
    first = calls(tool("system_control", {"action": "volume_set", "level": 30, "more": True}))
    groq = FakeChatProvider("groq", first, calls(NOTEPAD))
    result = make(toolset, clock, groq).route("set the sound to thirty and then open notepad")
    assert [i.name for i in result.intents] == ["volume.set", "apps.open"]
    assert result.intents[0].args == {"level": 30}  # `more` is not an action argument
    follow = groq.requests[1]["messages"]
    assert follow[: len(groq.requests[0]["messages"])] == groq.requests[0]["messages"]
    assistant, queued = follow[-2], follow[-1]
    assert assistant["role"] == "assistant" and assistant["tool_calls"][0]["id"] == "c1"
    assert queued == {"role": "tool", "tool_call_id": "c1", "content": "Queued. Next step, if any."}


def test_provider_extras_are_replayed_with_the_call(toolset, clock) -> None:
    from grace.intent.providers import ToolCall

    signed = ToolCall(
        "c1",
        "system_control",
        '{"action": "volume_set", "level": 30, "more": true}',
        {"extra_content": {"google": {"thought_signature": "sig"}}},
    )
    groq = FakeChatProvider("gemini", calls(signed), calls(NOTEPAD))
    make(toolset, clock, groq).route("x")
    replayed = groq.requests[1]["messages"][-2]["tool_calls"][0]
    assert replayed["extra_content"] == {"google": {"thought_signature": "sig"}}
    assert replayed["id"] == "c1" and replayed["function"]["name"] == "system_control"


def test_rounds_are_bounded(toolset, clock) -> None:
    again = calls(tool("mouse", {"action": "input_scroll", "more": True}))
    groq = FakeChatProvider("groq", again, again, again, again)
    router = make(toolset, clock, groq, max_intents=10)
    result = router.route("scroll forever")
    assert len(groq.requests) == 3 and len(result.intents) == 3


def test_a_failure_in_a_later_round_restarts_on_the_next_provider(toolset, clock) -> None:
    first = calls(tool("system_control", {"action": "volume_set", "level": 30, "more": True}))
    groq = FakeChatProvider("groq", first, ProviderTimeoutError("slow"))
    local = FakeChatProvider("local", calls(VOLUME_40))
    result = make(toolset, clock, groq, local).route("x")
    assert result.provider == "local" and [i.args for i in result.intents] == [{"level": 40}]


def test_invalid_later_round_keeps_the_earlier_steps(toolset, clock) -> None:
    first = calls(tool("system_control", {"action": "volume_set", "level": 30, "more": True}))
    bad = calls(tool("nope", {}))
    groq = FakeChatProvider("groq", first, bad, bad)
    result = make(toolset, clock, groq).route("x")
    assert result.ok and [i.name for i in result.intents] == ["volume.set"]
