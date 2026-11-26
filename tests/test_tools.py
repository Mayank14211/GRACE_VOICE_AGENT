"""Router tool schema: generated from the registry, stable, validated into Intents. No network."""

from __future__ import annotations

import json

import pytest

from grace.actions import create_fake_registry
from grace.actions.registry import ActionArgs, ActionResult
from grace.intent.tools import (
    EXCLUDED_ACTIONS,
    GROUPS,
    ToolCallError,
    build_toolset,
    clean_schema,
    sub_name,
)
from grace.intent.types import APP_INTENTS, Intent


@pytest.fixture
def registry():
    return create_fake_registry()[0]


def test_generated_schema_matches_the_registry(registry) -> None:
    ts = build_toolset(registry, extras=True)
    offered = ts.intents()
    expected = {s.name for s in registry if not s.risky and s.name not in EXCLUDED_ACTIONS}
    # Every safe registry action is offered exactly once, and nothing that isn't registered
    # (apart from app intents the voice agent handles).
    assert expected <= offered
    assert offered - expected <= APP_INTENTS
    seen = [s.intent for subs in ts.subs.values() for s in subs.values()]
    assert len(seen) == len(set(seen))
    for subs in ts.subs.values():
        for sub in subs.values():
            if sub.intent in registry:
                assert sub.args_model is registry.spec(sub.intent).args_model
                assert sub.name == sub_name(sub.intent)


def test_parameters_come_from_the_args_models(registry) -> None:
    ts = build_toolset(registry)
    system = next(t for t in ts.tools if t["function"]["name"] == "system_control")
    params = system["function"]["parameters"]
    assert params["required"] == ["action"]
    assert params["properties"]["action"]["enum"] == [
        "volume_set", "volume_step", "volume_mute", "volume_get",
        "brightness_set", "brightness_step", "brightness_get",
    ]  # fmt: skip
    level = clean_schema(registry.spec("volume.set").args_model.model_json_schema())
    assert params["properties"]["level"]["type"] == level["properties"]["level"]["type"]
    mouse = next(t for t in ts.tools if t["function"]["name"] == "mouse")["function"]
    assert mouse["parameters"]["properties"]["button"]["enum"] == ["left", "right", "middle"]


def test_eight_to_ten_grouped_tools(registry) -> None:
    ts = build_toolset(registry, extras=True)
    assert ts.names == [
        "system_control", "mouse", "keyboard_text", "apps_windows", "read_screen",
        "web_search", "generate_image", "answer_user",
    ]  # fmt: skip
    assert len(GROUPS) + 3 == 9  # + notifications once S09 registers notifications.*
    assert "web_search" not in build_toolset(registry).names  # no Gemini extras


def test_notifications_group_appears_when_registered(registry) -> None:
    class ReadArgs(ActionArgs):
        count: int = 3

    registry.register("notifications.read", ReadArgs, description="Read recent notifications.")(
        lambda args: ActionResult(True, "None.")
    )
    ts = build_toolset(registry, extras=True)
    assert "notifications" in ts.names and len(ts.names) == 9
    intent = ts.to_intent("notifications", {"action": "notifications_read", "count": 2}, "x")
    assert intent == Intent("notifications.read", {"count": 2}, "x", "router")


def test_pointer_primitives_and_show_image_are_never_offered(registry) -> None:
    offered = build_toolset(registry, extras=True, allow_risky=True).intents()
    assert not offered & EXCLUDED_ACTIONS


def test_risky_actions_only_with_allow_risky(registry) -> None:
    assert (
        not {"system.shutdown", "system.restart", "system.sleep"}
        & build_toolset(registry).intents()
    )
    ts = build_toolset(registry, allow_risky=True)
    assert ts.risky == {"system.shutdown", "system.restart", "system.sleep"}
    assert ts.to_intent("system_control", {"action": "system_shutdown"}, "x").name == (
        "system.shutdown"
    )


def test_schema_is_byte_identical_across_builds(registry) -> None:
    a = build_toolset(registry, extras=True)
    b = build_toolset(create_fake_registry()[0], extras=True)
    assert a.fingerprint() == b.fingerprint()
    assert json.dumps(a.tools) == json.dumps(b.tools)  # same key order too


def test_valid_calls_become_router_intents(registry) -> None:
    ts = build_toolset(registry, extras=True)
    assert ts.to_intent("system_control", '{"action": "volume_set", "level": 40}', "t") == Intent(
        "volume.set", {"level": 40}, "t", "router"
    )
    # Models often send every union parameter, null where unused: those are dropped.
    raw = '{"action": "volume_step", "delta": -10, "level": null, "mute": null}'
    assert ts.to_intent("system_control", raw, "t").args == {"delta": -10}
    assert ts.to_intent("answer_user", {"text": "Hi."}, "t").name == "app.answer"
    assert ts.to_intent("read_screen", {"action": "read_screen"}, "t").name == "app.read_screen"
    assert ts.to_intent("web_search", {"query": "news"}, "t").args == {"query": "news"}
    img = ts.to_intent("generate_image", {"prompt": "a cat"}, "t")
    assert img.args == {"prompt": "a cat", "aspect_ratio": "1:1"}


@pytest.mark.parametrize(
    ("tool", "arguments", "message"),
    [
        ("system_control", "{not json", "not valid JSON"),
        ("system_control", "[1, 2]", "JSON object"),
        ("launch_rockets", "{}", "unknown tool"),
        ("system_control", '{"action": "volume_blast"}', "'action' must be one of"),
        ("system_control", "{}", "'action' must be one of"),
        ("system_control", '{"action": "volume_set"}', "invalid level"),
        ("system_control", '{"action": "volume_set", "level": "loud"}', "invalid level"),
        ("apps_windows", '{"action": "apps_open", "name": ""}', "invalid name"),
        ("answer_user", '{"text": ""}', "invalid text"),
        ("generate_image", '{"prompt": "x", "aspect_ratio": "2:1"}', "invalid aspect_ratio"),
        ("system_control", '{"action": "system_shutdown"}', "'action' must be one of"),
    ],
)
def test_invalid_calls_raise_with_a_reason_for_the_model(
    registry, tool: str, arguments: str, message: str
) -> None:
    ts = build_toolset(registry, extras=True)
    with pytest.raises(ToolCallError, match=message):
        ts.to_intent(tool, arguments, "t")


def test_describe_for_the_dialogue_context(registry) -> None:
    ts = build_toolset(registry)
    assert (
        ts.describe(Intent("volume.set", {"level": 40})) == "system_control(volume_set, level=40)"
    )
    assert ts.describe(Intent("apps.close_tab")) == "apps_windows(apps_close_tab)"
    assert ts.describe(Intent("input.move_to", {"x": 1})) == "input.move_to"


def test_every_action_tool_takes_more(registry) -> None:
    ts = build_toolset(registry, extras=True)
    for t in ts.tools:
        props = t["function"]["parameters"]["properties"]
        assert ("more" in props) == (t["function"]["name"] != "answer_user")
    intent, more = ts.parse_call("mouse", {"action": "input_click", "more": True}, "t")
    assert more is True and intent.args == {"button": "left", "count": 1}
    assert ts.parse_call("mouse", {"action": "input_click", "more": "yes"}, "t")[1] is False
