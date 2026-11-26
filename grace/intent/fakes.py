"""Fake chat provider for tests and offline dry runs (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from grace.intent.providers import ChatReply, ToolCall

Step = ChatReply | Exception | Callable[[Sequence[Mapping[str, Any]]], ChatReply]


def tool(name: str, arguments: Mapping[str, Any] | str | None = None, id_: str = "c1") -> ToolCall:
    raw = arguments if isinstance(arguments, str) else json.dumps(dict(arguments or {}))
    return ToolCall(id_, name, raw)


def calls(*tool_calls: ToolCall, content: str = "") -> ChatReply:
    return ChatReply(content=content, tool_calls=tuple(tool_calls), latency_ms=5.0)


def says(text: str) -> ChatReply:
    return calls(tool("answer_user", {"text": text}))


class FakeChatProvider:
    """Answers each chat() from a script: a ChatReply, an exception to raise, or a callable
    taking the messages. Records every request. An exhausted script raises AssertionError."""

    def __init__(self, name: str, *script: Step, timeout_s: float = 3.0) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.script: list[Step] = list(script)
        self.requests: list[dict[str, Any]] = []

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        timeout_s: float | None = None,
    ) -> ChatReply:
        self.requests.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "max_tokens": max_tokens,
                "timeout_s": timeout_s,
            }
        )
        if not self.script:
            raise AssertionError(f"{self.name}: no scripted reply left")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return step(messages)
        return step
