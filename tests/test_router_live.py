"""Live router checks against each configured provider. OS actions go to fakes, so nothing on
the PC changes.

    uv run pytest -m network tests/test_router_live.py -v

Each enabled provider (key in .env; local only when llama-server answers) is tested on its own:
a plain answer, memory across turns, a tool call, and a refused shutdown. Fixtures never yield
key strings (pytest prints fixture values on failure; provider reprs hide keys).
"""

from __future__ import annotations

import re

import httpx
import pytest

from grace.actions import create_fake_registry
from grace.config import load_settings
from grace.intent.providers import build_chat_providers
from grace.intent.router import Router
from grace.intent.tools import build_toolset
from grace.legacy_voice import GraceVoiceAgent

pytestmark = pytest.mark.network


def _providers():
    settings = load_settings()
    found = []
    for provider in build_chat_providers(settings):
        if provider.name == "local":
            try:
                httpx.get(settings.router.local.base_url + "/models", timeout=2.0)
            except httpx.HTTPError:
                continue
        found.append(provider)
    return found


PROVIDERS = _providers()


class Speech:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak_async(self, text: str) -> None:
        self.spoken.append(text)

    speak = speak_async


def digits(text: str) -> str:
    words = {"eighty-four": "84", "eighty four": "84", "twenty-eight": "28", "twenty eight": "28"}
    t = text.lower()
    for w, d in words.items():
        t = t.replace(w, d)
    return " ".join(re.findall(r"\d+", t))


@pytest.mark.skipif(not PROVIDERS, reason="no provider key in .env and no llama-server running")
@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p.name)
def test_answer_memory_and_tool_call(provider) -> None:
    provider.timeout_s = max(provider.timeout_s, 20.0)  # a cold free tier can be slow
    registry, b = create_fake_registry()
    router = Router([provider], build_toolset(registry))
    agent = GraceVoiceAgent(registry=registry, tts=Speech(), screen_reader=object(), router=router)

    agent._process_command("What is twelve times seven? Answer with the number only.")
    assert "84" in digits(agent.tts.spoken[-1]), agent.tts.spoken
    agent._process_command("Now divide that by three. Number only.")
    assert "28" in digits(agent.tts.spoken[-1]), agent.tts.spoken

    agent._process_command("Could you set the sound to thirty, and then open notepad for me")
    assert b.volume.level == 30 and b.launcher.launched == ["notepad"], agent.tts.spoken

    agent._process_command("Shut down my computer right now.")
    assert b.power.calls == [] and agent.gate.pending is None
