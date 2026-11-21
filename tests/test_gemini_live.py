"""Live Gemini checks (free tier). OS actions go to fakes, so nothing on the PC changes.

    uv run pytest -m network tests/test_gemini_live.py -v

The fixture yields a GeminiClient, never the key string: pytest prints fixture values in
failure output, and GeminiClient's repr hides the key.
"""

from __future__ import annotations

import re

import pytest

from grace.actions import create_fake_registry
from grace.config import load_settings
from grace.intent.assistant import GeminiAssistant
from grace.intent.providers import GeminiClient

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def client() -> GeminiClient:
    settings = load_settings()
    if not settings.api_key("gemini"):
        pytest.skip("GEMINI_API_KEY not set in .env")
    return GeminiClient(settings.api_key("gemini"), timeout_s=30)


def digits(text: str) -> str:
    words = {"eighty-four": "84", "eighty four": "84", "twenty-eight": "28", "twenty eight": "28"}
    t = text.lower()
    for w, d in words.items():
        t = t.replace(w, d)
    return " ".join(re.findall(r"\d+", t))


def test_answer_memory_and_tool_call(client: GeminiClient, tmp_path) -> None:
    registry, b = create_fake_registry(screenshot_dir=tmp_path)
    a = GeminiAssistant(client, registry, image_dir=tmp_path)

    first = a.ask("What is twelve times seven? Answer with the number only.")
    assert first.ok and "84" in digits(first.text), first.text
    follow = a.ask("Now divide that by three. Number only.")
    assert follow.ok and "28" in digits(follow.text), follow.text

    acted = a.ask("Set the volume to thirty, then open notepad.")
    assert acted.ok, acted.text
    assert b.volume.level == 30 and b.launcher.launched == ["notepad"]

    refused = a.ask("Shut down my computer now.")
    assert b.power.calls == [] and refused.ok
