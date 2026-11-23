"""Groq ``whisper-large-v3-turbo`` over HTTPS: optional accelerator for Hindi / accurate dictation.

Free tier (docs/PLAN.md section 6): 20 requests/min, 2,000/day, 28,800 audio-seconds/day. Needs
GROQ_API_KEY in .env. Every request has a timeout and a retry limit; HTTP 429 or any failure is an
SttError so the caller falls back to faster-whisper (and cools Groq down). The key is only ever
sent in the Authorization header; it never appears in messages or logs.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from grace.input.stt.base import SAMPLE_RATE, BatchEngine, SttError

log = logging.getLogger(__name__)

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def wav_bytes(audio: np.ndarray) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class GroqStt(BatchEngine):
    name = "groq"
    streaming = False

    def __init__(
        self,
        api_key: str | None,
        *,
        model: str = "whisper-large-v3-turbo",
        language: str | None = "en",
        timeout_s: float = 8.0,
        max_retries: int = 1,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._key = api_key
        self.model = model
        self.language = language  # None = detect
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = client
        self._sleep = sleep

    def load(self) -> None:
        if not self._key:
            raise SttError("GROQ_API_KEY not set in .env")
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout_s)

    def _transcribe(self, audio: np.ndarray) -> str:
        import httpx

        data = {"model": self.model, "response_format": "json", "temperature": "0"}
        if self.language:
            data["language"] = self.language
        body = wav_bytes(audio)
        problem = "no attempt"
        for attempt in range(self.max_retries + 1):
            if attempt:
                self._sleep(0.3 * 2 ** (attempt - 1))
            try:
                resp = self._client.post(
                    GROQ_TRANSCRIBE_URL,
                    headers={"Authorization": f"Bearer {self._key}"},
                    files={"file": ("speech.wav", body, "audio/wav")},
                    data=data,
                    timeout=self.timeout_s,
                )
            except httpx.HTTPError as exc:
                problem = f"{type(exc).__name__}"  # no URL/headers: nothing secret in the text
                continue
            if resp.status_code == 200:
                try:
                    return str(resp.json().get("text", "")).strip()
                except ValueError as exc:
                    raise SttError("Groq returned something that is not JSON") from exc
            if resp.status_code == 429:
                raise SttError("Groq rate limit reached (HTTP 429)")
            problem = f"HTTP {resp.status_code}"
            if resp.status_code < 500:
                break  # 4xx other than 429: retrying won't help
        raise SttError(f"Groq transcription failed: {problem}")
