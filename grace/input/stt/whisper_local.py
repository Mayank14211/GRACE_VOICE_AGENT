"""faster-whisper on the CPU: Hindi / Hinglish and accurate dictation offline (extra [stt-whisper]).

    uv sync --extra stt-whisper && uv run python scripts/fetch_models.py --whisper

int8, beam_size=1, built-in VAD filter, ``cpu_threads = inference.threads``. Batch only: the
utterance is transcribed after it ends. Imported and loaded lazily, never downloaded at runtime
(``local_files_only``): a missing package or model is an SttError and the caller falls back.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from grace.input.stt.base import BatchEngine, SttError

log = logging.getLogger(__name__)


class WhisperLocalStt(BatchEngine):
    name = "whisper_local"
    streaming = False

    def __init__(
        self,
        size: str = "base",
        *,
        threads: int = 4,
        language: str | None = "en",
        download_root: Path,
    ) -> None:
        self.size = size
        self.threads = threads
        self.language = language  # None = detect
        self.download_root = Path(download_root)
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise SttError("faster-whisper not installed: uv sync --extra stt-whisper") from exc
        started = time.perf_counter()
        try:
            self._model = WhisperModel(
                self.size,
                device="cpu",
                compute_type="int8",
                cpu_threads=self.threads,
                download_root=str(self.download_root),
                local_files_only=True,
            )
        except Exception as exc:  # noqa: BLE001 - not downloaded, corrupt files
            raise SttError(
                f"faster-whisper '{self.size}' not available ({type(exc).__name__}): "
                f"run scripts/fetch_models.py --whisper {self.size}"
            ) from exc
        log.info(
            "faster-whisper %s loaded in %.0f ms (%d threads)",
            self.size,
            (time.perf_counter() - started) * 1000,
            self.threads,
        )

    def _transcribe(self, audio: np.ndarray) -> str:
        try:
            segments, _info = self._model.transcribe(
                audio,
                language=self.language,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            return " ".join(s.text.strip() for s in segments).strip()
        except Exception as exc:  # noqa: BLE001
            raise SttError(f"faster-whisper failed: {exc}") from exc
