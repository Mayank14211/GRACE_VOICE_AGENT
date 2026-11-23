"""Moonshine Voice (moonshine-voice), the default English engine: streaming, offline, CPU.

Small Streaming by default (stt.moonshine_model). Each utterance gets its own stream on one shared
Transcriber: audio is added as it arrives, Moonshine re-transcribes every ``update_interval``
seconds (partials), and ``stop()`` finishes the last bit, so the final text is ready about 0.1 s
after the end of speech on the base machine.

moonshine-voice runs its own bundled onnxruntime 1.23 (moonshine.dll + onnxruntime.dll in the
package) with no thread-count option; ``MOONSHINE_ORT_SINGLE_THREAD=1`` forces one thread.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from grace.input.stt.base import SAMPLE_RATE, AudioClock, Partial, SttError, SttEvent

log = logging.getLogger(__name__)


class MoonshineStt:
    name = "moonshine"
    streaming = True

    def __init__(self, model_dir: Path, arch: int, *, update_interval: float = 0.3) -> None:
        self.model_dir = Path(model_dir)
        self.arch = arch
        self.update_interval = update_interval
        self._transcriber: Any = None

    def load(self) -> None:
        if self._transcriber is not None:
            return
        if not self.model_dir.is_dir():
            raise SttError("Moonshine model missing: run scripts/fetch_models.py")
        try:
            from moonshine_voice import Transcriber
            from moonshine_voice.moonshine_api import ModelArch
        except Exception as exc:  # noqa: BLE001 - DLL load errors are OSError, not ImportError
            raise SttError(f"moonshine-voice unavailable: {type(exc).__name__}: {exc}") from exc
        started = time.perf_counter()
        try:
            self._transcriber = Transcriber(
                model_path=str(self.model_dir),
                model_arch=ModelArch(self.arch),
                update_interval=self.update_interval,
            )
        except Exception as exc:  # noqa: BLE001
            raise SttError(f"Moonshine model failed to load: {exc}") from exc
        log.info("moonshine loaded in %.0f ms", (time.perf_counter() - started) * 1000)

    def transcribe_stream(self, chunks: Iterable[np.ndarray]) -> Iterator[SttEvent]:
        self.load()
        lines: dict[int, str] = {}

        def on_event(event: Any) -> None:
            line = getattr(event, "line", None)
            if line is not None:
                lines[line.line_id] = line.text

        def joined() -> str:
            return " ".join(t.strip() for t in lines.values() if t and t.strip())

        clock = AudioClock()
        stream = self._transcriber.create_stream(update_interval=self.update_interval)
        stream.add_listener(on_event)
        partials = 0
        try:
            stream.start()
            last = ""
            for chunk in chunks:
                clock.add(chunk)
                stream.add_audio(np.asarray(chunk, dtype=np.float32).tolist(), SAMPLE_RATE)
                text = joined()
                if text and text != last:
                    last = text
                    partials += 1
                    yield Partial(text, time.perf_counter())
            clock.end()
            transcript = stream.stop()
            if transcript is not None and transcript.lines:
                text = " ".join(ln.text.strip() for ln in transcript.lines if ln.text.strip())
            else:
                text = joined()
        except SttError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface engine errors as SttError
            raise SttError(f"Moonshine failed: {exc}") from exc
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                log.debug("closing a moonshine stream failed", exc_info=True)
        yield clock.final(text, self.name, partials)
