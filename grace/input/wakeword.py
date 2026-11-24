"""Wake word: openWakeWord with the team's custom .onnx model (docs/wakeword.md).

    det = OpenWakeWord("models/wakeword/grace.onnx", threshold=0.5, patience=2)
    score = det.process(frame)    # 512-sample float32 frames; a score when the word was heard

The wake word is optional. ``load_wake_word`` returns None when it is turned off or the model
file is missing, and the voice pipeline falls back (grace/input/voice.py): no model -> the spoken
``runtime.wake_word`` checked in the offline transcript; turned off -> no wake word at all. The mic
hotkey and push-to-talk work in every case.

Patience is applied here, on openWakeWord's raw scores: openwakeword 0.6.0's own ``patience``
argument compares against its prediction buffer *after* zeroing, so with patience >= 1 it never
fires (checked on the hey_jarvis model: raw scores 0.99 for 8 frames, patience=2 -> nothing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from grace.input.audio import to_int16

log = logging.getLogger(__name__)

CHUNK = 1280  # openWakeWord works on 80 ms of 16 kHz audio


class WakeWordDetector(Protocol):
    name: str

    def process(self, frame: np.ndarray) -> float | None:
        """Feed one frame; returns the score when the wake word was detected on it."""
        ...

    def reset(self) -> None: ...


class PatienceGate:
    """Pure: fires when ``patience`` scores in a row reach ``threshold``, then stays quiet for
    ``refractory`` chunks so one spoken word gives one detection."""

    def __init__(self, threshold: float, patience: int, refractory: int = 12) -> None:
        self.threshold = threshold
        self.patience = patience
        self.refractory = refractory
        self.reset()

    def reset(self) -> None:
        self._run = 0
        self._quiet = 0

    def push(self, score: float) -> bool:
        if self._quiet:
            self._quiet -= 1
            self._run = 0
            return False
        self._run = self._run + 1 if score >= self.threshold else 0
        if self._run >= self.patience:
            self._run = 0
            self._quiet = self.refractory
            return True
        return False


class OpenWakeWord:
    def __init__(
        self,
        model_path: str | Path,
        *,
        melspec_path: str | Path,
        embedding_path: str | Path,
        threshold: float = 0.5,
        patience: int = 2,
    ) -> None:
        from openwakeword.model import Model

        self._model: Any = Model(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
            melspec_model_path=str(melspec_path),
            embedding_model_path=str(embedding_path),
        )
        self.name = next(iter(self._model.models))
        self.gate = PatienceGate(threshold, patience)
        self._buf = np.zeros(0, dtype=np.int16)
        self.last_score = 0.0

    def process(self, frame: np.ndarray) -> float | None:
        self._buf = np.concatenate([self._buf, to_int16(frame)])
        hit: float | None = None
        while len(self._buf) >= CHUNK:
            chunk, self._buf = self._buf[:CHUNK], self._buf[CHUNK:]
            score = float(self._model.predict(chunk)[self.name])
            self.last_score = score
            if self.gate.push(score) and hit is None:
                hit = score
        return hit

    def reset(self) -> None:
        """Clear audio features and scores (after a detection or when the mic reopens)."""
        self._model.reset()
        self.gate.reset()
        self._buf = np.zeros(0, dtype=np.int16)

    def raw_scores(self, samples: np.ndarray) -> list[float]:
        """Raw score of every 80 ms chunk of ``samples`` (threshold tuning; resets first)."""
        self.reset()
        pcm = to_int16(np.asarray(samples, dtype=np.float32))
        out = [
            float(self._model.predict(pcm[i : i + CHUNK])[self.name])
            for i in range(0, len(pcm) - CHUNK + 1, CHUNK)
        ]
        self.reset()
        return out


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    false_reject_rate: float  # share of wake-word clips with no detection
    false_accepts_per_hour: float  # detections in audio without the wake word


def evaluate_threshold(
    positives: list[list[float]],
    negatives: list[list[float]],
    threshold: float,
    patience: int,
) -> ThresholdResult:
    """Pure: run the patience gate over per-clip raw scores (from ``raw_scores``)."""

    def detections(scores: list[float]) -> int:
        gate = PatienceGate(threshold, patience)
        return sum(gate.push(s) for s in scores)

    missed = sum(detections(s) == 0 for s in positives)
    neg_hours = sum(len(s) for s in negatives) * CHUNK / 16000 / 3600
    false = sum(detections(s) for s in negatives)
    return ThresholdResult(
        threshold,
        missed / len(positives) if positives else 0.0,
        false / neg_hours if neg_hours else 0.0,
    )


def load_wake_word(
    enabled: bool,
    model_path: Path,
    *,
    models_dir: Path,
    threshold: float,
    patience: int,
) -> tuple[OpenWakeWord | None, str]:
    """(detector, reason). None + reason when the wake word is off or can't be loaded."""
    from grace.models import OWW_EMBEDDING, OWW_MELSPEC

    if not enabled:
        return None, "wake word turned off (wakeword.enabled: false)"
    if not model_path.is_file():
        return None, f"no wake-word model at {model_path} (see docs/wakeword.md)"
    melspec, embedding = models_dir / OWW_MELSPEC, models_dir / OWW_EMBEDDING
    if not (melspec.is_file() and embedding.is_file()):
        return None, "openWakeWord feature models missing: run scripts/fetch_models.py"
    try:
        det = OpenWakeWord(
            model_path,
            melspec_path=melspec,
            embedding_path=embedding,
            threshold=threshold,
            patience=patience,
        )
    except Exception as exc:  # noqa: BLE001 - a broken model must not stop voice control
        log.warning("wake-word model failed to load: %s", exc)
        return None, f"wake-word model failed to load: {type(exc).__name__}: {exc}"
    return det, f"openWakeWord '{det.name}' (threshold {threshold}, patience {patience})"
