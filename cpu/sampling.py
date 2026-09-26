"""
Next-token sampling for GPT-2 logits (NumPy only).

Pipeline (stochastic mode, temperature > 0):

  logits
    -> optional repetition penalty (divide logits for tokens already in context)
    -> temperature scaling (logits / T)
    -> top-k masking (non-top-k set to -inf)
    -> top-p (nucleus) masking
    -> softmax + renormalize
    -> categorical sample

temperature == 0 bypasses all stochastic steps and uses argmax on the raw
last-step logits (identical to historical greedy_generate when penalty is 1.0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpt2 import softmax

NEG_INF = float("-inf")


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None

    def validate(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top-k must be >= 0 (0 = disabled), got {self.top_k}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top-p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition-penalty must be > 0, got {self.repetition_penalty}")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    @property
    def strategy_label(self) -> str:
        return "greedy" if self.is_greedy else "sampling"


def apply_repetition_penalty(logits: np.ndarray, context_ids: list[int], penalty: float) -> np.ndarray:
    """
    For each vocabulary index that appears in `context_ids`, divide its logit by
    `penalty` (only when penalty != 1.0). Values > 1.0 discourage reuse.
    """
    if penalty == 1.0 or not context_ids:
        return logits
    out = logits.copy()
    unique = set(context_ids)
    for tid in unique:
        out[tid] /= penalty
    return out


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return logits
    return logits / temperature


def apply_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0:
        return logits
    out = logits.copy()
    k = min(top_k, out.shape[-1])
    keep = np.argpartition(out, -k)[-k:]
    mask = np.ones(out.shape[0], dtype=bool)
    mask[keep] = False
    out[mask] = NEG_INF
    return out


def apply_top_p(logits: np.ndarray, top_p: float) -> np.ndarray:
    """Nucleus (top-p) filter on logits via sorted cumulative probability mass."""
    if top_p >= 1.0:
        return logits
    out = logits.copy()
    sorted_idx = np.argsort(out)[::-1]
    sorted_logits = out[sorted_idx]
    probs = softmax(sorted_logits)
    cum = np.cumsum(probs)
