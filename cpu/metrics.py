"""Benchmark metrics, reporting, and JSON export for cpu/run.py."""

from __future__ import annotations

import json
import platform
import resource
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gpt2 import GPT2Config, GPT2Weights


def count_parameters(weights: GPT2Weights) -> int:
    arrays: list[np.ndarray] = [
        weights.wte,
        weights.wpe,
        weights.ln_f_w,
        weights.ln_f_b,
    ]
    for lst in (
        weights.ln_1_w,
        weights.ln_1_b,
        weights.c_attn_w,
        weights.c_attn_b,
        weights.c_proj_w,
        weights.c_proj_b,
        weights.ln_2_w,
        weights.ln_2_b,
        weights.c_fc_w,
        weights.c_fc_b,
        weights.c_proj_mlp_w,
        weights.c_proj_mlp_b,
    ):
        arrays.extend(lst)
    return int(sum(a.size for a in arrays))


def weight_bytes(weights: GPT2Weights) -> int:
    arrays: list[np.ndarray] = [weights.wte, weights.wpe, weights.ln_f_w, weights.ln_f_b]
    for lst in (
        weights.ln_1_w,
        weights.ln_1_b,
        weights.c_attn_w,
        weights.c_attn_b,
        weights.c_proj_w,
        weights.c_proj_b,
        weights.ln_2_w,
        weights.ln_2_b,
        weights.c_fc_w,
        weights.c_fc_b,
        weights.c_proj_mlp_w,
        weights.c_proj_mlp_b,
    ):
        arrays.extend(lst)
    return int(sum(a.nbytes for a in arrays))


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile; p in [0, 100]."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, p, method="linear"))


def summarize_latencies_ms(per_token_ms: list[float]) -> dict[str, float]:
    if not per_token_ms:
        return {
            "total_ms": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "stddev_ms": 0.0,
            "tokens_per_sec": 0.0,
            "first_token_ms": 0.0,
            "last_token_ms": 0.0,
        }
    arr = np.asarray(per_token_ms, dtype=np.float64)
    total = float(arr.sum())
    n = len(arr)
    return {
        "total_ms": total,
        "mean_ms": float(arr.mean()),
        "p50_ms": percentile(per_token_ms, 50),
        "p90_ms": percentile(per_token_ms, 90),
