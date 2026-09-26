#!/usr/bin/env python3
"""Educational CPU GPT-2 benchmark + generation demo (NumPy reference)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Tokenizer is the only non-NumPy runtime dependency for I/O.
from transformers import GPT2Tokenizer

from gpt2 import forward, load_gpt2_config, load_gpt2_weights, softmax
from metrics import (
    RunMetrics,
    count_parameters,
    print_report,
    read_memory_rss_bytes,
    weight_bytes,
    write_metrics_json,
)
from sampling import SamplingConfig, choose_next_token


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_model = root / "models" / "gpt2"

    p = argparse.ArgumentParser(description="GPT-2 NumPy CPU benchmark demo")
    p.add_argument("--model-dir", type=Path, default=Path(os.environ.get("MODEL_DIR", default_model)))
    p.add_argument(
        "--prompt",
        type=str,
        default="The strangest thing I found inside my refrigerator was",
    )
    p.add_argument("--steps", type=int, default=20, help="Number of tokens to generate after the prompt")

    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy argmax")
    p.add_argument("--top-k", dest="top_k", type=int, default=0, help="0 = disabled")
    p.add_argument("--top-p", dest="top_p", type=float, default=1.0, help="1.0 = disabled")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for sampling")
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Divide logits of tokens already in context by this value (>1 discourages repeats)",
    )

    p.add_argument(
        "--show-topk",
        type=int,
        default=10,
        help="Show top-N next-token probabilities after prefill",
    )
    p.add_argument("--verbose-timing", action="store_true", help="Print per generated token latency")
    p.add_argument("--metrics-json", type=Path, default=None, help="Write benchmark metrics JSON")
    p.add_argument("--run-label", type=str, default="", help="Label stored in metrics JSON (e.g. 01_greedy)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sampling = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    sampling.validate()
    if sampling.is_greedy and (args.top_k > 0 or args.top_p < 1.0 or args.repetition_penalty != 1.0):
        print(
            "Note: temperature=0 uses raw greedy argmax; top-k/top-p/repetition-penalty are ignored.",
            file=sys.stderr,
        )

    rng: np.random.Generator | None = None
    if not sampling.is_greedy:
        rng = np.random.default_rng(args.seed)

    startup_begin = time.perf_counter()

    t0 = time.perf_counter()
    cfg = load_gpt2_config(args.model_dir)
    config_load_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    weights = load_gpt2_weights(args.model_dir, cfg)
    weights_load_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    tok = GPT2Tokenizer.from_pretrained(str(args.model_dir))
