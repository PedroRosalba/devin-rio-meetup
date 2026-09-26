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
    tokenizer_load_ms = (time.perf_counter() - t0) * 1000

    startup_total_ms = (time.perf_counter() - startup_begin) * 1000

    prompt = args.prompt
    input_ids = tok.encode(prompt, add_special_tokens=False)
    print(f"Prompt tokens ({len(input_ids)}): {input_ids}")

    # --- Prefill ---
    t0 = time.perf_counter()
    arr = np.array(input_ids, dtype=np.int64)
    logits = forward(arr, cfg, weights)
    prefill_ms = (time.perf_counter() - t0) * 1000

    if args.show_topk > 0:
        probs = softmax(logits[-1])
        top_idx = np.argsort(probs)[::-1][: args.show_topk]
        print(f"\nTop next-token probabilities (last prompt position):")
        for i in top_idx:
            piece = tok.decode([int(i)])
            print(f"  {piece!r:20} {probs[i]:.4f}")

    # --- Decode (per-token timing) ---
    ids = list(input_ids)
    decode_ms: list[float] = []
    ttft_ms = 0.0

    for step in range(args.steps):
        if len(ids) >= cfg.n_ctx:
            break
        t_step = time.perf_counter()
        arr = np.array(ids, dtype=np.int64)
        logits = forward(arr, cfg, weights)
        next_id = choose_next_token(logits, ids, sampling, rng)
        step_ms = (time.perf_counter() - t_step) * 1000
        decode_ms.append(step_ms)
        if step == 0:
            ttft_ms = prefill_ms + step_ms
        ids.append(next_id)

        if args.verbose_timing:
            piece = tok.decode([next_id])
            print(f"token {step + 1:3d} | id={next_id:5d} | {piece!r:12} | {step_ms:7.1f} ms")

    generated = len(ids) - len(input_ids)
    decode_total_ms = sum(decode_ms)
    total_request_ms = startup_total_ms + prefill_ms + decode_total_ms

    _, peak_rss, mem_note = read_memory_rss_bytes()

    metrics = RunMetrics(
        model_name=args.model_dir.name,
        parameter_count=count_parameters(weights),
        weight_bytes=weight_bytes(weights),
        prompt_chars=len(prompt),
        prompt_bytes=len(prompt.encode("utf-8")),
        prompt_tokens=len(input_ids),
        generated_tokens=generated,
        total_sequence_tokens=len(ids),
        config_load_ms=config_load_ms,
        weights_load_ms=weights_load_ms,
        tokenizer_load_ms=tokenizer_load_ms,
        startup_total_ms=startup_total_ms,
        prefill_ms=prefill_ms,
        decode_per_token_ms=decode_ms,
        ttft_ms=ttft_ms if decode_ms else prefill_ms,
        total_request_ms=total_request_ms,
        strategy=sampling.strategy_label,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
        memory_note=mem_note,
        peak_rss_bytes=peak_rss,
        run_label=args.run_label or (args.metrics_json.stem if args.metrics_json else ""),
        prompt_text=prompt,
        completion_text=tok.decode(ids),
        prompt_token_ids=list(input_ids),
        generated_token_ids=ids[len(input_ids) :],
    )
    payload = metrics.to_dict()
    print_report(metrics, payload)

    text = metrics.completion_text
    print("--- completion ---\n")
    print(text)

    if args.metrics_json:
        write_metrics_json(args.metrics_json, payload)
        print(f"Wrote metrics JSON: {args.metrics_json}")


if __name__ == "__main__":
    main()
