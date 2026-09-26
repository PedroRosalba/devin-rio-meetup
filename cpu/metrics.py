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
        "p95_ms": percentile(per_token_ms, 95),
        "p99_ms": percentile(per_token_ms, 99),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "stddev_ms": float(arr.std(ddof=0)),
        "tokens_per_sec": (n / total * 1000.0) if total > 0 else 0.0,
        "first_token_ms": float(arr[0]),
        "last_token_ms": float(arr[-1]),
    }


def read_memory_rss_bytes() -> tuple[int | None, int | None, str]:
    """
    Returns (current_rss, peak_rss, notes).
    ru_maxrss units: bytes on macOS, kilobytes on Linux.
    """
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except OSError:
        return None, None, "resource.getrusage unavailable on this platform"
    peak = usage.ru_maxrss
    if sys.platform == "darwin":
        peak_bytes = int(peak)
        note = "peak_rss from getrusage (bytes on macOS)"
    else:
        peak_bytes = int(peak) * 1024
        note = "peak_rss from getrusage (ru_maxrss is KiB on Linux)"
    # Current RSS not provided by stdlib portably; mirror peak for peak field only.
    return None, peak_bytes, note


@dataclass
class RunMetrics:
    model_name: str = "gpt2"
    parameter_count: int = 0
    dtype: str = "float32"
    weight_bytes: int = 0
    backend: str = "NumPy CPU"
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    numpy_version: str = field(default_factory=lambda: np.__version__)
    platform: str = field(default_factory=platform.platform)

    prompt_chars: int = 0
    prompt_bytes: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_sequence_tokens: int = 0

    config_load_ms: float = 0.0
    weights_load_ms: float = 0.0
    tokenizer_load_ms: float = 0.0
    startup_total_ms: float = 0.0

    prefill_ms: float = 0.0
    decode_per_token_ms: list[float] = field(default_factory=list)

    ttft_ms: float = 0.0
    total_request_ms: float = 0.0

    strategy: str = "greedy"
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None
    repetition_penalty: float = 1.0

    memory_note: str = ""
    peak_rss_bytes: int | None = None

    run_label: str = ""
    prompt_text: str = ""
    completion_text: str = ""
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)

    definitions: dict[str, str] = field(
        default_factory=lambda: {
            "ttft_ms": "Time from end of startup through prefill until first decode step completes "
            "(first generated token available).",
            "total_request_ms": "startup_total_ms + prefill_ms + sum(decode per-token latencies).",
            "overall_tokens_per_sec": "generated_tokens / (prefill_ms + decode_total_ms) * 1000, "
            "decode-only portion excludes startup.",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        decode = summarize_latencies_ms(self.decode_per_token_ms)
        decode_tokens = self.generated_tokens
        infer_ms = self.prefill_ms + decode["total_ms"]
        overall_tps = (decode_tokens / infer_ms * 1000.0) if infer_ms > 0 and decode_tokens else 0.0
        prompt_tps = (self.prompt_tokens / self.prefill_ms * 1000.0) if self.prefill_ms > 0 else 0.0
        ms_per_prompt_token = self.prefill_ms / self.prompt_tokens if self.prompt_tokens else 0.0

        generated_suffix = self.generated_token_ids
        return {
            "schema_version": 2,
            "run_label": self.run_label,
            "model": {
                "name": self.model_name,
                "parameter_count": self.parameter_count,
                "dtype": self.dtype,
                "weight_bytes": self.weight_bytes,
                "backend": self.backend,
                "python_version": self.python_version,
                "numpy_version": self.numpy_version,
                "platform": self.platform,
            },
            "input": {
                "prompt_chars": self.prompt_chars,
                "prompt_bytes": self.prompt_bytes,
                "prompt_tokens": self.prompt_tokens,
                "generated_tokens": self.generated_tokens,
                "total_sequence_tokens": self.total_sequence_tokens,
            },
            "io": {
                "prompt_text": self.prompt_text,
                "completion_text": self.completion_text,
                "generated_suffix_text": (
                    self.completion_text[len(self.prompt_text) :]
                    if self.prompt_text and self.completion_text.startswith(self.prompt_text)
                    else ""
                ),
                "prompt_token_ids": list(self.prompt_token_ids),
                "generated_token_ids": list(generated_suffix),
            },
            "startup": {
                "config_load_ms": self.config_load_ms,
                "weights_load_ms": self.weights_load_ms,
                "tokenizer_load_ms": self.tokenizer_load_ms,
                "total_ms": self.startup_total_ms,
            },
            "prefill": {
                "latency_ms": self.prefill_ms,
                "prompt_tokens_per_sec": prompt_tps,
                "ms_per_prompt_token": ms_per_prompt_token,
            },
            "decode": {**decode, "per_token_ms": list(self.decode_per_token_ms)},
            "end_to_end": {
                "ttft_ms": self.ttft_ms,
                "total_request_ms": self.total_request_ms,
                "overall_tokens_per_sec": overall_tps,
                "definitions": self.definitions,
            },
            "memory": {
                "peak_rss_bytes": self.peak_rss_bytes,
                "note": self.memory_note,
            },
            "generation": {
                "strategy": self.strategy,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "seed": self.seed,
                "repetition_penalty": self.repetition_penalty,
            },
        }


def print_report(m: RunMetrics, d: dict[str, Any]) -> None:
    model = d["model"]
    inp = d["input"]
    startup = d["startup"]
    prefill = d["prefill"]
    decode = d["decode"]
    e2e = d["end_to_end"]
    mem = d["memory"]
    gen = d["generation"]

    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)

    print("\n[MODEL]")
    print(f"  name              {model['name']}")
    print(f"  parameters        {model['parameter_count']:,}")
    print(f"  dtype             {model['dtype']}")
    print(f"  weight_bytes      {model['weight_bytes']:,} ({model['weight_bytes'] / 1e6:.1f} MB)")
    print(f"  backend           {model['backend']}")
    print(f"  python            {model['python_version']}")
    print(f"  numpy             {model['numpy_version']}")

    print("\n[INPUT]")
    print(f"  prompt_chars      {inp['prompt_chars']}")
    print(f"  prompt_bytes      {inp['prompt_bytes']}")
    print(f"  prompt_tokens     {inp['prompt_tokens']}")
    print(f"  generated_tokens  {inp['generated_tokens']}")
    print(f"  total_seq_tokens  {inp['total_sequence_tokens']}")

    print("\n[STARTUP]")
    print(f"  config_load_ms    {startup['config_load_ms']:.2f}")
    print(f"  weights_load_ms   {startup['weights_load_ms']:.2f}")
    print(f"  tokenizer_load_ms {startup['tokenizer_load_ms']:.2f}")
    print(f"  total_ms          {startup['total_ms']:.2f}")

    print("\n[PREFILL]")
    print(f"  latency_ms        {prefill['latency_ms']:.2f}")
    print(f"  prompt_tok/sec    {prefill['prompt_tokens_per_sec']:.2f}")
    print(f"  ms/prompt_tok     {prefill['ms_per_prompt_token']:.2f}")

    print("\n[DECODE] (per generated token)")
    print(f"  total_ms          {decode['total_ms']:.2f}")
    print(f"  mean_ms/token     {decode['mean_ms']:.2f}")
    print(f"  p50_ms/token      {decode['p50_ms']:.2f}")
    print(f"  p90_ms/token      {decode['p90_ms']:.2f}")
    print(f"  p95_ms/token      {decode['p95_ms']:.2f}")
    print(f"  p99_ms/token      {decode['p99_ms']:.2f}")
    print(f"  min_ms/token      {decode['min_ms']:.2f}")
    print(f"  max_ms/token      {decode['max_ms']:.2f}")
    print(f"  stddev_ms/token   {decode['stddev_ms']:.2f}")
    print(f"  decode_tok/sec    {decode['tokens_per_sec']:.2f}")
    print(f"  first_token_ms    {decode['first_token_ms']:.2f}")
    print(f"  last_token_ms     {decode['last_token_ms']:.2f}")

    print("\n[END TO END]")
    print(f"  TTFT_ms           {e2e['ttft_ms']:.2f}")
    print(f"  total_request_ms  {e2e['total_request_ms']:.2f}")
    print(f"  overall_tok/sec   {e2e['overall_tokens_per_sec']:.2f}")
    print("  definitions:")
    for k, v in e2e["definitions"].items():
        print(f"    {k}: {v}")

    print("\n[MEMORY]")
    if mem["peak_rss_bytes"] is not None:
        print(f"  peak_rss_bytes    {mem['peak_rss_bytes']:,}")
    else:
        print("  peak_rss_bytes    (unavailable)")
    print(f"  note              {mem['note']}")

    print("\n[GENERATION]")
    print(f"  strategy          {gen['strategy']}")
    print(f"  temperature       {gen['temperature']}")
    print(f"  top_k             {gen['top_k']}")
    print(f"  top_p             {gen['top_p']}")
    print(f"  seed              {gen['seed']}")
    print(f"  repetition_penalty {gen['repetition_penalty']}")
    print("=" * 60 + "\n")


def write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
