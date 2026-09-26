#!/usr/bin/env python3
"""
Build demo HTML + matplotlib PNGs from bench/runs/*.json.
Optional: open file:// URL in the default browser.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from pathlib import Path

from bench_plots import (
    plot_decode_curves,
    plot_prefill_vs_decode_scatter,
    plot_run_comparison_bars,
    plot_throughput_memory,
    run_label,
    save_pngs,
)

METRIC_HELP: list[tuple[str, str]] = [
    ("prefill_ms", "One forward pass over the prompt tokens (attention over prompt length)."),
    ("mean decode ms/token", "Average time for each greedy/sample step; full sequence re-forward (no KV cache)."),
    ("p95 decode ms/token", "95th percentile of per-token decode latencies — tail behavior."),
    ("TTFT ms", "Prefill + first decode step: time until the first new token is ready (excludes model load)."),
    ("overall tok/s", "generated_tokens ÷ (prefill + decode); excludes startup/load."),
    ("peak RSS", "Process high-water memory (macOS: bytes from getrusage)."),
    (
        "decode curve",
        "Token index vs ms — upward drift hints at O(T²) attention cost as context grows.",
    ),
]


def load_runs(bench: Path) -> list[dict]:
    runs = []
    for f in sorted(bench.glob("*.json")):
        d = json.loads(f.read_text())
        d["_stem"] = f.stem
        runs.append(d)
    return runs


def io_block(d: dict) -> str:
    io = d.get("io") or {}
    prompt = io.get("prompt_text") or "(re-run demo-batch to capture prompt in JSON)"
    completion = io.get("completion_text") or "(missing)"
    suffix = io.get("generated_suffix_text") or ""
    g = d["generation"]
    strat = (
        f"greedy"
        if g["temperature"] == 0
        else f"T={g['temperature']} top_p={g['top_p']} top_k={g['top_k']} seed={g['seed']}"
    )
    return f"""
    <article class="run-card">
      <h3>{html.escape(run_label(d))}</h3>
      <p class="tag">{html.escape(strat)}</p>
      <div class="io">
        <div><span class="lbl">Prompt</span><pre>{html.escape(prompt)}</pre></div>
        <div><span class="lbl">Completion</span><pre>{html.escape(completion)}</pre></div>
        {f'<div><span class="lbl">Generated suffix</span><pre>{html.escape(suffix)}</pre></div>' if suffix else ''}
      </div>
      <ul class="mini-metrics">
        <li>prefill: {d['prefill']['latency_ms']:.1f} ms</li>
        <li>mean decode: {d['decode']['mean_ms']:.1f} ms/tok</li>
        <li>TTFT: {d['end_to_end']['ttft_ms']:.1f} ms</li>
        <li>tok/s: {d['end_to_end']['overall_tokens_per_sec']:.2f}</li>
      </ul>
    </article>
    """


def glossary_html() -> str:
    rows = "".join(
        f"<tr><td><code>{html.escape(k)}</code></td><td>{html.escape(v)}</td></tr>"
        for k, v in METRIC_HELP
    )
    return f"""
    <section id="glossary">
      <h2>Metrics glossary</h2>
      <table class="glossary">
        <thead><tr><th>Metric</th><th>Meaning</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    """


def render_html_page(
    runs: list[dict],
    bench_dir: Path,
