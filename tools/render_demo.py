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
    *,
    embed_plots: bool,
    plot_b64: dict[str, str],
    asset_paths: dict[str, str],
) -> str:
    def img(tag: str, title: str) -> str:
        if embed_plots:
            src = f"data:image/png;base64,{plot_b64[tag]}"
        else:
            src = asset_paths[tag]
        return f"""
        <figure>
          <img src="{src}" alt="{html.escape(title)}"/>
          <figcaption>{html.escape(title)}</figcaption>
        </figure>
        """

    cards = "".join(io_block(d) for d in runs)
    uri = bench_dir.resolve().as_uri()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>GPT-2 NumPy CPU — demo dashboard</title>
  <style>
    :root {{ --bg:#0f1419; --fg:#e7ecf3; --muted:#8b9cb3; --line:#2a3544; --card:#1a2332; }}
    body {{ font-family: system-ui, sans-serif; margin: 0; background: var(--bg); color: var(--fg); }}
    header {{ padding: 20px 28px; border-bottom: 1px solid var(--line); }}
    header h1 {{ margin: 0 0 6px; font-size: 1.35rem; }}
    header p {{ margin: 0; color: var(--muted); font-size: 0.9rem; }}
    main {{ padding: 24px 28px 48px; max-width: 1100px; margin: 0 auto; }}
    h2 {{ font-size: 1.1rem; margin-top: 32px; }}
    section.plots {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
    figure {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px; margin: 0; }}
    figure img {{ width: 100%; height: auto; display: block; border-radius: 6px; }}
    figcaption {{ color: var(--muted); font-size: 0.8rem; margin-top: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.875rem; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 12px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); }}
    table.glossary td:first-child {{ white-space: nowrap; width: 12rem; }}
    .runs {{ display: grid; gap: 16px; }}
    .run-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }}
    .run-card h3 {{ margin: 0 0 4px; font-size: 1rem; }}
    .tag {{ color: var(--muted); font-size: 0.8rem; margin: 0 0 12px; }}
    .io pre {{ background: #0b0f14; padding: 10px; border-radius: 6px; overflow-x: auto;
               white-space: pre-wrap; word-break: break-word; font-size: 0.85rem; margin: 4px 0 12px; }}
    .lbl {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    .mini-metrics {{ margin: 0; padding-left: 18px; color: var(--muted); font-size: 0.85rem; }}
    code {{ color: #93c5fd; }}
  </style>
</head>
<body>
  <header>
    <h1>GPT-2 124M — NumPy CPU baseline</h1>
    <p>In-house demo dashboard · data: <code>{html.escape(str(bench_dir))}</code></p>
    <p>Local URL: <a href="{html.escape(uri)}/demo.html">{html.escape(uri)}/demo.html</a></p>
  </header>
  <main>
    {glossary_html()}

    <section id="charts">
      <h2>Charts (matplotlib)</h2>
      <div class="plots">
        {img("decode_curves", "Decode latency grows as context lengthens (same run, each new token).")}
        {img("latency_bars", "Compare prefill, mean decode, and TTFT across batch runs.")}
        {img("prefill_vs_decode", "Prefill cost vs mean decode cost; point size scales with prompt length.")}
        {img("throughput_memory", "End-to-end throughput vs peak RSS — resource/latency tradeoff snapshot.")}
      </div>
    </section>

    <section id="runs">
      <h2>Inputs &amp; outputs</h2>
      <div class="runs">{cards}</div>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bench_dir", type=Path, help="Directory with *.json bench files")
    p.add_argument("--out", type=Path, default=None, help="HTML output (default BENCH/demo.html)")
    p.add_argument("--open", action="store_true", help="Open demo.html in default browser")
    p.add_argument(
        "--embed",
        action="store_true",
        help="Embed PNGs as base64 in HTML (single file, larger)",
    )
    args = p.parse_args()

    bench = args.bench_dir
    runs = load_runs(bench)
    if not runs:
        print(f"No JSON in {bench}", file=sys.stderr)
        raise SystemExit(1)

    out_html = args.out or bench / "demo.html"
    assets = bench / "assets"

    plot_b64 = {
        "decode_curves": plot_decode_curves(runs),
        "latency_bars": plot_run_comparison_bars(runs),
        "throughput_memory": plot_throughput_memory(runs),
        "prefill_vs_decode": plot_prefill_vs_decode_scatter(runs),
    }
    asset_paths = save_pngs(runs, assets)

    page = render_html_page(
        runs,
        bench,
        embed_plots=args.embed,
        plot_b64=plot_b64,
        asset_paths=asset_paths,
    )
    out_html.write_text(page)
    print(f"Wrote {out_html.resolve()}")
    print(f"PNG assets: {(assets).resolve()}")

    if args.open:
        url = out_html.resolve().as_uri()
        print(f"Opening {url}")
        if not webbrowser.open(url):
            print("Could not open browser; paste the URL above.", file=sys.stderr)


if __name__ == "__main__":
    main()
