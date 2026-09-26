"""Matplotlib figures for bench JSON (Agg backend, no GUI required)."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Dark theme aligned with dashboard HTML
BG = "#0f1419"
FG = "#e7ecf3"
GRID = "#2a3544"
COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#ef4444", "#06b6d4"]


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.35, linewidth=0.6)


def run_label(d: dict[str, Any]) -> str:
    return d.get("_stem") or d.get("run_label") or "run"


def decode_label(d: dict[str, Any]) -> str:
    g = d["generation"]
    if g["temperature"] > 0:
        return f"T={g['temperature']}"
    return "greedy"


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_decode_curves(runs: list[dict[str, Any]]) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    _style_ax(ax)
    for i, d in enumerate(runs):
        pts = d["decode"].get("per_token_ms") or []
        if not pts:
            continue
        xs = list(range(1, len(pts) + 1))
        ax.plot(
            xs,
            pts,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=COLORS[i % len(COLORS)],
            label=f"{run_label(d)} ({decode_label(d)})",
        )
    ax.set_xlabel("Generated token index (context grows each step)")
    ax.set_ylabel("Forward pass latency (ms)")
    ax.set_title("Decode cost vs token index (no KV cache)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    return fig_to_base64(fig)


def plot_run_comparison_bars(runs: list[dict[str, Any]]) -> str:
    labels = [run_label(d) for d in runs]
    x = range(len(labels))
    w = 0.25
    prefill = [d["prefill"]["latency_ms"] for d in runs]
    mean_dec = [d["decode"]["mean_ms"] for d in runs]
    ttft = [d["end_to_end"]["ttft_ms"] for d in runs]

    fig, ax = plt.subplots(figsize=(9, 4))
    _style_ax(ax)
    ax.bar([i - w for i in x], prefill, width=w, label="prefill_ms", color=COLORS[0])
    ax.bar(x, mean_dec, width=w, label="mean decode ms/token", color=COLORS[1])
    ax.bar([i + w for i in x], ttft, width=w, label="TTFT ms", color=COLORS[2])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", color=FG)
    ax.set_ylabel("milliseconds")
    ax.set_title("Latency breakdown by run")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    return fig_to_base64(fig)


def plot_throughput_memory(runs: list[dict[str, Any]]) -> str:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    _style_ax(ax1)
    labels = [run_label(d) for d in runs]
    x = list(range(len(labels)))
    tps = [d["end_to_end"]["overall_tokens_per_sec"] for d in runs]
    bars = ax1.bar(x, tps, color=COLORS[0], alpha=0.85, label="overall tok/s")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", color=FG)
    ax1.set_ylabel("tokens / sec (prefill + decode)")
    ax1.set_title("Throughput vs peak memory")

    ax2 = ax1.twinx()
    peak_mb = []
    for d in runs:
        b = d["memory"].get("peak_rss_bytes")
        peak_mb.append(b / 1e6 if b else 0)
    ax2.plot(x, peak_mb, color=COLORS[3], marker="D", linewidth=2, label="peak RSS (MB)")
    ax2.set_ylabel("peak RSS (MB)")
    ax2.tick_params(colors=FG)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    return fig_to_base64(fig)


def plot_prefill_vs_decode_scatter(runs: list[dict[str, Any]]) -> str:
    """Shows that prefill and per-token decode are related but not identical costs."""
    fig, ax = plt.subplots(figsize=(6, 5))
    _style_ax(ax)
    for i, d in enumerate(runs):
        px = d["prefill"]["latency_ms"]
        py = d["decode"]["mean_ms"]
        n = d["input"]["prompt_tokens"]
        ax.scatter(px, py, s=80 + n * 4, color=COLORS[i % len(COLORS)], alpha=0.9, edgecolors=FG)
        ax.annotate(
            run_label(d),
            (px, py),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=FG,
        )
    ax.set_xlabel("prefill latency (ms)")
    ax.set_ylabel("mean decode latency (ms / token)")
    ax.set_title("Prefill vs decode (bubble size ≈ prompt tokens)")
    return fig_to_base64(fig)


def save_pngs(runs: list[dict[str, Any]], assets: Path) -> dict[str, str]:
    """Write PNG files and return name -> relative path."""
    assets.mkdir(parents=True, exist_ok=True)
    makers = {
        "decode_curves": plot_decode_curves,
        "latency_bars": plot_run_comparison_bars,
        "throughput_memory": plot_throughput_memory,
        "prefill_vs_decode": plot_prefill_vs_decode_scatter,
    }
    paths: dict[str, str] = {}
    for name, fn in makers.items():
        b64 = fn(runs)
        out = assets / f"{name}.png"
        out.write_bytes(base64.b64decode(b64))
        paths[name] = f"assets/{name}.png"
    return paths
