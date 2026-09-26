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

