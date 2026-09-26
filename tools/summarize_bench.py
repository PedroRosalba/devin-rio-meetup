#!/usr/bin/env python3
"""Terminal metrics dashboard (no prompt/completion text). See report_bench.py for I/O."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt_mb(b: int | None) -> str:
    if b is None:
        return "n/a"
    return f"{b / 1e6:.0f}MB"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: summarize_bench.py BENCH_DIR", file=sys.stderr)
        raise SystemExit(2)

    bench = Path(sys.argv[1])
    files = sorted(bench.glob("*.json"))
    if not files:
        print(f"No JSON in {bench}")
        return

    rows: list[tuple[str, str, float, float, float, float, str]] = []
    for f in files:
        d = load(f)
        dec = d["decode"]
        gen = d["generation"]
        mem = d["memory"]
        strat = gen["strategy"]
        if gen["temperature"] > 0:
            strat = f"T={gen['temperature']} p={gen['top_p']} k={gen['top_k']}"
        rows.append(
            (
                f.stem,
                strat,
                d["prefill"]["latency_ms"],
                dec["mean_ms"],
                dec["max_ms"] - dec["min_ms"],
                d["end_to_end"]["overall_tokens_per_sec"],
                fmt_mb(mem.get("peak_rss_bytes")),
            )
        )

    print("\n=== BENCHMARK BATCH SUMMARY (NumPy CPU baseline) ===\n")
    hdr = f"{'run':<28} {'decode':<22} {'prefill':>8} {'mean_t':>8} {'t_spread':>9} {'tok/s':>7} {'peak':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r[0]:<28} {r[1]:<22} {r[2]:8.1f} {r[3]:8.1f} {r[4]:9.1f} {r[5]:7.2f} {r[6]:>8}")

    # Per-token sparkline from latest timing run if present
    timing = bench / "02_greedy_timing.json"
    if timing.is_file():
        pts = load(timing)["decode"].get("per_token_ms") or []
        if pts:
            mx = max(pts)
            bar = lambda v: "#" * max(1, int(40 * v / mx)) if mx else ""
            print("\nDecode latency curve (02_greedy_timing):")
            for i, ms in enumerate(pts, 1):
                print(f"  {i:2d}  {ms:6.1f} ms  {bar(ms)}")
        print()


if __name__ == "__main__":
    main()
