#!/usr/bin/env python3
"""Human-readable batch report: prompt, completion, and metrics per run."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt_gen(d: dict) -> str:
    g = d["generation"]
    if g["temperature"] > 0:
        return f"sampling T={g['temperature']} top_p={g['top_p']} top_k={g['top_k']} seed={g['seed']}"
    return "greedy"


def section(title: str, width: int = 72) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}\n"


def render_run(name: str, d: dict) -> str:
    io = d.get("io") or {}
    pre = d["prefill"]
    dec = d["decode"]
    e2e = d["end_to_end"]
    mem = d["memory"]
    lines = [
        section(name),
        "[I/O]",
        f"  prompt:     {io.get('prompt_text') or '(missing — re-run with current cpu/run.py)'}",
        f"  completion: {io.get('completion_text') or '(missing)'}",
    ]
    suffix = io.get("generated_suffix_text")
    if suffix:
        lines.append(f"  suffix only: {suffix!r}")
    lines.extend(
        [
            "",
            "[GENERATION]",
            f"  {fmt_gen(d)}",
            f"  repetition_penalty: {d['generation']['repetition_penalty']}",
            "",
            "[METRICS]",
            f"  prefill_ms:        {pre['latency_ms']:.2f}",
            f"  decode_mean_ms:    {dec['mean_ms']:.2f}",
            f"  decode_p95_ms:     {dec['p95_ms']:.2f}",
            f"  ttft_ms:           {e2e['ttft_ms']:.2f}",
            f"  overall_tok/s:     {e2e['overall_tokens_per_sec']:.2f}",
            f"  peak_rss:          {mem.get('peak_rss_bytes')}",
            f"  prompt_tokens:     {d['input']['prompt_tokens']}",
            f"  generated_tokens:  {d['input']['generated_tokens']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: report_bench.py BENCH_DIR [OUT.txt]", file=sys.stderr)
        raise SystemExit(2)
    bench = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else bench / "report.txt"
    files = sorted(bench.glob("*.json"))
    if not files:
        print(f"No JSON in {bench}")
        return
    body = "".join(render_run(f.stem, load(f)) for f in files)
    out_path.write_text(body)
    print(body)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
