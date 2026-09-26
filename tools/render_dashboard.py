#!/usr/bin/env python3
"""Alias for render_demo.py (writes bench/runs/demo.html)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    bench = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "bench" / "runs"
    script = ROOT / "tools" / "render_demo.py"
    subprocess.run([sys.executable, str(script), str(bench)], check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
