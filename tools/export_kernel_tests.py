#!/usr/bin/env python3
"""Small reference tensors for GPU kernel bring-up (GEMM / vecadd)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cpu"))

from gpt2 import linear, load_gpt2  # noqa: E402


def main() -> None:
    out = ROOT / "common" / "test_vectors"
    out.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models" / "gpt2"
    _, w = load_gpt2(model_dir)

    # Vecadd: a,b length 1024
    rng = np.random.default_rng(0)
    a = rng.standard_normal(1024, dtype=np.float32)
    b = rng.standard_normal(1024, dtype=np.float32)
    np.savez(out / "vecadd.npz", a=a, b=b, expected=a + b)

    # GEMM: one real Conv1D from layer 0 (768 -> 768 slice of c_proj for smaller test use 768x768)
    w_mat = w.c_proj_w[0]  # [768, 768]
    x = rng.standard_normal((4, 768), dtype=np.float32)
    y = linear(x, w_mat, w.c_proj_b[0])
    np.savez(out / "gemm_c_proj_l0.npz", x=x, w=w_mat, bias=w.c_proj_b[0], expected=y)

    meta = {
        "vecadd": {"file": "vecadd.npz", "n": 1024},
        "gemm_c_proj_l0": {"file": "gemm_c_proj_l0.npz", "m": 4, "k": 768, "n": 768},
    }
    (out / "kernel_tests.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("Wrote", out)


if __name__ == "__main__":
    main()
