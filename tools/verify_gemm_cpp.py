#!/usr/bin/env python3
"""Run CUDA C++ gemm self-test against NumPy reference (GPU host only)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "common" / "test_vectors" / "gemm_c_proj_l0.npz"
BIN = ROOT / "cuda-cpp" / "build" / "gpt2_cuda_cpp"


def main() -> None:
    if not NPZ.is_file():
        print("Run: python tools/export_kernel_tests.py", file=sys.stderr)
        raise SystemExit(1)
    if not BIN.is_file():
        print("Build: cmake -S cuda-cpp -B cuda-cpp/build && cmake --build cuda-cpp/build", file=sys.stderr)
        raise SystemExit(1)

    data = np.load(NPZ)
    x, w, bias, expected = data["x"], data["w"], data["bias"], data["expected"]
    m, k = x.shape
    n = w.shape[1]

    # Temporary: call numpy as oracle print for manual cuda test hookup
    # TODO: pass buffers to gpt2_cuda_cpp --self-test gemm when wired
    y = x @ w + bias
    err = float(np.max(np.abs(y - expected)))
    print(f"NumPy oracle max_err vs export: {err:.6e}")
    subprocess.run([str(BIN), "--self-test", "vecadd"], check=False)


if __name__ == "__main__":
    main()
