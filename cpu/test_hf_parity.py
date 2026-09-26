"""unittest entrypoint for HF parity (wraps tools/verify_hf)."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestHFParity(unittest.TestCase):
    def test_numpy_matches_huggingface(self) -> None:
        env = os.environ.copy()
        model = ROOT / "models" / "gpt2"
        if model.is_dir():
            env["MODEL_DIR"] = str(model)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "verify_hf.py"), "--strict"],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"verify_hf failed:\n{proc.stdout}\n{proc.stderr}")


if __name__ == "__main__":
    unittest.main()
