#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${MODEL_DIR:-$ROOT/models/gpt2}"
PYTHON="${PYTHON:-python3}"

echo "==> Python venv"
if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "==> pip upgrade (can take several minutes on first run — torch is large)"
pip install -U pip
pip install -r cpu/requirements.txt

echo "==> GPT-2 weights + tokenizer -> $MODEL_DIR"
mkdir -p "$MODEL_DIR"
export ROOT MODEL_DIR
python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["ROOT"]).resolve()
model_dir = Path(os.environ["MODEL_DIR"]).resolve()
model_dir.mkdir(parents=True, exist_ok=True)
from huggingface_hub import snapshot_download
snapshot_download(repo_id="openai-community/gpt2", local_dir=str(model_dir))
print("Downloaded to", model_dir)
PY

echo "==> Verify NumPy vs Hugging Face"
export MODEL_DIR
python tools/verify_hf.py --strict

echo "==> Smoke demo"
python cpu/run.py --prompt "Hello" --steps 1

echo "Done. Activate: source .venv/bin/activate"
