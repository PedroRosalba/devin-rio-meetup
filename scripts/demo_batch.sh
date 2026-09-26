#!/usr/bin/env bash
# Sequential demo runs for live explanation. Each step writes metrics JSON.
set -euo pipefail

PROMPT="${1:?usage: demo_batch.sh PROMPT BENCH_DIR}"
BENCH_DIR="${2:?usage: demo_batch.sh PROMPT BENCH_DIR}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PY="${PYTHON:-python}"
RUN=( "$PY" cpu/run.py --prompt "$PROMPT" )

echo ""
echo "========== 1/5 CORRECTNESS (already ran via make verify) =========="
echo "Message: NumPy reference matches Hugging Face logits + greedy tokens."
echo ""

echo "========== 2/5 GREEDY BASELINE (honest base LM) =========="
"${RUN[@]}" --steps 8 --temperature 0 \
  --run-label 01_greedy --metrics-json "$BENCH_DIR/01_greedy.json"

echo ""
echo "========== 3/5 TOP LOGITS (what the model thinks is next) =========="
echo "(Included in run output above; same prefill as step 2.)"
echo ""

echo "========== 4/5 DECODE COST CURVE (no KV cache) =========="
"${RUN[@]}" --steps 12 --temperature 0 --verbose-timing \
  --run-label 02_greedy_timing --metrics-json "$BENCH_DIR/02_greedy_timing.json"

echo ""
echo "========== 5/5 SAMPLING (same weights, nicer text) =========="
"${RUN[@]}" --steps 16 --temperature 0.85 --top-p 0.92 --top-k 50 --seed 42 \
  --run-label 03_sample_topp --metrics-json "$BENCH_DIR/03_sample_topp.json"

echo ""
echo "Optional: repetition penalty vs greedy loop"
"${RUN[@]}" --steps 12 --temperature 0.8 --top-p 0.9 --seed 7 \
  --repetition-penalty 1.15 \
  --run-label 04_sample_rep_penalty --metrics-json "$BENCH_DIR/04_sample_rep_penalty.json"

echo ""
echo "Batch complete. JSON under: $BENCH_DIR"
