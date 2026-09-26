# 2–3 day plan (simple first)

## Before the hack (your Mac)

```bash
./scripts/setup_env.sh
# optional: HF_TOKEN=... for faster re-downloads
```

Weights land in `models/gpt2/` (~500MB). No OpenAI API.

## Day 1 — CPU truth + demo script

- [x] Explicit `cpu/gpt2.py` (embed → blocks → ln_f → tied LM head)
- [x] `cpu/run.py` — prefill ms, top-k logits, greedy continuation
- [x] `tools/verify_hf.py` — logits + hidden states + greedy vs HF (`--strict`)

**Talk slice already works:** absurd prompt + timings + top tokens.

## Day 2 — GPU pod (C++ *or* Rust first, not both required for v0)

Rent A5000/A6000, `nvidia-smi` driver ≥ 580.

1. VecAdd + H2D copy
2. **One** path: naive GEMM + softmax matching CPU on tiny shapes
3. If time: full forward in CUDA for **prefill only** (no KV cache)

## Day 3 — second runtime or polish

- Second CUDA implementation **or** side-by-side bench table in `run.py`
- Slides: equation → function name in `gpt2.py` → kernel name

## Explicitly later

- KV cache, tiled GEMM, tensor cores, CI, attention heatmap UI
- GPT-2 Medium/Large (same code, swap checkpoint)
