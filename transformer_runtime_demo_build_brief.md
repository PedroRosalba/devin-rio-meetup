# Transformer Runtime Demo Build Brief

## Objective
Implement the same GPT-2 forward pass in three paths:
1. CPU reference
2. CUDA C++
3. Rust + CUDA-Oxide

Validate numerical equivalence, then benchmark them on the same Linux NVIDIA machine.

## Scope decision
- Inference only for v1.
- Start GPT-2 124M, float32, greedy decoding.
- Keep model config generic so GPT-2 Medium/Large/XL can be loaded later.
- No Kimi/MoE, training, autograd, quantization, distributed inference, or FlashAttention in v1.
- PyTorch/Hugging Face may be used only as an external correctness oracle.

## Final demo
- Same prompt and checkpoint through all three runtimes.
- Show close logits and identical greedy tokens.
- Display CPU vs CUDA C++ vs Rust/CUDA-Oxide latency/tokens-per-second.
- Map each Transformer equation to the kernel executing it.
- Optional attention matrix visualization and optimization ladder.

## GPT-2 forward path
```text
Prompt -> tokenizer -> token IDs
-> token embedding + positional embedding
-> N x [
  LayerNorm
  QKV projection
  split heads
  QK^T / sqrt(d_head)
  causal mask
  softmax
  attention @ V
  output projection
  residual
  LayerNorm
  MLP linear
  GELU-new
  MLP linear
  residual
]
-> final LayerNorm
-> LM head
-> logits
-> greedy/sample next token
```

## Core primitives
- embedding lookup
- GEMM / linear + bias
- LayerNorm
- GELU-new
- causal mask
- stable softmax
- Q/K/V head reshape/split/merge
- attention matmuls
- residual add

## Repo skeleton
```text
transformer-runtime-demo/
├── common/
│   ├── tensor_layout.md
│   └── test_vectors/
├── cpu/
├── cuda_cpp/
├── rust_cuda/
│   └── src/kernels/
│       ├── gemm.rs
│       ├── layernorm.rs
│       ├── softmax.rs
│       ├── gelu.rs
│       ├── embedding.rs
│       └── elementwise.rs
├── tools/
│   ├── export_reference.py
│   └── compare_tensors.py
└── benches/
```

## Implementation order
1. Environment bring-up: CPU binary, CUDA C++ vecadd, `cargo oxide run vecadd`.
2. CPU float32 primitives + deterministic test vectors.
3. CUDA C++ naive GEMM vs CPU.
4. Rust/CUDA-Oxide naive GEMM vs CPU.
5. GELU-new, residual.
6. LayerNorm reductions.
7. Stable softmax.
8. Single-head attention.
9. Multi-head attention.
10. One GPT-2 block.
11. Load GPT-2 weights/tokenizer.
12. Export Hugging Face reference intermediates.
13. Match full forward logits.
14. Greedy generation.
15. Benchmark harness.
16. KV cache.
17. Tiled GEMM; optional f16/tensor cores.

## Correctness rules
- Never debug the full model first.
- Compare every kernel against CPU.
- Export a fixed Hugging Face reference prompt and compare:
  embeddings, LayerNorms, Q/K/V, attention, MLP, each block output, final logits.
- Keep tensor shapes explicit.
- Unit-test every transpose/reshape.
- Start float32.
- Do not introduce cuBLAS into the hand-written-kernel comparison; if added, make it a separate baseline.

## CUDA-Oxide environment
Current docs (23 Sep 2026) say:
- Linux; Ubuntu 24.04 tested
- NVIDIA Ampere+ (sm_80+)
- NVIDIA driver R580+
- CUDA Toolkit 13.0+
- LLVM 21+ with NVPTX
- Clang 21+
- pinned Rust nightly from cuda-oxide repo

Prefer the provided devcontainer or Nix environment.

First checks:
```bash
nvidia-smi
cargo oxide doctor
cargo oxide run vecadd
```

## Where to run
No API/model credits are required. You pay only for compute/storage if renting a GPU.

Current price snapshot:
- RunPod RTX A5000 24 GB: ~ $0.27/hr
- RunPod RTX 3090 24 GB: ~ $0.50/hr
- RunPod A40 48 GB: ~ $0.49/hr
- RunPod A6000 48 GB: ~ $0.53/hr
- Lambda A6000 48 GB: ~ $1.09/hr

For CUDA-Oxide, always verify `nvidia-smi` shows driver >= 580 on the actual rented host.

## Benchmark rules
- Same host/GPU, weights, token IDs, dtype.
- Separate load time from inference.
- Warm up.
- Measure prefill and decode separately.
- Use repeated runs; report median + tail percentile.
- CUDA events for GPU timing.
- Record GPU, driver, CUDA, compiler flags, model, sequence length, commit hash.

## Demo ideas for GPT-2
- CPU vs C++ vs Rust generation race.
- Story continuation from absurd prompt.
- Top-10 next-token probability viewer.
- Prompt branching with several random seeds.
- Attention visualization for one head/layer.
- Naive -> tiled -> tensor-core GEMM optimization ladder.

GPT-2 is a base LM, not an instruction-tuned model, so do not center the demo on IMO/problem-solving quality.

## Reading path
1. https://nvlabs.github.io/cuda-oxide/
2. https://nvlabs.github.io/cuda-oxide/getting-started/installation.html
3. https://nvlabs.github.io/cuda-oxide/getting-started/hello-gpu.html
4. https://nvlabs.github.io/cuda-oxide/gpu-programming/execution-model.html
5. https://nvlabs.github.io/cuda-oxide/gpu-programming/memory-and-data-movement.html
6. https://nvlabs.github.io/cuda-oxide/gpu-programming/kernels-and-device-functions.html
7. https://nvlabs.github.io/cuda-oxide/gpu-programming/launching-kernels.html
8. https://nvlabs.github.io/cuda-oxide/advanced/shared-memory-and-synchronization.html
9. https://nvlabs.github.io/cuda-oxide/advanced/warp-level-programming.html
10. https://nvlabs.github.io/cuda-oxide/projects/async-mlp-pipeline.html
11. https://nvlabs.github.io/cuda-oxide/advanced/matrix-multiply-accelerators.html
12. https://huggingface.co/docs/transformers/en/model_doc/gpt2
13. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

## Definition of done
A fresh compatible Linux GPU host can load GPT-2, run CPU/CUDA-C++/Rust-CUDA-Oxide, show close logits, generate identical greedy tokens, and print a benchmark comparison.
