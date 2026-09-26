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
