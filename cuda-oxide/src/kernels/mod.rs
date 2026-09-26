//! CUDA-Oxide kernels (GEMM, layernorm, softmax, …) — mirror `cuda-cpp/cuda/` and `cpu/gpt2.py`.
//!
//! Example first kernel (after oxide toolchain on GPU):
//! ```ignore
//! #[kernel]
//! pub fn vecadd(a: &[f32], b: &[f32], y: &mut [f32]) { ... }
//! ```
