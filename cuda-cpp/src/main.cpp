#include "gpt2_cuda/kernels.cuh"
#include "gpt2_cuda/weights.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static void print_usage() {
  std::fprintf(stderr,
               "Usage: gpt2_cuda_cpp --manifest PATH [--check-weights] [--self-test vecadd|gemm]\n");
}

int main(int argc, char** argv) {
  std::string manifest = "../common/weights/gpt2_manifest.json";
  std::string self_test;
  bool check_weights = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--manifest" && i + 1 < argc) manifest = argv[++i];
    else if (arg == "--check-weights") check_weights = true;
    else if (arg == "--self-test" && i + 1 < argc) self_test = argv[++i];
    else {
      print_usage();
      return 1;
    }
  }

#ifdef GPT2_HAS_CUDA
  if (self_test == "vecadd") {
    gpt2::cuda::vecadd_self_test();
    return 0;
  }
#endif

  try {
    auto store = gpt2::WeightStore::load(manifest);
    const auto& cfg = store->config();
    std::printf("Loaded GPT-2 weights: n_layer=%d n_embd=%d vocab=%d (%.1f MiB float payload)\n", cfg.n_layer,
                cfg.n_embd, cfg.vocab_size, store->total_bytes() * sizeof(float) / (1024.f * 1024.f));

    const auto* wte = store->find("wte");
    if (!wte || wte->shape.size() != 2) {
      std::fprintf(stderr, "wte missing or bad shape\n");
      return 1;
    }
    std::printf("wte shape: [%d, %d]\n", wte->shape[0], wte->shape[1]);

    if (check_weights) {
      const auto* h0 = store->find("h.0.attn.c_proj.weight");
      if (h0) std::printf("h.0.attn.c_proj.weight shape: [%d, %d]\n", h0->shape[0], h0->shape[1]);
      std::printf("weight check OK\n");
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }

#ifdef GPT2_HAS_CUDA
  if (self_test == "gemm") {
    std::fprintf(stderr, "gemm self-test: run tools/verify_gemm_cpp.py on GPU host (loads npz reference)\n");
    return 0;
  }
#else
  if (!self_test.empty()) {
    std::fprintf(stderr, "CUDA not compiled — self-test '%s' needs Linux + nvcc\n", self_test.c_str());
    return 1;
  }
  std::printf("(host-only build — rebuild on GPU machine for CUDA self-tests)\n");
#endif
  return 0;
}
