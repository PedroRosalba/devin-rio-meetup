# CUDA C++ (GPT-2 inference)

Mirrors `cpu/gpt2.py` kernel boundaries. **Inference only.**

## Mac (now): host-only build

```bash
python tools/export_gpt2_weights.py
python tools/export_kernel_tests.py

cmake -S cuda-cpp -B cuda-cpp/build
cmake --build cuda-cpp/build
./cuda-cpp/build/gpt2_cuda_cpp --manifest common/weights/gpt2_manifest.json --check-weights
```

No GPU required — validates exported weights + manifest parsing.

## Linux GPU (Modal / pod)

```bash
cmake -S cuda-cpp -B cuda-cpp/build
cmake --build cuda-cpp/build -j
./cuda-cpp/build/gpt2_cuda_cpp --self-test vecadd
# next: full forward + benchmark JSON
```

Weights: `common/weights/gpt2_manifest.json` + `gpt2_weights_fp32.bin` (Conv1D `[in,out]`).
