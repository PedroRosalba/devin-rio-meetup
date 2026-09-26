# GPU setup (do this before renting long sessions)

## Rent checklist

- **OS:** Linux (Ubuntu 24.04 is a safe bet).
- **GPU:** Ampere or newer (e.g. RTX A5000, A6000, 3090).
- **Driver:** `nvidia-smi` must show **580+**.
- **CUDA Toolkit:** 13.0+ for CUDA-Oxide.

## CUDA C++

```bash
nvcc --version
# build when cuda-cpp/CMakeLists.txt exists:
# cmake -S cuda-cpp -B cuda-cpp/build && cmake --build cuda-cpp/build
```

## CUDA-Oxide (Rust)

Automated path (RunPod + SSH): **`docs/cuda-oxide-setup.md`**

```bash
export RUNPOD_API_KEY='...'   # after registering ~/.ssh/id_ed25519.pub on RunPod
bash scripts/runpod_provision_gpu.sh
bash scripts/runpod_remote_bootstrap.sh   # needs RUNPOD_SSH from script output
```

Manual checks on the pod:

```bash
bash scripts/cuda_oxide_host_setup.sh
# → cargo oxide doctor && cargo oxide run vecadd
```

## Model weights on the pod

Clone this repo and run `./scripts/setup_env.sh` so `models/gpt2/` is local (no API keys).

## Suggested RunPod tiers (snapshot)

| GPU | ~$/hr | Notes |
|-----|-------|--------|
| RTX A5000 24GB | ~0.27 | Enough for GPT-2 124M |
| RTX 3090 | ~0.50 | Ampere |
| A40 / A6000 | ~0.49–0.53 | More headroom |

Always verify **driver 580+** on the actual instance before configuring toolchains.
