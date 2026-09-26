# Modal (Python cloud runner)

[Modal](https://modal.com/) is a **Python SDK** for running functions and jobs on Modal’s cloud workers. There is no first-party Modal client for C++ or Rust.

| Runtime in this repo | How it runs today | With Modal |
|----------------------|-------------------|------------|
| `cpu/` (NumPy reference) | Local venv | `@app.function()` + image with `cpu/requirements.txt` |
| `cuda-cpp/` | Native binary on a GPU pod | Custom **image** (CUDA toolkit), build in `Image.run_commands`, invoke binary via `subprocess` or shell |
| `cuda-oxide/` | `cargo oxide` on Ampere Linux | Same pattern: Rust/CUDA in the container build, run artifact on `@app.function(gpu=...)` |

Typical GPU pattern (later, when kernels exist):

```python
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("cmake", "build-essential")
    .copy_local_dir("cuda-cpp", "/root/cuda-cpp")
    .run_commands("cmake -S /root/cuda-cpp -B /build && cmake --build /build")
)

@app.function(image=image, gpu="A10G")
def run_vecadd():
    import subprocess
    subprocess.check_call(["/build/vecadd"])
```

For this meetup, Modal is optional: **RunPod + SSH** (`docs/gpu-setup.md`) matches native CUDA-Oxide devcontainers; Modal is useful if you want Python-orchestrated remote GPU without managing the pod yourself.

## Setup (once per machine)

From repo root:

```bash
./scripts/setup_env.sh          # venv + CPU deps (if not done)
pip install -r modal/requirements.txt
python3 -m modal setup          # browser login → API token
```

## Smoke test

```bash
source .venv/bin/activate
modal run modal/get_started.py
```

You should see `the square is 1764` after the remote worker prints its line.

## GPU from your Mac (no local NVIDIA GPU)

```bash
modal run modal/gpu_smoke.py          # remote nvidia-smi (T4)
modal run modal/gpu_cpp_vecadd.py     # build cuda-cpp + --self-test vecadd
```

CUDA-Oxide (`cargo oxide run vecadd`) still needs a **Linux GPU host** — RunPod + SSH: `docs/cuda-oxide-setup.md`.
