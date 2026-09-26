"""Remote GPU sanity check from your Mac: modal run modal/gpu_smoke.py"""

import modal

app = modal.App("devin-meetup-rio-gpu-smoke")

cuda_image = modal.Image.from_registry(
    "nvidia/cuda:13.0.0-devel-ubuntu24.04",
    add_python="3.11",
).apt_install("curl")


@app.function(image=cuda_image, gpu="T4", timeout=120)
def nvidia_smi() -> str:
    import subprocess

    return subprocess.check_output(["nvidia-smi"], text=True)


@app.local_entrypoint()
def main() -> None:
    print(nvidia_smi.remote())
