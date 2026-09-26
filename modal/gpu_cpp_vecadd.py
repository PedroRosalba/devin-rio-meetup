"""Build cuda-cpp on a Modal GPU worker and run --self-test vecadd.

  modal run modal/gpu_cpp_vecadd.py
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]

app = modal.App("devin-meetup-rio-gpu-cpp-vecadd")

build_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install("cmake", "build-essential")
    .add_local_dir(ROOT / "cuda-cpp", "/src/cuda-cpp", copy=True)
    .add_local_dir(ROOT / "common", "/src/common", copy=True)
    .run_commands(
        "cmake -S /src/cuda-cpp -B /build -DGPT2_BUILD_CUDA=ON",
        "cmake --build /build -j",
    )
)


@app.function(image=build_image, gpu="T4", timeout=600)
def run_vecadd() -> int:
    import subprocess

    proc = subprocess.run(
        ["/build/gpt2_cuda_cpp", "--self-test", "vecadd"],
        cwd="/src",
        check=False,
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    print(proc.stderr, end="")
    return proc.returncode


@app.local_entrypoint()
def main() -> None:
    code = run_vecadd.remote()
    if code != 0:
        raise SystemExit(code)
    print("vecadd self-test OK on Modal GPU")
