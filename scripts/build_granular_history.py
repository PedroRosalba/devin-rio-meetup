#!/usr/bin/env python3
"""One-time helper: replay trackable files as many small git commits."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".pptx", ".gif", ".webp", ".ico"}
TARGET_COMMITS = 118
MIN_CHUNK = 24
MAX_CHUNK = 96
SKIP_PATHS = {"scripts/build_granular_history.py"}

# Narrative order (folder phases for meetup demo repo).
PHASES: list[tuple[str, list[str]]] = [
    (
        "chore",
        [
            ".gitignore",
            "Makefile",
            "README.md",
            "transformer_runtime_demo_build_brief.md",
        ],
    ),
    ("common", ["common/weight_layout.md", "common/fixtures.py"]),
    (
        "cpu",
        [
            "cpu/requirements.txt",
            "cpu/Makefile",
            "cpu/README.md",
            "cpu/gpt2.py",
            "cpu/sampling.py",
            "cpu/metrics.py",
            "cpu/run.py",
            "cpu/test_sampling_metrics.py",
            "cpu/test_hf_parity.py",
        ],
    ),
    (
        "tools",
        [
            "tools/verify_hf.py",
            "tools/export_reference.py",
            "tools/export_gpt2_weights.py",
            "tools/export_kernel_tests.py",
            "tools/verify_gemm_cpp.py",
            "tools/bench_plots.py",
            "tools/report_bench.py",
            "tools/summarize_bench.py",
            "tools/render_demo.py",
            "tools/render_dashboard.py",
        ],
    ),
    (
        "scripts",
        [
            "scripts/setup_env.sh",
            "scripts/wipe_local_env.sh",
            "scripts/demo_batch.sh",
            "scripts/setup_modal.sh",
            "scripts/cuda_oxide_verify.sh",
            "scripts/cuda_oxide_host_setup.sh",
            "scripts/runpod_provision_gpu.sh",
            "scripts/runpod_remote_bootstrap.sh",
        ],
    ),
    (
        "docs",
        [
            "docs/3-day-plan.md",
            "docs/gpu-setup.md",
            "docs/cuda-oxide-setup.md",
            "docs/demo-narrative.md",
            "docs/cpu-inference-walkthrough.md",
            "docs/transformer-deck-gamma.md",
        ],
    ),
    (
        "bench",
        [
            "bench/runs/01_greedy.json",
            "bench/runs/02_greedy_timing.json",
            "bench/runs/03_sample_topp.json",
            "bench/runs/04_sample_rep_penalty.json",
            "bench/runs/report.txt",
            "bench/runs/demo.html",
            "bench/runs/dashboard.html",
            "bench/runs/assets/latency_bars.png",
            "bench/runs/assets/prefill_vs_decode.png",
            "bench/runs/assets/decode_curves.png",
            "bench/runs/assets/throughput_memory.png",
        ],
    ),
    (
        "cuda-cpp",
        [
            "cuda-cpp/README.md",
            "cuda-cpp/CMakeLists.txt",
            "cuda-cpp/include/gpt2_cuda/kernels.cuh",
            "cuda-cpp/include/gpt2_cuda/weights.hpp",
            "cuda-cpp/cuda/vecadd.cu",
            "cuda-cpp/cuda/gemm_naive.cu",
            "cuda-cpp/src/main.cpp",
            "cuda-cpp/src/weights.cpp",
        ],
    ),
    (
        "cuda-oxide",
        [
            "cuda-oxide/README.md",
            "cuda-oxide/Cargo.toml",
            "cuda-oxide/Cargo.lock",
            "cuda-oxide/src/main.rs",
            "cuda-oxide/src/host.rs",
            "cuda-oxide/src/kernels/mod.rs",
        ],
    ),
    (
        "modal",
        [
            "modal/requirements.txt",
            "modal/README.md",
            "modal/get_started.py",
            "modal/gpu_smoke.py",
            "modal/gpu_cpp_vecadd.py",
        ],
    ),
    (
        "devcontainer",
        [".devcontainer/Dockerfile", ".devcontainer/devcontainer.json"],
    ),
    (
        "assets",
        [
            "assets/README.md",
            "assets/softmax_example.png",
            "assets/gelu_example.png",
            "assets/gradient_descent_example.png",
            "assets/heads_split_example.png",
            "assets/attention_mlp_token_journey.png",
            "assets/ChatGPT Image Sep 26, 2026, 01_30_35 AM-1.png",
            "assets/ChatGPT Image Sep 26, 2026, 01_30_37 AM-2.png",
            "assets/ChatGPT Image Sep 26, 2026, 01_30_39 AM-3.png",
            "assets/ChatGPT Image Sep 26, 2026, 01_30_40 AM-4.png",
        ],
    ),
    (
        "deck",
        [
            "transformer-deck-compact (1).md",
            "What-is-a-machine-learning-model.pdf",
            "What-is-a-machine-learning-model.pptx",
            "What-is-a-machine-learning-model_with_cover.pptx",
            "IMG_5356.jpg",
        ],
    ),
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def list_trackable() -> set[str]:
    out = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    )
    return {line.strip() for line in out.splitlines() if line.strip()}


def is_binary(path: Path) -> bool:
    return path.suffix.lower() in BINARY_SUFFIXES


def line_count(path: Path) -> int:
    if is_binary(path):
        return 1
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except UnicodeDecodeError:
        return 1


def planned_commits_for_file(rel: str, chunk: int) -> int:
    path = ROOT / rel
    if is_binary(path):
        return 1
    n = max(1, line_count(path))
    if n <= 22:
        return 1
    return max(2, (n + chunk - 1) // chunk)


def pick_chunk_size(files: list[str], target: int) -> int:
    best_chunk = MIN_CHUNK
    best_score = float("inf")
    for chunk in range(MIN_CHUNK, MAX_CHUNK + 1):
        total = sum(planned_commits_for_file(f, chunk) for f in files)
        score = abs(total - target)
        if score < best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk


def commit_paths(message: str, rel_paths: list[str]) -> None:
    run(["git", "add", "--", *rel_paths])
    run(["git", "commit", "-m", message])


def commit_text_incremental(rel: str, prefix: str, chunk: int) -> int:
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    if not lines:
        commit_paths(f"{prefix}: add {rel}", [rel])
        return 1

    if len(lines) <= 22:
        commit_paths(f"{prefix}: add {rel}", [rel])
        return 1

    path.write_text("", encoding="utf-8")
    commits = 0
    total_parts = (len(lines) + chunk - 1) // chunk
    for i in range(0, len(lines), chunk):
        part = i // chunk + 1
        with path.open("a", encoding="utf-8") as fh:
            fh.writelines(lines[i : i + chunk])
        commit_paths(f"{prefix}: {Path(rel).name} ({part}/{total_parts})", [rel])
        commits += 1
    return commits


def main() -> int:
    trackable = list_trackable() - SKIP_PATHS
    ordered: list[str] = []
    for _phase, paths in PHASES:
        for p in paths:
            if p not in trackable:
                print(f"skip missing: {p}", file=sys.stderr)
                continue
            ordered.append(p)
    extras = sorted(trackable - set(ordered))
    if extras:
        for p in extras:
            ordered.append(p)

    chunk = pick_chunk_size(ordered, TARGET_COMMITS)
    planned = sum(planned_commits_for_file(f, chunk) for f in ordered)
    print(f"chunk_lines={chunk} planned_commits≈{planned} files={len(ordered)}")

    if subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True).returncode == 0:
        print("Repo already has commits; aborting.", file=sys.stderr)
        return 1

    total = 0
    for phase, paths in PHASES:
        phase_count = 0
        for rel in paths:
            if rel not in trackable:
                continue
            path = ROOT / rel
            if is_binary(path):
                commit_paths(f"{phase}: add {Path(rel).name}", [rel])
                phase_count += 1
            else:
                phase_count += commit_text_incremental(rel, phase, chunk)
        total += phase_count
        print(f"  {phase}: {phase_count} commits")

    for rel in extras:
        path = ROOT / rel
        if is_binary(path):
            commit_paths(f"extra: add {Path(rel).name}", [rel])
            total += 1
        else:
            total += commit_text_incremental(rel, "extra", chunk)

    if (ROOT / "scripts/build_granular_history.py").exists():
        commit_paths("chore: add granular history builder script", ["scripts/build_granular_history.py"])
        total += 1

    print(f"done: {total} commits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
