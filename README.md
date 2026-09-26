# Transformer runtime demo (GPT-2)

Same open weights, three runtimes later. **v0 (2–3 day goal):** environment ready + **CPU reference** forward pass aligned with the Transformer/GPT-2 equations. CUDA C++ and Rust/CUDA-Oxide come next on a rented Ampere box.

## Quick start (macOS or Linux)

```bash
./scripts/setup_env.sh
source .venv/bin/activate   # if created
python cpu/run.py --prompt "The strangest thing I found inside my refrigerator was"
```

## GPU path (when you rent)

1. Linux, NVIDIA Ampere+, driver **580+**, CUDA **13+** (see [CUDA-Oxide install](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html)).
2. On the host: `nvidia-smi` then follow `docs/gpu-setup.md`.
3. Implement/compare kernels against `cpu/` outputs (export with `tools/export_reference.py` when needed).

## Layout

| Path | Role |
|------|------|
| `cpu/` | **NumPy reference oracle** — explicit float32 GPT-2 forward |
| `cuda-cpp/` | CUDA C++ kernels (next) |
| `cuda-oxide/` | Rust + CUDA-Oxide (next) |
| `common/weight_layout.md` | HF checkpoint shapes + Conv1D `[in, out]` rules |
| `modal/` | Optional **Modal** Python client (remote CPU/GPU jobs) |
| `scripts/` | Model download + venv |
| `docs/` | Talk narrative, inference walkthrough, GPU setup |

Before GPU work:

```bash
python tools/verify_hf.py --strict
```

**Meetup demo (sequential narrative + JSON baselines):**

```bash
make demo-batch
```

On stage: follow [`docs/demo-narrative.md`](docs/demo-narrative.md). Prep / deep trace: [`docs/cpu-inference-walkthrough.md`](docs/cpu-inference-walkthrough.md).

**Graphs + browser dashboard (after batch):**

```bash
make demo-open    # writes bench/runs/demo.html + PNGs, opens file:// in browser
make demo-plot    # same graphs + terminal summary table
```

## Read more

| Doc | When to open it |
|-----|-----------------|
| [`docs/transformer-deck-gamma.md`](docs/transformer-deck-gamma.md) | **Gamma slide source (final)** — import this; 11 slides + speaker notes |
| [`docs/demo-narrative.md`](docs/demo-narrative.md) | **On stage** — command order + what to say (Acts 0–5) |
| [`docs/cpu-inference-walkthrough.md`](docs/cpu-inference-walkthrough.md) | **Prep / Q&A** — CLI → every Python call → equations (`cpu/run.py` → `gpt2.py`) |
| [`cpu/README.md`](cpu/README.md) | CPU oracle files, verify HF, `make demo-batch` |
| [`common/weight_layout.md`](common/weight_layout.md) | Checkpoint shapes, Conv1D `[in, out]` (GPU porting) |
| [`docs/gpu-setup.md`](docs/gpu-setup.md) | Renting a GPU box, driver/CUDA checklist |
| [`docs/3-day-plan.md`](docs/3-day-plan.md) | Milestones; slide mapping equation → function → kernel |
| [`What-is-a-machine-learning-model.pptx`](What-is-a-machine-learning-model.pptx) | Meetup deck (PowerPoint) |

Optional cloud runner (Python only; see `modal/README.md` for CUDA C++/Rust):

```bash
./scripts/setup_modal.sh
python3 -m modal setup
modal run modal/get_started.py
```

Full milestone list: `docs/3-day-plan.md`.

## Free disk space (keep the repo, drop env artifacts)

Heavy paths are gitignored: `.venv/`, `models/`. Remove them when you need space on a laptop:

```bash
./scripts/wipe_local_env.sh
```

Optional Modal CLI credentials live outside the repo: `rm -f ~/.modal.toml`.
