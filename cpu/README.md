# CPU reference (Python + NumPy)

Explicit float32 GPT-2 forward pass used as the **correctness oracle** for CUDA C++ and CUDA-Oxide.

- **`gpt2.py`** — equation-aligned ops; Conv1D weights are **`[in, out]`** (see `common/weight_layout.md`).
- **`run.py`** — benchmark report, sampling flags, JSON metrics export. Call stack: [`docs/cpu-inference-walkthrough.md`](../docs/cpu-inference-walkthrough.md).
- **`sampling.py`** — temperature / top-k / top-p / repetition penalty pipeline.
- **`metrics.py`** — shared benchmark schema for future CUDA runtimes.
- Layout checks run on every `load_gpt2()`.

Verify against Hugging Face (required before GPU work):

```bash
python tools/verify_hf.py --strict
```

From **`cpu/`** or repo root:

```bash
make demo-batch
```

(`cpu/Makefile` forwards to the root `Makefile`.)

Or:

```bash
python -m unittest cpu.test_hf_parity -v
```
