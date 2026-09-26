# CUDA-Oxide / Rust (GPT-2 inference)

Parallel track to `cuda-cpp/`. Same manifest: `common/weights/gpt2_manifest.json`.

## Mac (now): host stub

```bash
python tools/export_gpt2_weights.py
cargo run --manifest ../common/weights/gpt2_manifest.json --check-weights
```

(`cargo run` from `cuda-oxide/` — no GPU.)

## Linux GPU

Follow [CUDA-Oxide install](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html), then:

```bash
cargo oxide doctor
cargo oxide run vecadd
cargo oxide run -- --manifest ../common/weights/gpt2_manifest.json
```

Implement kernels under `src/kernels/` matching `cuda-cpp/cuda/` names.
