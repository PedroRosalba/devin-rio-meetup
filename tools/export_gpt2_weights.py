#!/usr/bin/env python3
"""Export GPT-2 float32 weights to a single .bin + manifest for C++/Rust GPU loaders."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cpu"))

from gpt2 import GPT2Config, GPT2Weights, load_gpt2  # noqa: E402

MAGIC = b"GPT2WGT1"


def tensor_entries(cfg: GPT2Config, w: GPT2Weights) -> list[tuple[str, np.ndarray]]:
    items: list[tuple[str, np.ndarray]] = [
        ("wte", w.wte),
        ("wpe", w.wpe),
        ("ln_f.weight", w.ln_f_w),
        ("ln_f.bias", w.ln_f_b),
    ]
    for i in range(cfg.n_layer):
        p = f"h.{i}"
        items.extend(
            [
                (f"{p}.ln_1.weight", w.ln_1_w[i]),
                (f"{p}.ln_1.bias", w.ln_1_b[i]),
                (f"{p}.attn.c_attn.weight", w.c_attn_w[i]),
                (f"{p}.attn.c_attn.bias", w.c_attn_b[i]),
                (f"{p}.attn.c_proj.weight", w.c_proj_w[i]),
                (f"{p}.attn.c_proj.bias", w.c_proj_b[i]),
                (f"{p}.ln_2.weight", w.ln_2_w[i]),
                (f"{p}.ln_2.bias", w.ln_2_b[i]),
                (f"{p}.mlp.c_fc.weight", w.c_fc_w[i]),
                (f"{p}.mlp.c_fc.bias", w.c_fc_b[i]),
                (f"{p}.mlp.c_proj.weight", w.c_proj_mlp_w[i]),
                (f"{p}.mlp.c_proj.bias", w.c_proj_mlp_b[i]),
            ]
        )
    return items


def export(model_dir: Path, out_dir: Path) -> None:
    cfg, w = load_gpt2(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "gpt2_weights_fp32.bin"
    manifest_path = out_dir / "gpt2_manifest.json"

    entries = tensor_entries(cfg, w)
    offset = 0
    tensors: dict[str, dict] = {}
    with bin_path.open("wb") as f:
        f.write(MAGIC)
        for name, arr in entries:
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            raw = arr.tobytes(order="C")
            f.write(raw)
            tensors[name] = {
                "shape": list(arr.shape),
                "offset": offset,
                "length_bytes": len(raw),
                "dtype": "float32",
            }
            offset += len(raw)

    tensor_list = [
        {"name": name, **tensors[name]} for name, _ in entries
    ]
    manifest = {
        "format_version": 1,
        "magic": MAGIC.decode("ascii"),
        "header_bytes": len(MAGIC),
        "weights_file": "gpt2_weights_fp32.bin",
        "model": {
            "name": model_dir.name,
            "vocab_size": cfg.vocab_size,
            "n_embd": cfg.n_embd,
            "n_head": cfg.n_head,
            "n_layer": cfg.n_layer,
            "n_ctx": cfg.n_ctx,
            "n_inner": cfg.n_inner,
            "layer_norm_epsilon": cfg.layer_norm_epsilon,
        },
        "layout_note": "Conv1D weights stored [in, out] row-major C contiguous",
        "total_bytes": offset,
        "tensor_list": tensor_list,
        "tensors": tensors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {bin_path} ({offset / 1e6:.1f} MB payload + header)")
    print(f"Wrote {manifest_path} ({len(tensors)} tensors)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=ROOT / "models" / "gpt2")
    p.add_argument("--out-dir", type=Path, default=ROOT / "common" / "weights")
    args = p.parse_args()
    export(args.model_dir, args.out_dir)


if __name__ == "__main__":
    main()
