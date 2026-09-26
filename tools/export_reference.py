#!/usr/bin/env python3
"""Optional: dump HF torch logits for one prompt (oracle). Run after setup_env."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=root / "models" / "gpt2")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--out", type=Path, default=root / "common" / "test_vectors" / "hf_logits.npy")
    args = p.parse_args()

    tok = GPT2Tokenizer.from_pretrained(str(args.model_dir))
    model = GPT2LMHeadModel.from_pretrained(str(args.model_dir))
    model.eval()

    ids = tok.encode(args.prompt, add_special_tokens=False)
    with torch.no_grad():
        out = model(torch.tensor([ids]))
        logits = out.logits[0].numpy().astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, logits)
    meta = {"prompt": args.prompt, "token_ids": ids, "shape": list(logits.shape)}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print("Wrote", args.out, meta)


if __name__ == "__main__":
    main()
