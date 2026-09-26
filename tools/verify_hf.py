#!/usr/bin/env python3
"""
Compare NumPy GPT-2 reference vs Hugging Face GPT2LMHeadModel on fixed prompts.
Exit 0 if within tolerance; use --strict for pytest/CI-style failure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cpu"))
sys.path.insert(0, str(ROOT))

from common.fixtures import HELLO_PROMPT, REFRIGERATOR_PROMPT  # noqa: E402
from gpt2 import forward, forward_hidden_states, layer_norm, load_gpt2  # noqa: E402

# HF float32 forward on CPU is the oracle for this project.
LOGITS_ATOL = 1e-4
LOGITS_RTOL = 1e-4
# Intermediates accumulate error over 12 layers; logits stay tight.
HIDDEN_ATOL = 1e-3
HIDDEN_RTOL = 1e-3


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _assert_close(name: str, ref: np.ndarray, got: np.ndarray, atol: float, rtol: float) -> None:
    if not np.allclose(ref, got, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name}: max_abs_diff={_max_abs(ref, got):.6e}, "
            f"ref shape={ref.shape}, got shape={got.shape}"
        )


@torch.no_grad()
def hf_forward(model: GPT2LMHeadModel, input_ids: list[int]) -> tuple[np.ndarray, list[np.ndarray]]:
    ids = torch.tensor([input_ids], dtype=torch.long)
    out = model(ids, output_hidden_states=True)
    logits = out.logits[0].numpy().astype(np.float32)
    # hidden_states[0] = embed; [i+1] = after block i
    hiddens = [h[0].numpy().astype(np.float32) for h in out.hidden_states]
    return logits, hiddens


def verify_prompt(
    model_dir: Path,
    prompt: str,
    *,
    strict: bool,
) -> bool:
    tok = GPT2Tokenizer.from_pretrained(str(model_dir))
    ids = tok.encode(prompt, add_special_tokens=False)
    print(f"\n=== prompt ({len(ids)} tokens) ===\n{prompt!r}\nids={ids}")

    cfg, weights = load_gpt2(model_dir)
    arr = np.array(ids, dtype=np.int64)

    np_logits, np_hiddens = forward_hidden_states(arr, cfg, weights)

    hf_model = GPT2LMHeadModel.from_pretrained(str(model_dir))
    hf_model.eval()
    hf_logits, hf_hiddens = hf_forward(hf_model, ids)

    ok = True
    try:
        _assert_close("logits", hf_logits, np_logits, LOGITS_ATOL, LOGITS_RTOL)
        print(f"logits: OK (max_abs <= ~{LOGITS_ATOL})")
    except AssertionError as e:
        ok = False
        print(f"logits: FAIL — {e}")

    # HF: hidden_states[0]=embed, [1..n_layer-1]=after blocks 0..n_layer-2,
    #      hidden_states[n_layer]=ln_f(output after all blocks).
    # NumPy: [0]=embed, [1..n_layer]=after blocks 0..n_layer-1 (incl. last block pre-ln_f).
    n_layer = cfg.n_layer
    if len(hf_hiddens) != n_layer + 1:
        ok = False
        print(f"hidden_states: FAIL — expected {n_layer + 1} HF tensors, got {len(hf_hiddens)}")
    else:
        for i in range(n_layer):
            try:
                _assert_close(f"hidden_states[{i}]", hf_hiddens[i], np_hiddens[i], HIDDEN_ATOL, HIDDEN_RTOL)
            except AssertionError as e:
                ok = False
                print(f"hidden_states[{i}]: FAIL — {e}")
        ln_f_np = layer_norm(
