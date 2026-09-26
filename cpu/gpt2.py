"""
Explicit GPT-2 (124M) forward pass — float32, equation-aligned.
Weights layout matches Hugging Face `GPT2LMHeadModel`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from safetensors.numpy import load_file as load_safetensors
except ImportError:
    load_safetensors = None


def gelu_new(x: np.ndarray) -> np.ndarray:
    # GPT-2 uses OpenAI's "NewGELU" (tanh approximation).
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * np.power(x, 3))))


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    # x: [..., hidden]
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return x_hat * weight + bias


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    # GPT-2 Conv1D weights are [in, out]; same as x @ W in the paper.
    y = x @ w
    if b is not None:
        y = y + b
    return y


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


@dataclass
class GPT2Config:
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    n_ctx: int = 1024
    n_inner: int = 3072
    vocab_size: int = 50257
    layer_norm_epsilon: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_json(cls, path: Path) -> GPT2Config:
        data = json.loads(path.read_text())
        n_embd = data["n_embd"]
        return cls(
            n_embd=n_embd,
            n_head=data["n_head"],
            n_layer=data["n_layer"],
            n_ctx=data["n_positions"],
            n_inner=data.get("n_inner", 4 * n_embd),
            vocab_size=data["vocab_size"],
            layer_norm_epsilon=data.get("layer_norm_epsilon", 1e-5),
        )


@dataclass
class GPT2Weights:
    wte: np.ndarray  # [vocab, n_embd]
    wpe: np.ndarray  # [n_ctx, n_embd]
    ln_f_w: np.ndarray
    ln_f_b: np.ndarray
    # per layer
    ln_1_w: list[np.ndarray]
    ln_1_b: list[np.ndarray]
    c_attn_w: list[np.ndarray]  # [n_embd, 3*n_embd]
    c_attn_b: list[np.ndarray]
    c_proj_w: list[np.ndarray]
    c_proj_b: list[np.ndarray]
    ln_2_w: list[np.ndarray]
    ln_2_b: list[np.ndarray]
    c_fc_w: list[np.ndarray]
    c_fc_b: list[np.ndarray]
    c_proj_mlp_w: list[np.ndarray]
    c_proj_mlp_b: list[np.ndarray]

