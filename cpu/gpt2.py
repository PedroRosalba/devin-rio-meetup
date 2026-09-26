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


def _load_state_dict(model_dir: Path) -> dict[str, np.ndarray]:
    st = model_dir / "model.safetensors"
    if st.is_file() and load_safetensors is not None:
        return {k: v.astype(np.float32) for k, v in load_safetensors(str(st)).items()}
    import torch

    bin_path = model_dir / "pytorch_model.bin"
    if not bin_path.is_file():
        raise FileNotFoundError(f"No weights in {model_dir}")
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    return {k: v.numpy().astype(np.float32) for k, v in state.items()}


def _key(sd: dict[str, np.ndarray], *candidates: str) -> np.ndarray:
    for k in candidates:
        if k in sd:
            return sd[k]
    raise KeyError(f"None of {candidates} in state dict (sample keys: {list(sd)[:5]})")


def _assert_shape(name: str, arr: np.ndarray, expected: tuple[int, ...]) -> None:
    if arr.shape != expected:
        raise ValueError(f"{name}: expected shape {expected}, got {arr.shape}")


def assert_gpt2_weight_layout(cfg: GPT2Config, w: GPT2Weights) -> None:
    """Validate layouts against HF GPT-2 124M Conv1D conventions."""
    c = cfg.n_embd
    h, d, n_inner, v = cfg.n_head, cfg.head_dim, cfg.n_inner, cfg.vocab_size
    if c != h * d:
        raise ValueError(f"n_embd {c} != n_head*head_dim ({h}*{d})")

    _assert_shape("wte", w.wte, (v, c))
    _assert_shape("wpe", w.wpe, (cfg.n_ctx, c))
    _assert_shape("ln_f.weight", w.ln_f_w, (c,))
    _assert_shape("ln_f.bias", w.ln_f_b, (c,))

    for i in range(cfg.n_layer):
        p = f"layer[{i}]"
        _assert_shape(f"{p}.ln_1.weight", w.ln_1_w[i], (c,))
        _assert_shape(f"{p}.ln_1.bias", w.ln_1_b[i], (c,))
        # Conv1D [in, out]
        _assert_shape(f"{p}.attn.c_attn.weight", w.c_attn_w[i], (c, 3 * c))
        _assert_shape(f"{p}.attn.c_attn.bias", w.c_attn_b[i], (3 * c,))
        _assert_shape(f"{p}.attn.c_proj.weight", w.c_proj_w[i], (c, c))
        _assert_shape(f"{p}.attn.c_proj.bias", w.c_proj_b[i], (c,))
        _assert_shape(f"{p}.ln_2.weight", w.ln_2_w[i], (c,))
        _assert_shape(f"{p}.ln_2.bias", w.ln_2_b[i], (c,))
        _assert_shape(f"{p}.mlp.c_fc.weight", w.c_fc_w[i], (c, n_inner))
        _assert_shape(f"{p}.mlp.c_fc.bias", w.c_fc_b[i], (n_inner,))
        _assert_shape(f"{p}.mlp.c_proj.weight", w.c_proj_mlp_w[i], (n_inner, c))
        _assert_shape(f"{p}.mlp.c_proj.bias", w.c_proj_mlp_b[i], (c,))


def load_gpt2_config(model_dir: Path) -> GPT2Config:
    return GPT2Config.from_json(model_dir / "config.json")


def load_gpt2_weights(model_dir: Path, cfg: GPT2Config, *, check_layout: bool = True) -> GPT2Weights:
    sd = _load_state_dict(model_dir)
    n = cfg.n_layer

    def blk(layer: int, name: str) -> np.ndarray:
        return _key(sd, f"h.{layer}.{name}", f"transformer.h.{layer}.{name}")

    w = GPT2Weights(
        wte=_key(sd, "wte.weight", "transformer.wte.weight"),
        wpe=_key(sd, "wpe.weight", "transformer.wpe.weight"),
        ln_f_w=_key(sd, "ln_f.weight", "transformer.ln_f.weight"),
        ln_f_b=_key(sd, "ln_f.bias", "transformer.ln_f.bias"),
        ln_1_w=[blk(i, "ln_1.weight") for i in range(n)],
        ln_1_b=[blk(i, "ln_1.bias") for i in range(n)],
        c_attn_w=[blk(i, "attn.c_attn.weight") for i in range(n)],
        c_attn_b=[blk(i, "attn.c_attn.bias") for i in range(n)],
        c_proj_w=[blk(i, "attn.c_proj.weight") for i in range(n)],
        c_proj_b=[blk(i, "attn.c_proj.bias") for i in range(n)],
        ln_2_w=[blk(i, "ln_2.weight") for i in range(n)],
        ln_2_b=[blk(i, "ln_2.bias") for i in range(n)],
        c_fc_w=[blk(i, "mlp.c_fc.weight") for i in range(n)],
        c_fc_b=[blk(i, "mlp.c_fc.bias") for i in range(n)],
        c_proj_mlp_w=[blk(i, "mlp.c_proj.weight") for i in range(n)],
        c_proj_mlp_b=[blk(i, "mlp.c_proj.bias") for i in range(n)],
    )
    if check_layout:
        assert_gpt2_weight_layout(cfg, w)
    return w


def load_gpt2(model_dir: Path, *, check_layout: bool = True) -> tuple[GPT2Config, GPT2Weights]:
    cfg = load_gpt2_config(model_dir)
    w = load_gpt2_weights(model_dir, cfg, check_layout=check_layout)
    return cfg, w


def causal_self_attention(
    x: np.ndarray,
    c_attn_w: np.ndarray,
    c_attn_b: np.ndarray,
    c_proj_w: np.ndarray,
    c_proj_b: np.ndarray,
    n_head: int,
) -> np.ndarray:
    """
    x: [T, C]
    Multi-head causal self-attention (paper-style: softmax(QK^T / sqrt(d)) V).
    """
    t, c = x.shape
    head_dim = c // n_head

    qkv = linear(x, c_attn_w, c_attn_b)  # [T, 3C]
    q, k, v = np.split(qkv, 3, axis=-1)

    def split_heads(arr: np.ndarray) -> np.ndarray:
        # [T, C] -> [n_head, T, head_dim]
        return arr.reshape(t, n_head, head_dim).transpose(1, 0, 2)

    qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
    scale = 1.0 / np.sqrt(head_dim)
    # scores [n_head, T, T]
    scores = (qh @ kh.transpose(0, 2, 1)) * scale
    mask = np.triu(np.ones((t, t), dtype=bool), k=1)
    scores = np.where(mask, -1e4, scores)
    attn = softmax(scores, axis=-1)
    out = attn @ vh  # [n_head, T, head_dim]
    out = out.transpose(1, 0, 2).reshape(t, c)
    return linear(out, c_proj_w, c_proj_b)


def transformer_block(
    x: np.ndarray,
    layer: int,
    cfg: GPT2Config,
    w: GPT2Weights,
) -> np.ndarray:
    # Pre-norm attention + residual
    h = layer_norm(x, w.ln_1_w[layer], w.ln_1_b[layer], cfg.layer_norm_epsilon)
    h = causal_self_attention(
        h,
        w.c_attn_w[layer],
        w.c_attn_b[layer],
        w.c_proj_w[layer],
        w.c_proj_b[layer],
        cfg.n_head,
    )
    x = x + h

    # Pre-norm MLP + residual
    h = layer_norm(x, w.ln_2_w[layer], w.ln_2_b[layer], cfg.layer_norm_epsilon)
    h = linear(h, w.c_fc_w[layer], w.c_fc_b[layer])
    h = gelu_new(h)
    h = linear(h, w.c_proj_mlp_w[layer], w.c_proj_mlp_b[layer])
    return x + h


def forward_hidden_states(
    token_ids: np.ndarray, cfg: GPT2Config, w: GPT2Weights
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Returns (logits [T,V], hidden_states list length n_layer+1, incl. embed output)."""
    t = token_ids.shape[0]
    if t > cfg.n_ctx:
        raise ValueError(f"sequence length {t} > n_ctx {cfg.n_ctx}")

    pos = np.arange(t, dtype=np.int64)
    x = w.wte[token_ids] + w.wpe[pos]
    hidden_states: list[np.ndarray] = [x.copy()]

    for i in range(cfg.n_layer):
        x = transformer_block(x, i, cfg, w)
        hidden_states.append(x.copy())

    x = layer_norm(x, w.ln_f_w, w.ln_f_b, cfg.layer_norm_epsilon)
    logits = x @ w.wte.T
    return logits, hidden_states


def forward(token_ids: np.ndarray, cfg: GPT2Config, w: GPT2Weights) -> np.ndarray:
    """token_ids: [T] int -> logits [T, vocab] (LM head tied to wte)."""
    logits, _ = forward_hidden_states(token_ids, cfg, w)
    return logits


def greedy_generate(
    token_ids: list[int],
    cfg: GPT2Config,
    w: GPT2Weights,
    max_new_tokens: int,
) -> list[int]:
    ids = list(token_ids)
    for _ in range(max_new_tokens):
        arr = np.array(ids, dtype=np.int64)
        logits = forward(arr, cfg, w)
