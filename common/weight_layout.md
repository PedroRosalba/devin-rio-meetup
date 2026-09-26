# GPT-2 weight layout (Hugging Face `openai-community/gpt2`)

This repo treats **GPT-2 `Conv1D` matrices as `[in_features, out_features]`**.  
That matches Hugging Face’s `Conv1D` storage and the forward `y = x @ W + b` (see [modeling_gpt2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)).

**Do not** assume generic `nn.Linear` layout `[out_features, in_features]` (`y = x @ W.T`).

## Config (124M)

| Field | Value |
|-------|--------|
| `vocab_size` | 50257 |
| `n_embd` | 768 |
| `n_head` | 12 |
| `n_layer` | 12 |
| `n_positions` / `n_ctx` | 1024 |
| MLP inner (`n_inner`) | **3072** (= 4 × `n_embd`, HF default when omitted) |
| `head_dim` | 64 (= `n_embd` / `n_head`) |

## Checkpoint key prefixes

Safetensors in this checkpoint use **short keys** (no `transformer.` prefix):

- `wte.weight`, `wpe.weight`, `ln_f.weight`, `ln_f.bias`
- `h.{L}.ln_1.weight`, `h.{L}.attn.c_attn.weight`, …

`pytorch_model.bin` exports may use the `transformer.` prefix; `load_gpt2()` accepts both.

## Tensor shapes (per layer `L`, float32)

| Parameter | Shape | Conv1D `[in, out]` |
|-----------|--------|---------------------|
| `wte.weight` | `[50257, 768]` | embedding table (not Conv1D) |
| `wpe.weight` | `[1024, 768]` | position table |
| `h.L.ln_1.weight` | `[768]` | LayerNorm γ |
| `h.L.ln_1.bias` | `[768]` | LayerNorm β |
| `h.L.attn.c_attn.weight` | `[768, 2304]` | 768 → Q∥K∥V |
| `h.L.attn.c_attn.bias` | `[2304]` | |
| `h.L.attn.c_proj.weight` | `[768, 768]` | attn out proj |
| `h.L.attn.c_proj.bias` | `[768]` | |
| `h.L.ln_2.weight` | `[768]` | |
| `h.L.ln_2.bias` | `[768]` | |
| `h.L.mlp.c_fc.weight` | `[768, 3072]` | MLP up |
| `h.L.mlp.c_fc.bias` | `[3072]` | |
| `h.L.mlp.c_proj.weight` | `[3072, 768]` | MLP down |
| `h.L.mlp.c_proj.bias` | `[768]` | |
| `ln_f.weight` | `[768]` | |
| `ln_f.bias` | `[768]` | |

LM head weights are **tied** to `wte` (no separate `lm_head.weight` in the checkpoint).

## Forward matmul convention (NumPy reference)

```python
# Conv1D: x [T, in] @ W [in, out] + b [out]
y = x @ W + b

# Logits with tied embeddings: hidden [T, 768], wte [50257, 768]
logits = hidden @ wte.T   # [T, 50257]
```

Assertions run in `assert_gpt2_weight_layout()` on every `load_gpt2()`.

## Hidden states vs Hugging Face

`output_hidden_states=True` returns **13** tensors for GPT-2 small: embedding, then after blocks **0..10**, then **`ln_f`** (post block 11). It does **not** expose a separate tensor for “after block 11, pre-`ln_f`”. Our NumPy trace keeps that extra tensor for kernel debugging; `tools/verify_hf.py` compares accordingly.
