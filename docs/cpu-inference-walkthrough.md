# CPU inference walkthrough (`cpu/run.py`)

Canonical **call stack** from one CLI command down to the equations in `cpu/gpt2.py`. Use this when reading code or mapping slides to function names.

For **live demo order and what to say**, see [`demo-narrative.md`](demo-narrative.md).

---

## Example command

From repo root (defaults: greedy, 20 new tokens, top-10 preview after prefill):

```bash
python cpu/run.py --prompt "The strangest thing I found inside my refrigerator was"
```

Python adds `cpu/` to `sys.path`, so `from gpt2 import ...` resolves to `cpu/gpt2.py`.

---

## Phase 0 — Interpreter entry

```text
python cpu/run.py
  → load cpu/run.py (imports gpt2, metrics, sampling, GPT2Tokenizer)
  → if __name__ == "__main__": main()
```

No model math yet — only module imports.

---

## Phase 1 — `main()` setup

| Order | Call | Role |
|------:|------|------|
| 1 | `parse_args()` | CLI → `model_dir`, `prompt`, `steps`, sampling flags |
| 2 | `SamplingConfig(...)` + `validate()` | Default `temperature=0` → greedy |
| 3 | `load_gpt2_config(model_dir)` | `GPT2Config.from_json` → read `models/gpt2/config.json` |
| 4 | `load_gpt2_weights(model_dir, cfg)` | Load tensors + `assert_gpt2_weight_layout` |
| 5 | `GPT2Tokenizer.from_pretrained(model_dir)` | Vocab / BPE |
| 6 | `tok.encode(prompt, add_special_tokens=False)` | Text → `input_ids: list[int]` |

**Nested: `load_gpt2_weights`**

```text
load_gpt2_weights
  └─ _load_state_dict
       └─ model.safetensors (or pytorch_model.bin) → float32 arrays
  └─ assemble GPT2Weights (wte, wpe, 12 layers)
  └─ assert_gpt2_weight_layout   # shapes only
```

---

## Phase 2 — Prefill (first full forward)

```text
arr = np.array(input_ids)
logits = forward(arr, cfg, weights)
```

**Stack:**

```text
forward
  └─ forward_hidden_states
       ├─ embeddings
       ├─ transformer_block × cfg.n_layer (12)
       └─ final layer norm + LM head
```

### Embeddings

```python
x = w.wte[token_ids] + w.wpe[pos]   # pos = 0..T-1
```

Per position \(i\): \(x_i = W_{\text{te}}[t_i] + W_{\text{pe}}[i]\). Shape `[T, 768]`.

### One `transformer_block` (pre-norm GPT-2)

```text
transformer_block
  ├─ layer_norm(x, ln_1)           # μ, σ² on last dim; scale/shift
  ├─ causal_self_attention(...)
  ├─ x = x + h                     # residual
  ├─ layer_norm(x, ln_2)
  ├─ linear → gelu_new → linear    # MLP
  └─ return x + h
```

**Nested: `causal_self_attention`**

```text
causal_self_attention
