# From Transformer Math to GPU Execution — Gamma deck (final)

**Import this file into Gamma.** Each `## Slide N` = one slide. Copy **Speaker notes** into Gamma presenter notes (not on the visual).

**Running example:**  
> “The strangest thing I found inside my refrigerator was”

**Talk (~40 min):** slides 1–11 + **live CPU** (`python cpu/run.py`, `make verify`). **CUDA-Oxide:** show diagram on slide 10 + verbal on slide 11 — **no GPU run** on stage.

**Slide rule:** slide = visuals + equations + a few words only.

**Images:** see [`assets/README.md`](../assets/README.md) — each slide below lists exact filenames.

---

## Slide 1 — What is a machine-learning model?

### On slide

$$
y=f_\theta(x)
$$

- $x$ = input
- $f$ = architecture / computation
- $\theta$ = learned parameters
- $y$ = output

**Training:** learn $\theta$  
**Inference:** freeze $\theta$, run $f_\theta$

Small footer:

$$
L=-\log p_{\text{correct}}
\qquad
\theta\leftarrow\theta-\eta\nabla_\theta L
$$

**Images (Gamma — upload from `assets/`):**

| Placement | File |
|-----------|------|
| **Main** (right half) | `assets/ChatGPT Image Sep 26, 2026, 01_30_37 AM-2.png` — encoder/decoder Transformer (Figure 1) |
| **Inset** (corner) | `assets/gradient_descent_example.png` — gradient descent on one $\theta$ |

Say: GPT-2 is **decoder-only**; we care about the masked self-attention + FFN stack, not the full encoder path.

---

## Slide 2 — GPT-2 Small: what are we actually running?

### On slide

| | GPT-2 Small |
|---|---:|
| Parameters | **124M** |
| Transformer blocks | **12** |
| Hidden size | **768** |
| Attention heads | **12 × 64** |
| Vocabulary | **50,257 tokens** |
| Context | **1,024 tokens** |

```text
tokens → embeddings → 12 Transformer blocks → LayerNorm → logits
```

**Scale (reference only):** GPT-2 **124M** params · modern LMs **100B+** · context **1K** vs **100K+**

### Speaker notes

Small enough to expose the **whole** forward pass in code and on CPU.

124M vs hundreds of billions is an **orders-of-magnitude** reference, not a quality ranking.

**Image:** none — keep the table readable (see `assets/README.md`).

---

## Slide 3 — Text → tokens → vectors

### On slide

```text
"The refrigerator..."
        ↓
 tokenizer
        ↓
[ token IDs ]
        ↓
embedding lookup
        ↓
X ∈ ℝ^(T × 768)
```

$$
