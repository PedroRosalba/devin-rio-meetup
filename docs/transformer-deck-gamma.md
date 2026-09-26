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
x_i = W_{te}[t_i] + W_{pe}[i]
$$

- $W_{te}$: token embeddings  
- $W_{pe}$: position embeddings  

**Image:** none — text pipeline only (no token-journey PNG in repo yet).

### Speaker notes

Byte-level BPE maps text → IDs. The model learns **embedding vectors**, not the tokenizer table.

Output is a $T \times 768$ matrix fed to block 1.

---

## Slide 4 — What does a Transformer block actually do?

### On slide

**1. Attention — mix information across tokens**

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_h}}+M
\right)V
$$

```text
token i  →  looks at earlier tokens  →  contextualized token i
```

**2. MLP — transform each token independently**

```text
768 → 3072 → GELU → 768
```

**Images (Gamma — `assets/`):**

| Placement | File |
|-----------|------|
| **Hero** (most of slide) | `assets/attention_mlp_token_journey.png` — attention vs MLP, token **“was”** + your prompt |
| **Inset** | `assets/ChatGPT Image Sep 26, 2026, 01_30_35 AM-1.png` — Figure 2: MatMul → Scale → Mask → Softmax → MatMul |
| **Optional** | `assets/gelu_example.png` or `assets/heads_split_example.png` (12 heads) — use one, or save heads for Q&A |

### Speaker notes

**Attention = communication between positions.**  
**MLP = computation inside each position** (no mixing across tokens).

$M$ = causal mask (no looking at the future).  
12 heads = 12 parallel patterns, $d_h = 64$ each.

---

## Slide 5 — From Transformer math to computer operations

### On slide

| Transformer | Computer |
|---|---|
| Q/K/V + projections | Matrix multiply (GEMM) |
| $QK^T$, attn $\times$ V | GEMM |
| Softmax | reductions + exp |
| LayerNorm | reductions + elementwise |
| Residual | elementwise add |
| Embedding | indexed read |

$$
\boxed{
\text{Transformer}
\approx
\text{GEMMs}
+
\text{reductions}
+
\text{elementwise}
+
\text{memory movement}
}
$$

$$
C_{ij}=\sum_k A_{ik}B_{kj}
$$

**Image:** none — equation + table only (no matmul PNG in repo).

### Speaker notes

High-level math → **GEMM + reductions + elementwise**.  
Many output elements of a GEMM can be computed **in parallel** — that’s the GPU hook.
