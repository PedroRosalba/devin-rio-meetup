# From Transformer Math to GPU Execution — Compact Deck Script

> **Superseded for Gamma:** use **[`docs/transformer-deck-gamma.md`](docs/transformer-deck-gamma.md)** (final, 11 slides + KV + CPU live demo notes).

**Running example:**  
> “The strangest thing I found inside my refrigerator was”

**Goal:** explain enough Transformer math to understand the code, then show how the same math maps to CPU, CUDA C++, and CUDA-Oxide Rust.

**Slide rule:** keep the **slide visual + equations + a few words**. Put the explanation below under **Speaker notes**.

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

**Image:** Transformer architecture figure + gradient-descent graphic.

### Speaker notes

A model is a parameterized function. The architecture tells us what computation to perform; the parameters are the numbers learned during training.

Training starts from parameters that are not yet useful. We predict the next token, measure how wrong the prediction is, compute gradients, and update the parameters. Repeat over a huge corpus.

For today's demo, none of that happens. The weights are already trained. We only do inference: one forward pass with frozen parameters.

Don't explain backpropagation mechanics here.

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

### Scale reference — orders of magnitude

```text
PARAMETERS
GPT-2         124M
Llama 3.1     405B
DeepSeek-V3   671B total / 37B active

CONTEXT
GPT-2         1K
DeepSeek-V3   128K
Gemini 2.5 Pro 1.05M
```

**Scale reference, not a quality ranking.**

### Speaker notes

The point of GPT-2 is not that it is representative of today's biggest models. It is small enough that we can expose the whole computation.

For scale: 124M versus hundreds of billions of parameters. DeepSeek-V3 is an MoE model, so 671B total parameters does not mean all 671B are active for every token.

Modern context windows can also be orders of magnitude larger than GPT-2's 1K-token context.

This is why GPT-2 is useful for an educational GPU implementation: we can actually see the machinery.

**Image:** optional small model-scale visual / parameter-count bars.

