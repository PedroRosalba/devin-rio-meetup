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

**Image:** token journey graphic.

### Speaker notes

The tokenizer defines a fixed mapping from text to token IDs. GPT-2 uses byte-level BPE; the model does not learn which ID means “The.”

What training learns is the vector stored in each embedding row.

For a sequence of $T$ tokens, we get a $T\times768$ matrix. That matrix is the input to the first Transformer block.

---

## Slide 4 — What does a Transformer block actually do?

### On slide

### 1. Attention — **mix information across tokens**

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_h}}+M
\right)V
$$

```text
token i
   │
   ├── looks at other tokens
   ↓
contextualized token i
```

### 2. MLP — **transform each token independently**

```text
token vector
    ↓
 expand 768 → 3072
    ↓
   GELU
    ↓
 compress 3072 → 768
```

### Speaker notes

This is the intuition I want people to leave with:

**Attention moves information between positions.**  
A token can look at earlier tokens and decide which information is useful.

**The MLP processes the resulting representation inside each position.**  
It does not mix tokens together.

A useful mental model is:

> Attention = communication between tokens.  
> MLP = computation performed on each token.

The 12 heads give the model multiple learned attention patterns in parallel. Each head operates on 64 dimensions.

The causal mask $M$ prevents a token from looking into the future.

**Image:** Figure 2 from *Attention Is All You Need* + GELU graphic.

---

## Slide 5 — From Transformer math to computer operations

### On slide

| Transformer | Computer |
|---|---|
| Q/K/V + projections | Matrix multiply |
| $QK^T$, attention × V | Matrix multiply |
| Softmax | reductions + exp |
| LayerNorm | reductions + elementwise |
| Residual | elementwise add |
| Embedding | indexed memory read |

$$
\boxed{
\text{Transformer}
\approx
\text{GEMMs}
+
\text{reductions}
+
\text{elementwise math}
+
\text{memory movement}
}
$$

$$
C_{ij}=\sum_k A_{ik}B_{kj}
$$

**Image:** small 2×2 matrix-multiply graphic.

### Speaker notes

This is the bridge to the hardware.

A Transformer sounds complicated because we describe it with high-level mathematical concepts. But eventually the machine sees matrix multiplications, reductions, elementwise functions, and memory accesses.

GEMM means general matrix multiplication. Every output element is a multiply-and-sum.

That matters because those output elements can be computed in parallel.

---

## Slide 6 — Our NumPy reference implementation

### On slide

```python
x = w.wte[token_ids] + w.wpe[pos]

qkv = linear(x, c_attn_w, c_attn_b)
q, k, v = np.split(qkv, 3, axis=-1)

scores = (qh @ kh.transpose(0, 2, 1)) * scale
attn   = softmax(scores, axis=-1)
out    = attn @ vh

h = layer_norm(x);  h = causal_self_attention(h);  x = x + h
h = layer_norm(x);  h = gelu_new(linear(h, ...));  x = x + h   # MLP

x = final_layer_norm(x)
logits = x @ w.wte.T
```

**Equation → NumPy → CUDA**

### Speaker notes

This is deliberately not a fast runtime. It is our correctness reference.

The important thing is that the equations are visible in the code.

For example:

$$
QK^T
$$

literally becomes:

```python
qh @ kh.transpose(...)
```

and the final language-model projection is:

```python
logits = x @ w.wte.T
```

This gives us something to compare the GPU implementations against.

---

## Slide 7 — What does the model actually output?

### On slide

```text
hidden state
     ↓
linear projection
     ↓
50,257 logits
     ↓
softmax
     ↓
50,257 probabilities
     ↓
decoding
     ↓
next token
```

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

**Greedy:** choose $\arg\max_i z_i$  
**Top-k / top-p:** sample from a restricted distribution

$$
\boxed{\text{model inference}\neq\text{decoding strategy}}
$$

**Image:** softmax before/after graphic.

### Speaker notes

The model itself does not output a word. It outputs 50,257 scores.

Softmax converts those scores into probabilities. Then the decoding strategy decides what token to select.

This is also why our CPU demo can produce repetitive text with greedy decoding: that is a property of the model plus decoding strategy, not because the CPU changed the model.

---

## Slide 8 — Why a GPU?

### On slide

$$
C_{ij}=\sum_k A_{ik}B_{kj}
$$

**Each output element can be computed independently.**

```text
CPU                         GPU
few powerful cores          many parallel execution units
      ↓                              ↓
  limited parallelism          massive data parallelism
```

**Question:** can we execute thousands of these small pieces simultaneously?

### Speaker notes

The math does not change when we move from CPU to GPU.

What changes is how much parallel hardware we can use.

This is why matrix-heavy workloads map naturally to GPUs.

Do not use theoretical TFLOPS to claim a specific speedup. We will measure our actual implementation.

---

## Slide 9 — CUDA execution model

### On slide

```text
GPU kernel
    ↓
  GRID
    ↓
  BLOCKS
    ↓
  WARPS (32 threads)
    ↓
  THREADS
```

```text
output matrix
      ↓
   tiles
      ↓
one block → one tile
      ↓
