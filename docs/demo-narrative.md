# Demo narrative (live / meetup)

Tell the story in **this order**. Each step answers one question the audience should have.

**Deep dive (MD, not slides):** [`cpu-inference-walkthrough.md`](cpu-inference-walkthrough.md) — full CLI → Python call stack → equations.

## Act 0 — Trust (30 s)

```bash
make verify
```

**Say:** “We don’t trust pretty text. NumPy forward matches Hugging Face logits and greedy tokens on fixed prompts.”

## Act 1 — What does GPT-2 actually do? (1 min)

```bash
python cpu/run.py --prompt "The strangest thing I found inside my refrigerator was" --steps 0
```

(use `--steps 0` only if we add it — today use `--steps 1` or skip decode)

Or full batch step 2:

**Say:** “Base model = next-token distribution. Top-k table is the real output; completion is just argmax repeated.”

## Act 2 — Greedy is supposed to look dumb (1 min)

```bash
python cpu/run.py --prompt "..." --steps 10 --temperature 0
```

**Say:** “Not broken — greedy + 2019 base LM loops. That’s why ChatGPT added instruction tuning and better decoders.”

## Act 3 — Systems hook: why decode hurts (2 min)

```bash
python cpu/run.py --prompt "..." --steps 12 --temperature 0 --verbose-timing
```

**Say:** “Each new token re-runs attention over the **whole** context. Watch ms/token drift up → motivates KV cache, then GPU kernels.”

Map to equation slide: `cpu/gpt2.py` → `causal_self_attention` → O(T²) per layer per step.

## Act 4 — Same weights, better demo (1 min)

```bash
python cpu/run.py --prompt "..." --steps 16 --temperature 0.85 --top-p 0.92 --top-k 50 --seed 42
```

**Say:** “Sampling doesn’t change the transformer; it changes how we pick from logits.”

## Act 5 — Baseline for the race (30 s)

```bash
make demo-batch
```

**Say:** “JSON metrics are the schema CUDA C++ and CUDA-Oxide will emit later — same prompt, compare `prefill`, `decode.mean_ms`, `peak_rss`.”

---

## One-liner for the whole room

```bash
make demo-batch
```

Runs: verify → greedy → verbose timing → sampling → optional rep penalty → summary table.

## What not to live-demo

- Long `--steps 50` greedy (boring + slow).
- Random seed without `--seed` (can’t reproduce on stage).
- Claiming GPT-2 “reasons” (use top-k + timing instead).
