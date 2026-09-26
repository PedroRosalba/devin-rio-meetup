# Slide images → Gamma placement

All paths relative to repo root. Upload these in Gamma when building each slide.

| File | Use on slide | Role |
|------|--------------|------|
| `gradient_descent_example.png` | **1** | Training footer: $\theta \leftarrow \theta - \eta \nabla L$ (infer-only today) |
| `ChatGPT Image Sep 26, 2026, 01_30_37 AM-2.png` | **1** | Classic encoder–decoder Transformer figure (*say: GPT-2 is decoder-only, same block ideas*) |
| *(none)* | **2** | Text/table only |
| *(none)* | **3** | Text pipeline only |
| `attention_mlp_token_journey.png` | **4** | **Hero:** attention vs MLP on your refrigerator prompt |
| `ChatGPT Image Sep 26, 2026, 01_30_35 AM-1.png` | **4** | Inset: Scaled dot-product + multi-head (Figure 2) |
| `gelu_example.png` | **4** | Optional small inset (also shown in hero diagram) |
| `heads_split_example.png` | **4** or **Q&A** | 12 heads × 64 → concat → $W_O$ |
| *(none)* | **5** | GEMM box + equation only |
| *(none)* | **6** | Code snippet only |
| `softmax_example.png` | **7** | Softmax: $e^z$ then normalize (vocab decoding) |
| `attention_mlp_token_journey.png` | **8** *(optional)* | Remind: attention uses K/V; we recompute each step |
| `ChatGPT Image Sep 26, 2026, 01_30_35 AM-1.png` | **8** *(optional)* | Q / K / V into attention block |
| `ChatGPT Image Sep 26, 2026, 01_30_39 AM-3.png` | **9** | CPU vs GPU: cores vs control/cache |
| `ChatGPT Image Sep 26, 2026, 01_30_40 AM-4.png` | **10** | SM / GPC / DRAM — stand-in for CUDA-Oxide SIMT art |
| *(none)* | **11** | Three-runtimes text + live terminal |

**Not in this folder (text-only slides):** token journey PNG, 2×2 matmul graphic, CUDA-Oxide SIMT screenshot.

Deck script: [`docs/transformer-deck-gamma.md`](../docs/transformer-deck-gamma.md).
