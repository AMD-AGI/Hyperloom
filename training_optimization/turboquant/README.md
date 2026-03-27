# TurboQuant — KV Cache Quantization

From-scratch implementation of Google's **TurboQuant** (ICLR 2026) for LLM KV cache compression.

**Paper:** Zandieh, Daliri, Hadian, Mirrokni — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

## Files

| File | Description |
|------|-------------|
| `turboquant_core.py` | Core quantization engine: Lloyd-Max codebook solver, random rotation, per-coordinate quantization, outlier-aware mixed-precision wrapper |
| `evaluate.py` | End-to-end evaluation: attention fidelity, needle-in-a-haystack retrieval, perplexity |

## Quick Start

```bash
# Dependencies
pip install torch transformers accelerate scipy

# Run full evaluation (takes ~3 minutes on a single GPU)
cd /shared_nfs/nehaprakriya/agentic-rc/turboquant
python evaluate.py
```

The script loads **Qwen/Qwen2.5-3B-Instruct**, calibrates outlier channels, then runs:

1. **Attention fidelity** — per-layer cosine similarity of attention scores (quantized vs full-precision)
2. **Needle-in-a-Haystack** — retrieval of a hidden sentence across context lengths 1K–4K tokens
3. **Perplexity** — language modeling quality on filler text

## How It Works

TurboQuant compresses KV cache vectors through three steps:

1. **Random rotation** (Haar-distributed orthogonal matrix via QR decomposition) — makes coordinates approximately i.i.d. Beta-distributed regardless of input
2. **Per-coordinate Lloyd-Max quantization** — optimal scalar quantizer for the Beta distribution, precomputed for each bit-width
3. **Outlier-aware mixed precision** (paper Section 4.3) — high-variance channels get more bits

The implementation also:
- Keeps first and last 2 layers at full precision (standard KV cache quantization practice)
- Calibrates outlier channels per layer using a short forward pass
- Integrates with HuggingFace `DynamicCache` for drop-in use with `model.generate()`

## Results vs Paper

### Needle-in-a-Haystack (Paper Section 4.2, Figure 4)

The paper reports a recall score of **0.997** for TurboQuant on Llama-3.1-8B-Instruct, identical to full precision. Our evaluation on Qwen2.5-3B-Instruct:

| Method | Score | Compression |
|--------|-------|-------------|
| Full Precision (16-bit) | **1.000** | 1x |
| TQ-3.5bit outlier-aware | **1.000** | ~4.5x |
| TQ-2.5bit outlier-aware | **1.000** | ~6x |
| TQ-4bit uniform | **1.000** | ~4x |

All quantized configurations achieve **identical retrieval accuracy** to full precision across 9 test cases (3 context lengths x 3 needle positions). This matches the paper's claim of "absolute quality neutrality."

### Perplexity

| Method | PPL |
|--------|-----|
| Full Precision | **1.12** |
| TQ-3.5bit outlier-aware | **1.12** |
| TQ-2.5bit outlier-aware | **1.13** |
| TQ-4bit uniform | **1.12** |

TQ-3.5bit achieves **identical perplexity** to full precision. TQ-2.5bit shows marginal degradation (+0.01). This matches the paper's claim: "quality neutrality with 3.5 bits per channel and marginal quality degradation with 2.5 bits."

### Per-Layer Attention Fidelity

| Method | Cosine Similarity | Top-1 Match |
|--------|-------------------|-------------|
| TQ-3.5bit outlier-aware | 0.9998 | 93.1% |
| TQ-2.5bit outlier-aware | 0.9994 | 84.7% |
| TQ-4bit uniform | 0.9998 | 90.3% |

### Theoretical Distortion (Paper Theorem 1)

Normalized MSE for unit vectors on the 128-dimensional sphere:

| Bits | Paper Prediction | Our Measured |
|------|-----------------|--------------|
| 3 | 0.030 | 0.034 |
| 4 | 0.009 | 0.009 |

## Configuration

The default 3.5-bit outlier-aware configuration allocates:
- **32 outlier channels** at 5 bits (identified per layer by KV variance)
- **96 regular channels** at 3 bits
- Effective: (32×5 + 96×3) / 128 = **3.5 bits/channel**

Edit `evaluate.py` `main()` to change bit allocations, number of outlier channels, or which layers to skip.
