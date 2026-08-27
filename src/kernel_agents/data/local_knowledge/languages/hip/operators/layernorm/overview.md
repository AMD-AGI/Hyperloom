---
title: layernorm — overview
kind: operator_overview
operator: layernorm
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp32]
regimes: [prefill, decode, both, training]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/normalization/norm.py
  - /sgl-workspace/aiter/aiter/ops/norm.py
  - https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
---

# layernorm  (`y = (x − μ)/√(σ² + ε) · γ + β`)

## TL;DR
LayerNorm is RMSNorm's heavier cousin: it subtracts the **mean** and adds a **bias**, so it needs a
**two-statistic reduction** (μ and σ²) per row. It dominates older/encoder models (BERT, ViT towers in
VLMs, GPT-2-style), while modern LLM decoders use rmsnorm. Same MI300X story — **bandwidth-bound** — but
the two reductions mean either a two-pass (mean, then variance) or a one-pass **Welford** scheme. On AMD,
authored Triton/HIP and aiter own it; **MIOpen has a layernorm primitive but is idle at LLM inference**.

## Math contract
For row `x[N]`: `μ = Σx/N`, `σ² = Σ(x−μ)²/N`, `y = (x−μ)·rsqrt(σ²+ε)·γ + β`. dtype: bf16/fp16 in,
**fp32 accumulate** for both μ and σ², bf16/fp16 out; `γ,β` fp32-promoted. Layout `x[M,N]`, reduce over the
contiguous last dim. Optional saved `mean,rstd` for the backward (training). Variants: `+residual add`,
`+dynamic/smooth quant` (aiter ships all).

## Shape regimes
- **prefill / encoder**: `M = tokens` (1k–64k), `N ∈ {768, 1024, 4096, 5120}`. Many rows → row-per-program.
- **decode**: `M = batch` (1..256) → CU starvation; persistent grid `min(M, num_sms)`. Same as rmsnorm.
- **N width** sets pass strategy: N ≤ block (`65536/elt`) → row-in-registers two-pass; N > block → blocked.

## Where it matters (Amdahl)
On a decoder LLM, LayerNorm is **near 0%** (RMSNorm replaced it). On **VLM vision towers** (CLIP/ViT) and
encoder models it's 2–5% and a fusion anchor (add+norm, norm+quant) just like rmsnorm. Treat the
optimization story as "rmsnorm + a mean reduction + a bias."

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢 sota (aiter's own impl, two-pass) | triton |
| aiter | 🟢 sota (CK/asm `layernorm2d_fwd` + fused) | aiter |
| hip | 🟢 sota (vLLM `layer_norm` kernel) | [hip.md](hip.md) |
| vllm_kernels | 🟢 sota (HIP + AITER wiring) | vllm_kernels |
| miopen | 🟡 competitive (primitive exists; idle for LLM) | miopen |
| flydsl | 🧪 experimental (two-statistic reduce **primitive** `make_block_reduce_add2` for μ/σ²; no standalone tuned op) | flydsl |

## Fusion neighbors
`+residual add` (`layernorm2d_fwd_with_add`), `+fp8/int8 dynamic/smooth quant`
(`layernorm2d_fwd_with_dynamicquant`/`_smoothquant`, residual+norm+quant triple) → fused_norm_quant,
cross-link quant_dequant_fp8 / quant_int8. See [fusion.md](fusion.md).

## Numerics
Two-pass vs Welford; fp32 μ/σ²; γ,β fp32-promote; bias-correction of variance. See [numerics.md](numerics.md).

## How to bench
`python3 op_tests/test_layernorm2d.py` (aiter) at `(M,N,dtype)`; fp64 oracle; median of ≥3 warm reps.

## Sources
- aiter Triton layernorm (two-pass mean/var, blocked, fused add/quant): `/sgl-workspace/aiter/aiter/ops/triton/normalization/norm.py`.
- aiter C++/CK/asm (`layernorm2d_fwd`, `_with_add`, `_with_smoothquant`): `/sgl-workspace/aiter/aiter/ops/norm.py`.
- vLLM HIP `layer_norm`: https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu.
- MIOpen norm primitive (idle for LLM): https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html.
