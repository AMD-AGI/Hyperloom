---
title: batched_gemm — fusion
kind: technique
operator: batched_gemm
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
  - ROCm/aiter@b467ce342:aiter/tuned_gemm.py
---

# batched_gemm — fusion

## TL;DR
The single most important "fusion" for batched GEMM is **into attention**: QKᵀ and PV batched matmuls
get fused with softmax into a fused-MHA kernel, so the [B,H,S,S] scores never hit HBM. After that, the
same epilogue fusions as dense (bias/act/scale) apply per batch.

## Fusion neighbors
- **→ attention (FMHA)**: QKᵀ·softmax·PV fused — the dominant case; standalone batched GEMM should only
  survive where attention isn't fused. See operators/attention_prefill_fmha/overview.md,
  operators/gqa_mqa_attention/overview.md, operators/mla_attention/overview.md.
- **+bias / +act epilogue (per batch)**: same CShuffle-style epilogue as dense →
  operators/gemm_epilogue_fused/overview.md.
- **+fp8 quant epilogue (per batch)**: scaled write-out so the next batched matmul consumes fp8 →
  operators/scaled_quant_gemm/overview.md.
- **→ grouped GEMM**: when per-batch shapes differ (MoE experts), fuse into a single grouped launch
  instead of padded batched → operators/grouped_gemm_moe/overview.md,
  operators/fused_moe_grouped_gemm/overview.md.

## Why it pays
The attention fusion removes the giant [B,H,S,S] scores tensor (memory-bound) — far larger than any
epilogue win. Per-batch epilogue fusions remove a [B,M,N] elementwise pass, same Amdahl logic as dense.

## How it's expressed per backend
- **CK**: `DeviceBatchedGemmMultipleD_Xdl_CShuffle` with a `CDEElementwise` epilogue per batch.
- **triton**: append epilogue after MFMA before the per-batch store; `OPTIMIZE_EPILOGUE=1`.
- For attention fusion, use the dedicated FMHA kernels, not a fused batched-GEMM.

## Pitfalls
- Fused shape/flags change the tuning key → tune the fused variant.
- Padding variable per-batch shapes into one batched kernel wastes FLOPs — prefer grouped GEMM.

## Sources
- CK batched MultipleD epilogue: ROCm "Optimizing with Composable Kernel".
- Epilogue args on live path: `ROCm/aiter@b467ce342:aiter/tuned_gemm.py`.
