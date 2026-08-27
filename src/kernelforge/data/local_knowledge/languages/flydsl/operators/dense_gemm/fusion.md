---
title: dense_gemm — fusion
kind: technique
operator: dense_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
---

# dense_gemm — fusion

## TL;DR
The biggest fusion wins are **GEMM epilogue fusions** (bias/act/residual/quant) — folding the elementwise
pass into the GEMM C write-out removes a full [M,N] memory round-trip. The dedicated card for this is
[[operators/gemm_epilogue_fused/overview.md]]; this note covers what dense_gemm fuses with and why.

## Fusion neighbors
- **+bias**: free in the epilogue (CShuffle `D = A·Bᵀ + bias`). Note `bias` is part of aiter's lookup key.
- **+act (silu/gelu) `act_and_mul`**: the up/gate MLP GEMM fuses the gated activation, removing a
  separate [M, inter*2] elementwise pass → see [[operators/act_and_mul_silu_gelu/overview.md]] and
  [[operators/gemm_epilogue_fused/overview.md]].
- **+residual add**: fold the skip connection into the epilogue, killing a [M,N] read+write.
- **+fp8/fp4 quant**: quantize C in the epilogue (scaled write-out) so the next GEMM consumes fp8
  directly → see [[operators/scaled_quant_gemm/overview.md]] and [[operators/quant_dequant_fp8/overview.md]].
- **norm→GEMM**: RMSNorm+quant before the GEMM is fused on the *producer* side →
  see [[operators/fused_norm_quant/overview.md]].

## Why it pays (Amdahl)
GEMM is compute-bound; the appended epilogue is essentially free vs the matmul FLOPs, but the *separate*
elementwise kernel it replaces is a full memory-bound pass over [M,N]. On MLP up/gate (N up to 34816)
that pass is large, so fusing `act_and_mul` is one of the better cheap wins.

## How it's expressed per backend
- **CK**: `DeviceGemmMultipleD_Xdl_CShuffle` with a `CDEElementwise` op (bias/act/residual via the
  CShuffle stage) → [[languages/composable_kernel/...]].
- **triton/tilelang/flydsl**: append the epilogue inside the kernel after the MFMA accumulation, before
  the store; use `OPTIMIZE_EPILOGUE=1` to avoid the 512B Tagram hotspot on the C write.
- **aiter/hipBLASLt**: bias + scale are supported epilogue args on the tuned path.

## Pitfalls
- Fusing changes the lookup key (bias/act/scale flags) → must capture/tune the *fused* shape, or the
  tuned DB misses (same failure mode as bias mismatch in [backends/aiter.md](backends/aiter.md)).
- Over-fusing rare paths increases kernel-variant count and tuning time for little Amdahl gain.
