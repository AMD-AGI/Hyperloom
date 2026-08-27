---
title: grouped_gemm_moe — numerics
kind: technique
operator: grouped_gemm_moe
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - https://github.com/ROCm/aiter
---

# grouped_gemm_moe — numerics

## TL;DR
> Same accumulation rules as dense GEMM (fp32 accumulate), with two extra accuracy seams: **per-expert
> quant scales** and the **fp32 routing-weight multiply** during combine — keep both in fp32 and apply
> scales in the documented order.

## Considerations
- **Accumulate fp32** for all dtypes; cast to out dtype (bf16/fp16) only at the epilogue.
- **Per-expert / per-block scales**: fp8 (e4m3 on CDNA3 FNUZ, OCP fp8 on CDNA4) and mxfp4/mxfp6 use
  per-group E8M0 block scales applied after the MFMA dot, before accumulation — see
  scaled_quant_gemm/numerics.md.
- **Routing weight multiply**: top-k gate weights multiply expert outputs during combine; do this in fp32
  to avoid bias on summed contributions of a token routed to multiple experts.
- **Tie-break**: argmax/top-k tie-breaking lives upstream in routing
  (moe_routing_topk/numerics.md), not in this GEMM.
- **Padding rows must not contaminate output**: padded tokens (align&sort) must write to a sink or be
  masked; a bug here corrupts real tokens' partial sums.

## Parity gate
- Oracle = per-expert dense reference `X_e @ W_e` in fp32, gathered back and combined with fp32 gate
  weights. For bf16 expect tight relative error; for fp8/fp4 use the quant accuracy gate from
  scaled_quant_gemm/numerics.md (e.g. err_ratio band, MMLU/PPL check).

## Sources
- AITER repo (scaled grouped-GEMM paths): https://github.com/ROCm/aiter
