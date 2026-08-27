---
title: splitk_streamk_gemm — fusion
kind: technique
operator: splitk_streamk_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-05
---

# splitk_streamk_gemm — fusion

## TL;DR
> Fusion happens in the **reduction/fix-up epilogue**: once partial sums are combined, fold bias /
> activation / dequant-scale there so the result needs no extra pass — but only after the cross-partition
> reduction completes (epilogue must run once per output element, not once per partition).

## Fusion neighbors
- **Bias + activation epilogue**: applied after the final reduction → [../gemm_epilogue_fused/overview.md](../gemm_epilogue_fused/overview.md).
- **Dequant / output quant scale**: for fp8/mxfp GEMMs, apply scales in the epilogue after combining
  partials → [../scaled_quant_gemm/fusion.md](../scaled_quant_gemm/fusion.md).
- **Decode small-M**: the decode GEMV path often uses split-K internally (e.g. wvSplitK-style) →
  [../skinny_gemv_decode/overview.md](../skinny_gemv_decode/overview.md).

## Caveat
- Do **not** apply bias/activation per K-partition (split-K) — it would be added `SPLIT_K` times. Apply
  only in the final reduction step. With atomic-add reduction, bias must be initialized into the output
  buffer once (or added in a separate finalize), never inside the per-partition atomic.
