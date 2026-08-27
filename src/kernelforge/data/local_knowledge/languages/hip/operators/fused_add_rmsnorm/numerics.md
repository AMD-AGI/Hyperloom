---
title: fused_add_rmsnorm — numerics & parity
kind: technique
operator: fused_add_rmsnorm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py
  - https://github.com/vllm-project/vllm/issues/42325
---

# fused_add_rmsnorm — numerics & parity

Inherits rmsnorm/numerics.md (fp32 Σx², fp32 γ promote, ε inside mean, fnuz fp8,
reduction-order re-gate) plus the residual-add specifics.

## 1. The add is in IO dtype; the reduction is fp32
`r' = x + residual_in` is computed and **stored** in the IO dtype (bf16/fp16) — `residual_out` must match
the running residual stream's dtype exactly (it feeds the next block's skip connection). The Σ(r')² for the
norm then promotes `r'` to fp32. Doing the add in fp32 and rounding to bf16 once is fine; doing the norm on
an fp32 `r'` while storing a bf16 `residual_out` would make the written residual and the normalized value
inconsistent — keep them consistent (normalize the **rounded** r' that you store).

## 2. residual_out parity is load-bearing
Because `residual_out` propagates through every subsequent layer, a tiny error compounds. The fused kernel
must produce a `residual_out` bit-identical (or within bf16-rounding) to the unfused `x + residual_in`.
This is the main correctness check that distinguishes fused-add-rmsnorm from plain rmsnorm.

## 3. γ fp32-promote (same regression as #42325)
`y = fp32(r')·rsqrt(...)·fp32(γ)`. The vLLM #42325 class applies here too.

## 4. quant-stacked variant
`rmsnorm2d_fwd_with_add_dynamicquant`: add → norm → fp8 quant, scale in fp32, fnuz on gfx942. The
`residual_out` is still bf16 (the residual stream isn't quantized); only `y` becomes fp8. Task gate. See
fused_norm_quant.

## Parity gate
1. `residual_out` vs fp64 `x+r` within bf16-rounding (this is the new check).
2. `y` vs fp64 `rmsnorm(x+r)` within band.
3. e2e greedy parity (residual error compounds → this catches it).
4. quant variant: task eval + fnuz dialect.

## Sources
- add-in-IO-dtype + fp32 reduction + store residual_out: `/sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py`.
- γ fp32-promote regression: https://github.com/vllm-project/vllm/issues/42325.
