---
title: batched_gemm — numerics
kind: quant
operator: batched_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - ROCm/aiter@HEAD:gradlib/gradlib/gemm_tuner.py
---

# batched_gemm — numerics

## TL;DR
Same numeric contract as dense GEMM — **fp32 accumulate per batch element**, independent across the
batch — so a same-dtype solution swap is parity-safe (`err_ratio<0.05`). The only batched-specific
caveat: **per-batch fp8 scales** must be applied per matmul, not globally, or accuracy collapses on
heads/experts with different dynamic range.

## Accumulation & dtype
- Each batch b accumulates in fp32 independently; no cross-batch reduction → no extra rounding vs B
  separate GEMMs (split-K within a batch does change reduction order, tiny ULP drift).
- fp8 batched: A[b],B[b] fp8 + scale[b] (per-batch / per-token-per-channel), fp32 accum, bf16 out.

## Parity bands
- Same-dtype solution swap: byte-comparable within band; gradlib gates `err_ratio<0.05`.
- fp8 batched quant: not byte-comparable; gate on task accuracy. Use per-batch scales (a shared scale
  across heads with differing magnitudes under-quantizes the large-range heads).
- Split-K batched: reproducible per pinned solution; cross-run ULP drift if solution not pinned.

## Tie-break / determinism
No argmax in the matmul itself. Determinism per pinned solution; batch order is fixed.

## How to gate
Same-math swap: `max_reldiff < 0.05` per batch vs reference. Quant: per-batch dequant reference +
downstream eval. Beware all-zero test tensors hiding per-batch scale errors.

## Sources
- fp8 GEMM accumulate/scale contract: ROCm CDNA4 GEMM blog.
- err_ratio<0.05 gate: `ROCm/aiter@HEAD:gradlib/gradlib/gemm_tuner.py`.
- Shared dense-GEMM numerics: operators/dense_gemm/numerics.md.
