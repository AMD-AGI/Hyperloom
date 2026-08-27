---
title: splitk_streamk_gemm — numerics
kind: technique
operator: splitk_streamk_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - https://arxiv.org/abs/2301.03598
  - https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
---

# splitk_streamk_gemm — numerics

## TL;DR
> The decomposition is mathematically identical to dense GEMM, but **the reduction across K-partitions
> changes summation order**, so atomic-add results are non-deterministic and can differ run-to-run; use a
> workspace reduction when you need bitwise reproducibility.

## Considerations
- **Accumulate in fp32** per partition; combine partials in fp32 before casting to out dtype.
- **Atomic reduction**: fast but non-associative across the differing add orders → results vary slightly
  run-to-run and differ from the single-pass dense result within fp rounding. Avoid bf16/fp16 atomics
  (large rounding) — accumulate/atomic in fp32.
- **Workspace reduction**: deterministic given a fixed partition→workspace layout; one extra read/write
  pass. Prefer for reproducibility-gated tests.
- **Stream-K fix-up**: partial output tiles owned by multiple workgroups must be combined exactly once;
  a fix-up bug double-counts or drops K-iterations (correctness, not just precision).
- For fp8/mxfp inputs, scales apply per dense rules before the cross-partition reduction →
  scaled_quant_gemm/numerics.md.

## Parity gate
- Oracle = single-pass dense fp32 reference. Expect agreement within fp accumulation tolerance; for the
  deterministic gate, use workspace reduction and require bitwise/near-bitwise match across repeats.

## Sources
- Stream-K (reduction/fix-up semantics): https://arxiv.org/abs/2301.03598
- Triton split-K/stream-K reduction modes: https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
