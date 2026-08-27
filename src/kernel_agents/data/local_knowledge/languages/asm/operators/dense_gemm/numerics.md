---
title: dense_gemm — numerics
kind: quant
operator: dense_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, mxfp4]
regimes: [prefill, decode, training]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://blog.vllm.ai/2025/02/24/ptpc-fp8-rocm.html
  - ROCm/aiter@HEAD:gradlib/gradlib/gemm_tuner.py
---

# dense_gemm — numerics

## TL;DR
bf16/fp16 GEMM with **fp32 accumulate** is the parity baseline; swapping library *solutions*
(hipBLASLt/CK/asm) is same-math → byte-comparable within `err_ratio<0.05`. Quantized variants
(fp8/mxfp4) change the math and must be gated on **task accuracy**, never byte parity.

## Accumulation & dtype contract
- In: bf16/fp16. Accumulate: **fp32** (MFMA accumulators are fp32). Out: bf16 (sglang `nn.Linear`).
- fp8 (E4M3FNUZ on gfx942 / E4M3FN on gfx950): A,B fp8 + per-tensor or per-token/per-channel scales,
  fp32 accumulate, bf16 out. CDNA4 adds native fp4/fp6 dense + block-scaled MFMA.

## Parity bands
- **Same-dtype solution swap (the aiter tune)**: parity-safe. gradlib gates each candidate on
  `err_ratio < 0.05` vs a reference GEMM; accepted swaps move no task metric.
- **fp8 quant**: not byte-comparable. PTPC-FP8 (per-token-activation, per-channel-weight) on vLLM/ROCm
  recovers near-bf16 accuracy and is the recommended fp8 recipe; per-tensor fp8 is faster but lossier.
- **mxfp4 (CDNA4)**: per-32-element E8M0 block scale. Naive MXFP4 loses accuracy; online-rotation +
  SmoothQuant gets near-lossless (W4A16/W4A8). Gate on downstream eval, not numeric tolerance.

## Tie-break / determinism
- Plain dense GEMM has no argmax/tie-break; output order is deterministic per solution. Split-K /
  Stream-K introduce a different fp32 reduction order → tiny ULP drift (still within band), so pin the
  solution if exact reproducibility across runs is required.

## How to gate
- Same-math swap: `max_reldiff < 0.05` vs untuned reference on the live shape (gradlib does this).
- Quant: run the task eval (e.g. accuracy on the serving workload) A/B; accept only if no regression.
- Watch the input-distribution effect: GEMM perf varies >20% by input values, but accuracy gating must
  use representative (not all-zero) tensors or you under-detect quant error.

## Sources
- fp8 GEMM on CDNA4 (E4M3FN, bf16 out, fp32 accum): ROCm CDNA4 GEMM blog.
- PTPC-FP8 accuracy recipe: vLLM blog 2025-02-24.
- err_ratio<0.05 gate: `ROCm/aiter@HEAD:gradlib/gradlib/gemm_tuner.py`.
- mxfp4 near-lossless: ROCm MXFP4 online-rotation blog (see fusion.md / scaled_quant_gemm).
