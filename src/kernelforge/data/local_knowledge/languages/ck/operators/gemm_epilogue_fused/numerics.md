---
title: gemm_epilogue_fused — numerics
kind: quant
operator: gemm_epilogue_fused
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, mxfp4]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://blog.vllm.ai/2025/02/24/ptpc-fp8-rocm.html
  - https://rocm.blogs.amd.com/software-tools-optimization/optimizing-with-composable-kernel.html
---

# gemm_epilogue_fused — numerics

## TL;DR
Fusing bias/act/residual is **parity-safe** as long as the epilogue is computed in **fp32 on the
accumulator before down-cast** — a fused kernel then matches (or is *more* accurate than) the unfused
GEMM-then-elementwise chain (which down-casts twice). The **quant epilogue** changes the math and must be
task-accuracy gated, never byte-compared.

## Accumulation & dtype contract
- GEMM accumulates fp32; apply `α`, `+bias`, `+β·residual`, then `act`, all in fp32; down-cast to bf16
  (or quantize to fp8/fp4) **last**.
- Fused vs unfused: the unfused path stores bf16 after the GEMM, reloads, then does elementwise → an
  extra round-trip rounding. Fused keeps fp32 through the epilogue → equal-or-better accuracy.

## Parity bands
- **bias/act/residual fusion (same dtype)**: parity-safe; expect ≤ unfused error. If you require byte
  match to a specific unfused reference, that reference's double-rounding may differ by ULPs.
- **fp8 output quant (PTPC-FP8)**: not byte-comparable; per-token-activation / per-channel-weight scales
  recover near-bf16 task accuracy (recommended fp8 recipe). Per-tensor is faster, lossier.
- **mxfp4 output (CDNA4)**: per-32 E8M0 block scale; gate on downstream eval (online-rotation +
  SmoothQuant → near-lossless).

## Tie-break / determinism
No argmax. Deterministic per pinned solution; split-K changes reduction order (ULP drift, still in band).
Activation must be a numerically-stable formulation (e.g. gelu tanh-approx consistent with the reference).

## How to gate
- Non-quant fusion: `max_reldiff < 0.05` vs fp32-epilogue reference (not vs a double-rounded one).
- Quant epilogue: dequant reference + downstream task eval A/B; accept only if no regression.
- Use representative (non-zero) tensors so per-channel/per-token scale errors surface.

## Sources
- fp8 GEMM accumulate/scale + bf16 out: ROCm CDNA4 GEMM blog.
- PTPC-FP8 accuracy: vLLM blog 2025-02-24.
- CShuffle epilogue (fp32 before down-cast): ROCm "Optimizing with Composable Kernel".
- Shared dtype contract: operators/dense_gemm/numerics.md.
