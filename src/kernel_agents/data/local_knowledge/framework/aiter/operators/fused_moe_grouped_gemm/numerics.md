---
title: fused_moe_grouped_gemm — numerics
kind: technique
operator: fused_moe_grouped_gemm
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, int8, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/configs/tuned_fmoe.csv
  - https://github.com/ROCm/aiter/issues/2421
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# fused_moe_grouped_gemm — numerics & parity

## What changes the numbers
The grouped GEMM is same-math GEMM (parity-safe tiling) **until you quantize**. The numeric risk is the
**fp8/int8/fp4 quant** of activations and weights, plus the **order** of the activation, routed-weight
multiply, and combine.

## Accumulation
fp32 accumulate for both stages on the MFMA path (matrix-core blog). The intermediate `A = act(G)⊙U` is
commonly kept bf16; quantizing it to fp8 before stage-2 (down) is an extra error source — block-scale fp8
keeps it bounded but it is a gate.

## Quant error (the gate)
- **fp8 block-scale / per-token** (E4M3FNUZ on gfx942, OCP on gfx950): the DB rows carry per-stage `err1`,
  `err2` (e.g. stage-2 ~2.3%). That is **kernel tolerance vs an fp8 reference**, not task accuracy —
  validate end-to-end (a small eval), not just the kernel `err`.
- **FP8 fused MoE precision** has open issues (aiter #2421) — fp8 MoE can lose accuracy on some models;
  re-run the eval when enabling, and prefer bf16 if the loss is unacceptable.
- **int8 / A4W4 (fp4)**: larger quant error; A4W4 only on CDNA4. fp4 is aggressive — gate hard.
- **fnuz vs OCP**: gfx942 fp8 is **fnuz** (exponent bias off-by-one); a weight/activation read in the wrong
  dialect is off by exactly **2×**. Match the dialect to the arch — this is the #1 silent fp8 bug.

## Routed-weight multiply ordering
The routed weight is multiplied in stage-1 (`doweight_stage1`), stage-2 epilogue (`MulRoutedWeight1`), or
in combine. Each is a different rounding point. Keep it **bf16/fp32** (never fp8) and use the **unbiased**
routing weight (DeepSeek bias is for selection only). Mismatched ordering vs the reference drifts the
output.

## Benign vs real divergence
- Benign: bf16 grouped-GEMM tiling differences and combine reduction order → small per-token deltas;
  argmax flips on long greedy decode near ties. Gate with a parity probe ≥10 prompts.
- Real: wrong fp8 dialect (2× off), wrong weight ordering, fp8 MoE accuracy loss on a sensitive model,
  CK coverage gap producing a wrong-shape fallback.

## Coverage-gap correctness trap
CK stage-2 may not cover an odd expert/inter shape → "device_gemm does not support this GEMM problem"
(crash, not silent), or a Triton fallback that is correct but slow. Pad/tune to a covered shape; don't
assume the fast kernel ran (check `AITER_LOG_MORE=1`).

## Verification recipe
1. Isolated: grouped-GEMM output vs a torch reference (bf16 within tolerance; fp8 within the DB `err`).
2. e2e: greedy/temp=0 parity on the MoE model, bf16 path first (parity-safe), then the quant path.
3. **Eval gate** (gsm8k or similar) for every quant path — the DB `err` is not a task-accuracy guarantee.

## Sources
- per-stage `err` columns: `ROCm/aiter@b467ce342:aiter/configs/tuned_fmoe.csv`.
- FP8 fused MoE precision issue: https://github.com/ROCm/aiter/issues/2421
- fp32 accumulate / fnuz fp8 on CDNA3: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- cross-ref: scaled_quant_gemm numerics, grouped_gemm_moe.
