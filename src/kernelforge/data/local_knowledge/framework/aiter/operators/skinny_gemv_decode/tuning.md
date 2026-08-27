---
title: skinny_gemv_decode — tuning
kind: technique
operator: skinny_gemv_decode
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/tuned_gemm.py
  - https://github.com/ROCm/aiter
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
---

# skinny_gemv_decode — tuning

## TL;DR
> Because it's bandwidth bound, tuning = **maximize weight-read efficiency and fill all CUs**: use split-K
> (wvSplitK-style) so the few output tiles spread across 304/256 CUs, prefer mfma_16x16 (or pure VALU GEMV)
> to avoid wasting lanes on the tiny M, and ensure coalesced W loads.

## The levers
- **Split-K across CUs**: with M=1..8 there are very few output tiles → split K so partial sums fill all
  CUs, then reduce. This is the wvSplitK pattern. See splitk_streamk_gemm/overview.md.
- **mfma_16x16 vs 32x32 vs VALU**: for M=1..8, 32x32 MFMA wastes most of its M lanes; mfma_16x16 wastes
  fewer, and for pure GEMV a VALU dot can beat MFMA entirely. Pick by M.
- **N/K tiling for coalesced W reads**: weight bandwidth dominates — tile so consecutive threads read
  consecutive W elements; avoid strided/uncoalesced loads.
- **padded_M bucket**: aiter rounds M up (`get_padded_m`, exposed in `aiter/ops/gemm_op_common.py`); small
  padded_M is exactly what selects the skinny path (10-tuple gfx-first key) → [./backends/aiter.md](aiter.md).
- **num_warps / waves_per_eu**: raise occupancy to hide HBM latency.

## Heuristic
- M=1: GEMV / heavy split-K. M=2..8: skinny GEMM with mfma_16x16 + split-K. M≳16: switch to dense GEMM.

## Pitfalls
- Using a 32x32-MFMA dense kernel at M=1 wastes ~31/32 of the matrix lanes → far below bandwidth-bound
  optimum.
- Too much split-K adds reduction overhead that exceeds the BW gain — sweep it.

## Verify
- Report achieved HBM GB/s vs peak (the real ceiling); A/B vs the dense GEMM kernel at the same shape.

## Sources
- AITER skinny path (padded_M selection): https://github.com/ROCm/aiter
- vLLM ROCm decode GEMV / split-K kernels: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
