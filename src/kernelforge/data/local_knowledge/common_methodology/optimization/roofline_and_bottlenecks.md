---
title: roofline and bottleneck classification
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, training, both]
updated: 2026-06-05
---

# roofline and bottleneck classification

## TL;DR
Before optimizing, **classify the kernel**: compute-bound vs bandwidth-bound, decided by **arithmetic
intensity** (FLOP per byte of HBM traffic) against the machine balance. Optimizing a bandwidth-bound
kernel for MFMA occupancy (or a compute-bound kernel for coalescing) wastes effort. Reality check:
MI300X sustains only **~45–55% of theoretical peak** matrix throughput across fp8/bf16/fp16 — a
software-maturity ceiling, not hardware — so the bar is the **best tuned library kernel**, never peak.
Use Omniperf's roofline (`[[profiling/]]`). See `[[hardware/cdna3_mi300/peak_tables.md]]`,
`[[hardware/cdna4_mi350/peak_tables.md]]`, `[[hardware/shared/hbm_infinity_fabric.md]]`.

## Concepts
- **Arithmetic intensity (AI)** = FLOPs / HBM bytes moved. Compare to **machine balance** =
  peak FLOP/s ÷ peak HBM BW (the roofline ridge point). `AI > balance` ⇒ compute-bound; `AI < balance`
  ⇒ bandwidth-bound.
- **Roofline**: achievable = `min(peak_compute, AI × peak_BW)`. The kernel's measured point sits under
  one of the two roofs — that roof is your bottleneck.
- **The ~45% reality**: MI300X delivers ~45–55% of theoretical matrix peak in practice (third-party
  bf16 ceiling ~890 TFLOP/s ≈ 68% of 1.3 PFLOP/s, power-limited at 750 W). Treat measured
  best-library throughput as the practical roof (`[[operators/dense_gemm/tuning.md]]`,
  `[[hardware/cdna3_mi300/clocks_power.md]]`).

## How to classify a kernel (procedure)
1. **Estimate AI analytically**: GEMM `M·N·K·2` FLOPs over `(M·K + K·N + M·N)·sizeof(dtype)` bytes
   (if it streams from HBM once). Large square GEMM ⇒ high AI ⇒ compute-bound; GEMV/decode, norm,
   elementwise, copy ⇒ low AI ⇒ bandwidth-bound.
2. **Measure**: Omniperf roofline / counters — `VALUBusy` & MFMA-busy vs HBM read+write BW. If MFMA
   busy is high and HBM far from peak ⇒ compute-bound; if HBM near peak and MFMA idle ⇒ bandwidth-bound;
   if both low ⇒ **latency/occupancy-bound** (stalls).
3. **Pick the lever set** from the table below.

## Bottleneck → lever map
| classification | symptom (counters) | levers |
|---|---|---|
| compute-bound | MFMA busy high, HBM low | `[[optimization/mfma_scheduling.md]]`, `[[optimization/occupancy_and_registers.md]]`, tile/MFMA shape (`[[operators/dense_gemm/tuning.md]]`) |
| bandwidth-bound | HBM near peak, MFMA idle | `[[optimization/vectorization_and_coalescing.md]]`, `[[optimization/xcd_l2_locality.md]]` (L2 reuse), `[[optimization/kernel_fusion_strategy.md]]` (cut traffic) |
| latency/occupancy-bound | both low, high stall cycles | `[[optimization/memory_pipelining.md]]`, more waves/EU, more workgroups (`[[optimization/wave_and_grid_sizing.md]]`) |
| LDS-bound | high `ds_*` stalls | `[[optimization/lds_and_bank_conflicts.md]]` (swizzle/padding) |

## Typical LLM classifications
- **Prefill GEMM / attention scores**: compute-bound (keep 304/256 CUs at high MFMA occupancy).
- **Decode GEMV / KV read / sampling**: bandwidth/latency-bound (coalesce, split-K to fill CUs).
- **RMSNorm / LayerNorm / elementwise / cast**: bandwidth-bound (fuse to cut passes,
  `[[optimization/kernel_fusion_strategy.md]]`).

## Pitfalls
- Optimizing the wrong roof (MFMA tuning a bandwidth-bound norm).
- Using theoretical peak as the denominator for "efficiency" — use measured best-library or the ~45–55%
  practical roof.
- Ignoring the third roof: many real kernels are **latency-bound** (under-occupied / stalled), not
  cleanly compute- or BW-bound.
- Forgetting fusion *changes AI* — fusing two BW-bound kernels can push the result toward compute-bound.

## Verify
- Omniperf roofline plot: kernel point vs HBM and compute roofs (`[[profiling/]]`).
- Counters: MFMA/`VALUBusy` vs HBM BW; stall-cycle breakdown for the latency case.
- Recompute AI after any fusion and re-classify.
