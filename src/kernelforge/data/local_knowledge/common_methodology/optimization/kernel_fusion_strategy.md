---
title: kernel fusion strategy (when to fuse, donors, when not)
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, training, both]
updated: 2026-06-05
---

# kernel fusion strategy

## TL;DR
Fuse to **cut HBM round-trips** (a bandwidth-bound producer + consumer) and to **hide one op behind
another** (compute behind comm, or epilogue/prologue behind a GEMM's MFMA pipeline). The high-value LLM
fusions on MI300X: **epilogue into GEMM** (bias/act/scale/quant), **prologue into GEMM** (dequant/RMSNorm
of activations), **norm + quant**, **rope + KV-cache write**, and **comm + norm** (all-reduce/all-gather
fused with RMSNorm). Do **not** fuse when it kills occupancy, blocks a faster library kernel, or hurts
reuse. First classify the kernels (`[[optimization/roofline_and_bottlenecks.md]]`) — fusion only pays
when it changes the binding bottleneck. See `[[operators/gemm_epilogue_fused/overview.md]]`,
`[[operators/fused_norm_quant/overview.md]]`, `[[operators/fused_add_rmsnorm/overview.md]]`,
`[[operators/fused_allreduce_rmsnorm/overview.md]]`.

## Why fusion helps (the mechanics)
- **Removes a HBM pass**: two BW-bound elementwise ops chained through HBM ⇒ fuse to read once, write
  once. Halves traffic on the binding roof (`[[optimization/roofline_and_bottlenecks.md]]`).
- **Donor latency hiding**: a GEMM is the natural **fusion donor** — its MFMA pipeline has spare
  VALU/memory cycles; folding the bias/activation/quant epilogue (or the input dequant prologue) into
  it costs ~free time (`[[optimization/mfma_scheduling.md]]`).
- **Overlaps comm with compute**: comm+norm fusion lets the collective progress while the norm's VALU
  work runs (`[[operators/fused_allreduce_rmsnorm/overview.md]]`,
  `[[operators/reduce_scatter/overview.md]]`).

## High-value fusions (when to fuse)
| fusion | donor / pattern | payoff | card |
|---|---|---|---|
| **epilogue → GEMM** (bias, act, scale, fp8 quant) | GEMM donor | free epilogue, no C round-trip; use `OPTIMIZE_EPILOGUE=1` | `[[operators/gemm_epilogue_fused/overview.md]]`, `[[operators/scaled_quant_gemm/overview.md]]` |
| **prologue → GEMM** (dequant / norm of A) | GEMM donor | skip a pre-pass over activations | `[[operators/dense_gemm/overview.md]]` |
| **norm + quant** | both BW-bound | one pass, one write of quantized out + scale | `[[operators/fused_norm_quant/overview.md]]`, `[[operators/quant_dequant_fp8/overview.md]]` |
| **add + RMSNorm** (residual) | BW-bound chain | fuse residual add into norm | `[[operators/fused_add_rmsnorm/overview.md]]` |
| **rope + KV-cache write** | BW/latency-bound | apply rope and write paged KV in one pass | `[[operators/rope/overview.md]]`, `[[operators/paged_kv_copy/overview.md]]` |
| **comm + norm** (all-reduce/all-gather + RMSNorm) | overlap comm/compute | hide collective latency | `[[operators/fused_allreduce_rmsnorm/overview.md]]` |
| **moe routing + dispatch** | latency-bound | fewer launches, less traffic | `[[operators/moe_dispatch_combine/overview.md]]` |

## Fusion donors (what to fold *into*)
- **GEMM / attention** — the biggest donors (deep MFMA pipeline absorbs epilogue/prologue work).
- **A norm pass** — absorbs residual add, quant, scale compute.
- **A copy/cast pass** — absorbs quant or layout shuffle (`[[operators/layout_shuffle/overview.md]]`).

## When NOT to fuse
- **It blocks a faster library kernel**: a hand-fused GEMM+epilogue that loses to aiter's tuned GEMM +
  a cheap separate epilogue is a regression — the live lever is aiter
  (`[[optimization/autotuning_methodology.md]]`, `[[operators/dense_gemm/backends/aiter.md]]`).
- **It blows the register/LDS budget**: extra fused state drops occupancy below the latency-hiding
  threshold (`[[optimization/occupancy_and_registers.md]]`, `[[optimization/lds_and_bank_conflicts.md]]`).
- **It destroys reuse / changes AI badly**: fusing a high-reuse op into a streaming one can force
  recompute or extra traffic — re-check the roofline (`[[optimization/roofline_and_bottlenecks.md]]`).
- **The ops have different ideal grid/tile shapes**: forcing one launch geometry penalizes both.
- **Numerics**: fusing across a needed fp32-accumulate or rescale boundary can hurt accuracy
  (`[[optimization/numerical_stability.md]]`).

## Launch-bound decode: diagnosing the other kind of fusion
Everything above fuses to cut **HBM traffic**. Serving decode has a second, distinct
opportunity: once GEMM and attention are tuned, GPU-busy time drains into a long tail of
**tiny ops** — copy, elementwise, rmsnorm, rope, activation, reduce, sample — each paying a
full launch. Here the win is arithmetic on **launch count**, not on traffic, and the donor
is not a GEMM but the chain itself. See `[[languages/fusion/]]` for the decode pattern
cards and the CUDA-graph authoring rules.

**Capture the trace with CUDA graphs OFF.** With graphs on, the launches you are trying to
count are already amortized and the tail disappears from the trace.

**`launch_bound_share`** = the fraction of GPU-busy time spent in those tiny-op categories
(everything that is not GEMM / attention / MoE). Below **0.10** the workload is compute or
attention dominated and no decode fusion will pay; that is the candidate floor.

**Predicting the CUDA-graph-ON gain.** The share you measured with graphs off overstates
what you will get back with graphs on, because graph replay already removes most launch
overhead. Measured decode fusions landed in the low single digits of e2e gain while their
graphs-off launch-bound shares were 0.25-0.45 — roughly a **0.13** discount. Use that as the
prior when nothing better is available; a memory-traffic signal, when present, is the more
accurate channel. Treat a predicted gain below **3%** as not worth an authoring campaign.

Both numbers rank candidates rather than veto them: `launch_bound_share` is a poor
discriminator on its own, and a low share with a strong identified chain still beats a high
share with nothing fusible in it.

## Pitfalls
- Fusing for fewer launches when launch overhead isn't the bottleneck (measure first).
- Mismatched bias/shape in a fused GEMM epilogue ⇒ defeats the aiter lookup
  (`[[optimization/autotuning_methodology.md]]` 9-tuple).
- Over-fusing into a megakernel that spills and runs slower than the staged version.

## Verify
- Counters: HBM traffic should drop and the binding roof should change after fusion
  (`[[optimization/roofline_and_bottlenecks.md]]`, `[[profiling/]]`).
- A/B end-to-end: fused vs staged, median of ≥3 warm runs, non-overlapping bands.
- Numeric oracle vs unfused reference within tolerance (`[[optimization/numerical_stability.md]]`).
