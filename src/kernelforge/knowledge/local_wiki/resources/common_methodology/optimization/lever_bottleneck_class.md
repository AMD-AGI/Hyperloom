---
title: bottleneck classification — pick the roof before you pick a lever
kind: lever
lever: bottleneck_class
gens: [gfx950]
bottleneck: entry-point (routes to every other lever)
updated: 2026-08-28
---

# Bottleneck classification

**This is the entry point. Run it before pulling any other lever.** Every lever in this folder is
useless — and often harmful — against the wrong bottleneck. Tuning MFMA shape on a bandwidth-bound
RMSNorm changes nothing; raising occupancy on a compute-bound GEMM makes it slower.

## Route here when
You have a kernel and a measurement, and you do not yet know **which roof it is under**. That is the
only prerequisite. If you have no measurement, go to `measure_protocol.md` first — classifying from
source-reading is guesswork.

## The four classes

| Class | Counter signature | Go to |
|---|---|---|
| **Compute-bound** | MFMA busy high (>60%), HBM BW well under peak | `lever_mfma_sched.md` → `lever_occupancy.md` |
| **Bandwidth-bound** | HBM BW near achievable peak, MFMA busy low | `lever_coalescing.md` → `lever_fusion.md` → `lever_xcd_locality.md` |
| **Latency / occupancy-bound** | *Both* low, high stall cycles, few resident waves | `lever_prefetch.md` → `lever_grid_sizing.md` → `lever_occupancy.md` |
| **LDS-bound** | `ds_*` stall cycles high, bank-conflict counter non-zero | `lever_lds_banks.md` |

"Both low" is the most common real answer and the most commonly misdiagnosed — an under-occupied or
stalled kernel looks like neither of the textbook two. Check it explicitly before assuming
compute/bandwidth.

## gfx950 machine balance (MI350X / MI355X)

Arithmetic intensity `AI = FLOPs / HBM bytes`. Compare against the **ridge point** for your dtype:

| dtype | peak | ridge (peak ÷ 8 TB/s) |
|---|---|---|
| FP16 / BF16 | 2.5 PFLOP/s | **≈ 312 FLOP/byte** |
| FP8 (OCP) | 5 PFLOP/s | ≈ 625 FLOP/byte |
| FP6 / FP4 | 10 PFLOP/s | ≈ 1250 FLOP/byte |
| FP32 | 157 TFLOP/s | ≈ 20 FLOP/byte |
| INT8 | ~5 POPS | ≈ 625 OP/byte |

`AI > ridge` ⇒ compute side. `AI < ridge` ⇒ bandwidth side. HBM3E is **288 GB @ 8.0 TB/s**.

The ridge on gfx950 is **higher than the previous generation** (≈312 vs ≈247 FP16) because the matrix
core doubled while bandwidth grew less. Practical consequence: **more kernels are bandwidth-bound here
than on MI300X.** A kernel that was borderline compute-bound before may now sit left of the ridge.

## Estimate AI before you measure (30 seconds, catches most cases)

GEMM `M×N×K`: `2·M·N·K` FLOPs over `(M·K + K·N + M·N)·sizeof(dtype)` bytes, assuming each operand
streams from HBM once.

| Shape class | AI | Verdict |
|---|---|---|
| Large square GEMM (prefill, M≥2048) | high | compute |
| Skinny GEMM / GEMV (decode, M=1..8) | ~2 | bandwidth |
| RMSNorm / LayerNorm / elementwise / cast | <1 | bandwidth |
| Attention prefill (fused) | moderate–high | compute |
| Attention decode / paged KV read | low | bandwidth |
| Softmax standalone, top-k, sampling | <1 | bandwidth |

If the analytic estimate and the counters disagree, trust the counters — the estimate assumes perfect
cache behaviour and ignores re-fetch.

## The bar is not peak

Tuned GEMM sustains **~45–55% of theoretical matrix peak** on Instinct. That gap is a software-maturity
ceiling, not a hardware defect. Consequences for how you judge a kernel:

- **Never** report efficiency against the datasheet number. A kernel at 50% of peak FP16 may already
  match the best library kernel.
- The real bar is **the best tuned library kernel for that shape** — measure it and use that as the
  denominator (`lever_autotune.md` for how to get one on the live path).
- If you are at ~50% of peak and the counters say compute-bound, the remaining headroom is small.
  Re-check whether a lower-precision path (FP8, MXFP4) is available before grinding the schedule.

## Fusion changes the answer — re-classify after every fusion

Fusing two bandwidth-bound kernels removes an HBM round-trip, which *raises* AI. The fused kernel can
land in a different class than either input. Always recompute AI and re-run this card after applying
`lever_fusion.md`. The same applies after a dtype change: moving BF16→FP8 halves the bytes and doubles
the peak, moving the point diagonally.

## Verify

| Check | How | Pass condition |
|---|---|---|
| Which roof | `measure_roofline.md`, empirical `--roof-only` roofs | kernel point sits clearly under one roof |
| Counter cross-check | `measure_triage.md` | MFMA-busy and HBM-BW agree with the roofline verdict |
| Post-change | re-run both | the point moved **up or right toward a roof**, not merely lower wall time |

A change that lowers wall time but leaves the point in the same place relative to the roofs usually
means you shifted work elsewhere rather than removing a bottleneck.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Lever applied, no change | wrong class | re-run this card; check the "both low" case |
| "Only 50% of peak, must be broken" | using datasheet peak as the bar | compare against best tuned library |
| Classification flips between runs | measurement noise, cold clocks | `measure_protocol.md` — warm, REPEATS=7, locked clocks |
| BW-bound verdict but HBM counter low | working set fits Infinity Cache (256 MiB) — you are L2/L3-bound, not HBM-bound | check L2/L3 hit rate; `lever_xcd_locality.md` |

## Deeper

`hardware/mi350_overview.md` (the peak and ridge numbers) ·
`hardware/mi350_memory.md` (the bandwidth ladder) ·
`measure_roofline.md` (building the empirical roofs) ·
`measure_triage.md` (the counter-level decision flow)
