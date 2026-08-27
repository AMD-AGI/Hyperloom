---
title: profiling — reading a kernel bottleneck (decision flow)
kind: technique
gens: [gfx942, gfx950]
updated: 2026-06-08
---

# Reading a kernel bottleneck: compute / BW / latency / occupancy

## TL;DR
Four failure modes, four counter signatures. Decide in this order:
1. **Compute-bound** — MFMA/VALU busy is high, on or near the compute roof.
2. **BW-bound** — HBM/L2 traffic high, on the BW roof, MFMA busy low.
3. **Occupancy-limited** — both roofs far away, few waves/CU resident (VGPR/LDS/workgroup limited).
4. **Latency-bound** — both roofs far away, occupancy is fine, but issue stalls dominate (memory or
   dependency latency not hidden by enough in-flight work).
Plot the kernel on the roofline first ([`roofline_on_mi.md`](roofline_on_mi.md)); then use counters
([`rocprof_compute_workflow.md`](rocprof_compute_workflow.md)) to disambiguate 3 vs 4.

## Decision flow
```
            ┌─ near COMPUTE roof? ── yes → COMPUTE-BOUND
roofline ───┤
   point    ├─ near BW roof?      ── yes → BANDWIDTH-BOUND
            │
            └─ far from BOTH roofs → look at occupancy
                         │
                         ├─ low waves/CU (VGPR/LDS/WG cap) → OCCUPANCY-LIMITED
                         └─ occupancy OK but high STALL %  → LATENCY-BOUND
```

## What each verdict looks like in counters
| Verdict | Tells | Fixes (link) |
|---|---|---|
| **Compute-bound** | `SQ_VALU_MFMA_BUSY_CYCLES` high; on compute roof; VALU/MFMA SoL ~peak | already near library; chase tile/MFMA shape, dtype downcast → [`../../framework/aiter/operators/dense_gemm/tuning.md`](../../framework/aiter/operators/dense_gemm/tuning.md) |
| **BW-bound** | high TCC miss + HBM bytes; AI left of ridge; MFMA busy low | raise arithmetic intensity: fuse, tile for L2/Infinity Cache reuse, larger BLOCK_K |
| **Occupancy-limited** | few waves/CU; VGPR or LDS per-wave high; <1024 workgroups | cut register/LDS use, `waves_per_eu`, more workgroups → [`../../hardware/cdna3_mi300/occupancy.md`](../../hardware/cdna3_mi300/occupancy.md) |
| **Latency-bound** | high ISSUE/STALL (ATT Summary), low IPC, occupancy fine | more in-flight: increase ILP, prefetch, `num_stages`, software pipelining |

## Heuristics specific to MI300/MI350
- **MFMA busy near peak but only ~45–55% of theoretical FLOPS**: that is the known software-maturity
  ceiling for fp8/bf16 GEMM on MI300X — you are likely already compute-bound vs the *best library*, so
  the bar is the tuned aiter/CK kernel, not theoretical peak
  ([`../../framework/aiter/operators/dense_gemm/tuning.md`](../../framework/aiter/operators/dense_gemm/tuning.md)).
- **High HBM bytes but low TCC hit**: data does not fit / is not reused across the Infinity Cache —
  classic BW-bound; tile for L2 reuse and check XCD placement
  ([`../../hardware/cdna3_mi300/xcd_chiplet.md`](../../hardware/cdna3_mi300/xcd_chiplet.md)).
- **Skinny decode GEMV / attention-decode**: almost always BW- or latency-bound, not compute-bound —
  do not chase MFMA occupancy; chase memory and launch overhead.
- **Many tiny kernels with big gaps**: not a kernel problem at all — host/launch overhead. Attack it
  with HIP-graph capture / dispatch collapse, not kernel tuning (see
  [`benchmarking_methodology.md`](benchmarking_methodology.md)).

## How to drive it
1. `rocprof-compute profile --roof-only` → place the point ([`rocprof_compute_workflow.md`](rocprof_compute_workflow.md)).
2. If far from both roofs, full `profile` + `analyze` SoL + memory chart.
3. Read MFMA-busy / TCC / HBM / waves-per-CU per the table above.
4. Apply one lever, re-profile, A/B ([`benchmarking_methodology.md`](benchmarking_methodology.md)).

## Pitfalls
- Calling a kernel "compute-bound" because it is slow — slow ≠ compute-bound. Place it on the roofline.
- Confusing occupancy- vs latency-bound: both sit far from the roofs; only waves/CU + STALL% separate them.
- Optimizing a kernel that is 2% of runtime — Amdahl. Pick targets from the trace's longest bars first.

## Verify
After the fix, the roofline point should move *toward* a roof (or up the BW roof), and the targeted
counter (MFMA busy ↑, or HBM bytes ↓) should change in the predicted direction — not just wall time.
