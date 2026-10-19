---
title: counter triage — turning a profile into one of four verdicts
kind: measure
measure: triage
gens: [gfx950]
updated: 2026-08-28
---

# Counter triage

Takes a profile and produces **one verdict**, which selects the lever. This is the step between
"the kernel is slow" and "here is what I am changing."

## Route here when
You have a roofline point and/or counters from `measure_rocpc_workflow.md` and need to decide which
lever to pull. If you have no measurement yet, start at `measure_protocol.md` — classifying from
source-reading is guesswork.

## The decision flow

```
            ┌─ near COMPUTE roof? ──── yes → COMPUTE-BOUND
roofline ───┤
   point    ├─ near BW roof? ───────── yes → BANDWIDTH-BOUND
            │
            └─ far from BOTH roofs → check occupancy
                         │
                         ├─ low waves/CU (VGPR / LDS / WG cap) → OCCUPANCY-LIMITED
                         └─ occupancy fine, high STALL %       → LATENCY-BOUND
```

**"Far from both roofs" is the most common real answer.** Occupancy-limited and latency-bound both live
there, and only `waves/CU` + stall% separate them. Do not collapse them — they take different levers.

## The four verdicts

| Verdict | Counter signature | Lever |
|---|---|---|
| **Compute-bound** | `SQ_VALU_MFMA_BUSY_CYCLES` high; on the compute roof; MFMA SoL near peak | `lever_mfma_sched.md` — shape, wave pattern, accumulator count |
| **Bandwidth-bound** | high TCC miss + HBM bytes; AI left of the ridge; MFMA busy low | `lever_coalescing.md` → `lever_fusion.md` → `lever_xcd_locality.md` |
| **Occupancy-limited** | few waves/CU; high VGPR or LDS per wave; <1024 workgroups | `lever_occupancy.md`, `lever_grid_sizing.md` |
| **Latency-bound** | high issue/stall, low IPC, occupancy **fine** | `lever_prefetch.md` — more in-flight work, deeper pipeline |
| **LDS-bound** (sub-case) | `ds_*` stall cycles high, bank-conflict counter non-zero | `lever_lds_banks.md` |

## gfx950 reading notes

- **MFMA busy near peak but only ~45–55% of theoretical FLOPS** — that is the known software-maturity
  ceiling, not a defect. You are compute-bound *relative to the best library*, so the bar is the tuned
  library kernel, not the datasheet. Remaining headroom is small; consider a lower-precision path
  (FP8, MXFP6/4) before grinding the schedule.
- **High HBM bytes with low TCC hit** — the working set isn't being reused across the **256 MiB
  Infinity Cache**. Classic bandwidth-bound: tile for reuse and check XCD placement
  (`lever_xcd_locality.md`). L2 is **per-XCD** — a cross-XCD hit is not an L2 hit.
- **The ridge moved right on gfx950** (FP16 ≈ **312 FLOP/byte**, up from ≈247) because the matrix core
  doubled while bandwidth grew less. Kernels that read as borderline compute-bound on MI300X can be
  bandwidth-bound here. Re-classify ports; do not carry the verdict over.
- **Skinny decode GEMV / attention-decode** — almost always bandwidth- or latency-bound. Do not chase
  MFMA occupancy; chase memory access and launch overhead.
- **Many tiny kernels with large gaps between them** — not a kernel problem at all. That is host/launch
  overhead: attack with HIP-graph capture and dispatch collapse (`lever_fusion.md`, launch-bound
  section), not kernel tuning.

## How to drive it

1. `rocprof-compute profile --roof-only` → place the point (`measure_rocpc_workflow.md`).
2. If far from both roofs → full `profile` + `analyze`, read the SoL and memory charts.
3. Read MFMA-busy / TCC / HBM bytes / waves-per-CU against the table above.
4. Apply **one** lever.
5. Re-profile and A/B (`measure_protocol.md`).

One lever at a time. Two simultaneous changes and you cannot attribute the delta — and if they
interact, you cannot even tell the sign of each.

## Verify the verdict was right

After the fix, **the roofline point should move toward a roof** and the targeted counter should change
in the predicted direction — MFMA busy up, or HBM bytes down. Wall time alone is not enough:

| Observation | Meaning |
|---|---|
| Point moved toward a roof, counter moved as predicted | verdict was right, lever worked |
| Wall time down, point and counters unchanged | you moved work elsewhere; re-triage |
| Nothing moved | wrong verdict — go back to the decision flow |

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| "It's slow so it's compute-bound" | slow ≠ compute-bound | place it on the roofline first |
| Occupancy vs latency confused | both sit far from the roofs | only waves/CU + stall% separate them |
| Optimized a kernel worth 2% of runtime | Amdahl | pick targets from the trace's longest bars |
| Verdict flips between runs | measurement noise | `measure_protocol.md` — warm, REPEATS=7, locked clocks |
| BW-bound verdict, HBM counter low | working set fits Infinity Cache — L2/L3-bound, not HBM-bound | check hit rates; `lever_xcd_locality.md` |

## Deeper
`measure_roofline.md` (building the empirical roofs) ·
`measure_rocpc_workflow.md` (running the profiler, reading the tables) ·
`lever_bottleneck_class.md` (the analytic side: AI, ridge, what each class means)
