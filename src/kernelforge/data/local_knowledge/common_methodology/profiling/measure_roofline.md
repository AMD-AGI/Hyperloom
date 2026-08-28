---
title: roofline — building the empirical roofs and placing a kernel on them
kind: measure
measure: roofline
gens: [gfx950]
updated: 2026-08-28
---

# Roofline

Places a kernel against two ceilings so the bottleneck stops being a guess. The output feeds
`measure_triage.md`, which turns the position into a lever.

## Route here when
- Establishing what a kernel *could* achieve before deciding whether to work on it.
- You need to know which roof it sits under (the input to every other decision).
- Reporting efficiency — the roofline gives you an honest denominator.

## The model in three lines

- **Sloped BW roof**: `achievable = AI × bandwidth`, one line per memory level (HBM, Infinity Cache, L2).
- **Flat compute roof**: per-dtype peak FLOP/s.
- **Ridge point** = where they cross. Left of it → bandwidth-bound. Right → compute-bound.

`AI` (arithmetic intensity) = FLOPs ÷ bytes moved.

## Build it empirically — not from the datasheet

```bash
rocprof-compute profile --name myrun --roof-only -- python bench.py
# → workloads/myrun/MI350X/{roofline.csv, empirRoof_gpu-0_FP16.pdf, ...}
rocprof-compute analyze -p workloads/myrun/MI350X/ --roofline-data-type FP16
```

`--roof-only` collects roofline counters **and runs on-device microbenchmarks** to measure your box's
real peaks into `roofline.csv`, then emits one PDF per dtype. Overlay dtypes with `--device`, label
kernels with `--kernel-names`.

> **In the forge loop, do not run this by hand.** Use
> `python3 rocpc_profile.py --driver <driver> --roofline` — it handles the dependency gate, isolates
> your kernel, and prints AI plus distance-to-roof directly (`measure_rocpc_workflow.md`).

## gfx950 anchors (context only — compare against the empirical roof)

MI350X / MI355X, 256 CU, 1024 matrix cores, HBM3E **288 GB @ 8.0 TB/s**:

| dtype | compute roof | ridge (peak ÷ 8 TB/s) |
|---|---|---|
| FP16 / BF16 | 2.5 PF | ≈ **312 FLOP/byte** |
| FP8 (OCP) | 5 PF | ≈ 625 FLOP/byte |
| FP6 / FP4 | 10 PF | ≈ 1250 FLOP/byte |
| FP32 | 157 TF | ≈ 20 FLOP/byte |

**TF32 is removed** on gfx950 — there is no roof to draw for it.

The ridge is **higher than the previous generation** (≈312 vs ≈247 FP16) because the matrix core
doubled while bandwidth grew less. Practical reading: *more* kernels land bandwidth-bound here, so
cutting bytes — lower precision, fusion, L2 reuse — pays more than it used to.

## Reading a point

| Where it sits | Verdict | Next |
|---|---|---|
| On the **sloped** roof | bandwidth-bound | raise AI: fuse epilogues, larger `BLOCK_K`, reuse in L2 / Infinity Cache |
| On the **flat** roof | compute-bound | only a lower-precision path or a better MFMA schedule helps |
| **Under both** | occupancy- or latency-bound | counters disambiguate → `measure_triage.md` |

Two things that trip people up:

- **Changing dtype moves you to a different roof** *and* shifts the ridge. A BF16→FP8 conversion halves
  the bytes and doubles the peak — the point moves diagonally, and it may change class.
- **~45–55% of the flat roof is the practical ceiling** for tuned GEMM. A point at ~50% of peak FP16
  may already match the best library kernel; the remaining gap is a software-maturity ceiling, not
  headroom you can grind out.

**Improvement means the point moves up or right toward a roof** — not merely lower wall time. A change
that lowers wall time without moving the point usually relocated work rather than removing a
bottleneck.

## Per-dtype caution

Use the matching `--roofline-data-type`. An FP8 GEMM compared against the FP32 roof (the tool's
default) looks artificially catastrophic. Pick the roof for the kernel's **actual MFMA dtype**.

## Verify

| Check | Pass |
|---|---|
| `roofline.csv` exists | the empirical run completed |
| Empirical compute roof vs datasheet peak | at or **below** peak, sane fraction — above means a bad run |
| Kernel marker position | matches where its measured AI predicts |
| Achievable HBM BW | **below** 8.0 TB/s — if the tool reports at or above, distrust the run |

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Kernel looks hopeless | drawn against the FP32 roof | set `--roofline-data-type` to the real dtype |
| "We're at 40% of peak, something's broken" | datasheet peak used as the bar | compare against the empirical roof and the best library kernel |
| BW-bound verdict, HBM counter low | working set is L2 / Infinity-Cache resident | check cache roofs, not just the HBM line |
| Point didn't move after a fix | wrong bottleneck, or the change wasn't live | `measure_triage.md`; confirm the edit is live |
| Roofs differ run to run | cold clocks / unlocked DVFS | `measure_protocol.md` |

## Deeper
`hardware/mi350_overview.md` (peaks and ridges) ·
`hardware/mi350_memory.md` (the bandwidth ladder and cache roofs) ·
`measure_rocpc_workflow.md` (running it) · `measure_triage.md` (acting on it) ·
`lever_bottleneck_class.md` (the analytic companion)
