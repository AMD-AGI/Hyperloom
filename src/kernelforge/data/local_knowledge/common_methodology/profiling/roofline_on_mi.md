---
title: profiling — building and reading a roofline on MI300X / MI350X
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1, fp6, int8]
updated: 2026-06-08
---

# Roofline on MI300X / MI350X

## TL;DR
The roofline plots **achieved FLOP/s vs arithmetic intensity (FLOP/byte)**. Two ceilings: a sloped
**BW roof** (HBM/L2/Infinity-Cache lines) and a flat **compute roof** (per-dtype peak FLOP/s). A
kernel left of the **ridge point** is BW-bound; right of it, compute-bound. On Instinct, build it
**empirically** with `rocprof-compute --roof-only`
([`rocprof_compute_workflow.md`](rocprof_compute_workflow.md)) — the tool runs microbenchmarks to
measure the *real* peaks of your box, not datasheet numbers.

## Build it
```bash
rocprof-compute profile --name myrun --roof-only -- python bench.py
# → workloads/myrun/MI300X/{roofline.csv, empirRoof_gpu-0_FP16.pdf, ...}
rocprof-compute analyze -p workloads/myrun/MI300X/ --roofline-data-type FP16
```
`--roof-only` collects only roofline counters and runs on-device microbenchmarks to get **empirical**
roofs (saved in `roofline.csv`), then emits one PDF per dtype. Overlay dtypes with `--device` and label
kernels with `--kernel-names`.

> **In the forge-loop, don't run this by hand** — use the script
> `python3 rocpc_profile.py --driver <driver> --roofline` (see
> [`rocprof_compute_workflow.md`](rocprof_compute_workflow.md)); it handles the dependency gate,
> isolates your kernel, and prints AI + distance-to-roof for you.

## The roofs to draw (theoretical anchors)
Datasheet peaks for context — but the *empirical* roof is what `--roof-only` measures and is what you
compare against:

**MI300X (gfx942), 304 CU @ 2.1 GHz** — [`../../hardware/cdna3_mi300/peak_tables.md`](../../hardware/cdna3_mi300/peak_tables.md):
| dtype | compute roof (peak) | ridge (peak ÷ HBM) |
|---|---|---|
| FP16 / BF16 matrix | 1307 TF | ≈ 247 FLOP/byte |
| FP8 / INT8 matrix | 2615 TF / TOPS | ≈ 491 FLOP/byte |
| FP32 (vec/matrix) | 163 TF | ≈ 31 FLOP/byte |
- HBM3 **5.325 TB/s peak**; **achievable is lower (~4.3 TB/s class)** in practice — the empirical BW
  roof from `--roof-only` will sit below the 5.325 line, which is the honest ceiling to measure against.
- Infinity Cache (256 MiB L3) ~11.9 TB/s measured, L2 4 MiB **per XCD** — a kernel that fits and reuses
  in L2/L3 rides a *higher* BW roof than HBM.

**MI355X / MI350X (gfx950), 256 CU / 1024 matrix cores** — [`../../hardware/cdna4_mi350/peak_tables.md`](../../hardware/cdna4_mi350/peak_tables.md):
| dtype | compute roof (peak) | ridge (peak ÷ HBM) |
|---|---|---|
| FP16 / BF16 | 2.5 PF | ≈ 312 FLOP/byte |
| FP8 | 5 PF | ≈ 625 FLOP/byte |
| FP6 / FP4 | 10 PF | ≈ 1250 FLOP/byte |
- HBM3E **8.0 TB/s**, 288 GB. **TF32 removed**. Higher ridge than MI300X → *more* kernels land
  BW-bound, so cutting bytes (lower-precision, fusion, L2 reuse) pays even more.

## Reading a point
- **On the sloped roof** → BW-bound. Raise arithmetic intensity (fuse epilogues, larger BLOCK_K, reuse
  in L2/Infinity Cache). Going to a smaller dtype moves you to a *different* roof and shifts the ridge.
- **On the flat roof** → compute-bound. You are near the dtype peak; only a lower-precision path or a
  better MFMA-shaped kernel helps. Remember MI300X GEMM sustains only ~45–55% of the flat roof
  (software ceiling) — a point at ~50% of peak FP16 may already match the best library kernel.
- **Under both roofs** → occupancy- or latency-bound; counters disambiguate
  ([`reading_a_kernel_bottleneck.md`](reading_a_kernel_bottleneck.md)).
- **Improvement = point moves up/right toward a roof**, not merely lower wall time.

## Per-dtype caution
Use the matching `--roofline-data-type`: an FP8 GEMM compared against the FP32 roof (the tool default)
looks artificially terrible. Pick the roof for the kernel's actual MFMA dtype.

## Pitfalls
- Quoting the **datasheet** roof as achievable — always compare to the *empirical* `roofline.csv` roof
  and remember HBM achievable (~4.3 TB/s) < peak (5.325 TB/s).
- Drawing the FP32 roof for a low-precision kernel.
- Forgetting cache roofs: a "BW-bound" verdict against the HBM roof may actually be L2/L3-resident.

## Verify
`roofline.csv` exists and the empirical compute roof is within a sane fraction (not above) the datasheet
peak; the kernel marker sits where its measured AI predicts.
