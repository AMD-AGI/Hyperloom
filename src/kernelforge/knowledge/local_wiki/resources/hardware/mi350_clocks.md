---
title: MI350X vs MI355X — clocks, power, and what that does to measurements
kind: hardware
topic: clocks
gens: [gfx950]
updated: 2026-08-28
---

# Clocks, power and measurement hygiene

The two SKUs have **identical compute** and differ in cooling and power envelope. That difference does
not show up in the peak tables — it shows up in **sustained clock**, and therefore in every number you
measure.

## The two SKUs

| Param | MI350X | MI355X |
|---|---|---|
| Arch / ISA | CDNA4 / gfx950 | CDNA4 / gfx950 |
| Cooling | **air** | **liquid** |
| TDP | **1000 W** | **1400 W** |
| Peak engine clock | ~2.2–2.4 GHz | up to **~2400 MHz** |
| Compute | 256 CU, identical per-CU matrix core | same |
| HBM | 288 GB HBM3E, 8 TB/s | same |
| Process | TSMC N3P (XCD) + N6 (IOD), 185 B transistors | same |
| Rack density | up to 10U (air) | 5U (liquid) |

Same peak-FLOP tables at a given clock. **MI355X's higher power and cooling sustain higher clocks under
heavy AI load**, so it realizes more throughput on compute-bound work.

## What bites kernels

- **Peak ≠ sustained.** Sustained AI-load clock settles below boost. The 1400 W MI355X envelope keeps
  clock up longer. **Always compute achieved TFLOP/s from wall time, never from an assumed clock.**
- **Per-XCD clock variance ~3–10%.** Device-wide-synchronized kernels run at the slowest XCD;
  independent grids are unaffected beyond load balance (`mi350_chiplet.md`).
- **HBM bandwidth (8 TB/s) is set by the memory data rate**, independent of engine clock —
  bandwidth-bound kernels gain nothing from clock headroom, only from moving fewer bytes.
- **2× matrix throughput per CU vs CDNA3** makes compute-bound GEMM *more* sensitive to throttling. On
  the 1000 W MI350X specifically, watch for power-capped clock under sustained FP8/FP16.
- **DVFS ramp lag** — a short kernel can finish before the clock ramps. This is what warmup hides.

## Consequences for measurement

| Rule | Why |
|---|---|
| **Warm up, discard cold runs** | DVFS ramp + cache fill + JIT/autotune resolution |
| **Median of ≥3 warm repeats**, report the spread | XCD clock variance shows up as spread |
| **Lock or at least monitor clocks** | otherwise DVFS drift masquerades as a speedup |
| **Same-session, non-overlapping A/B** | never compare numbers from two sessions/days/boxes |
| **Reject runs where clock drifted** between ref and candidate | that A/B is invalid |
| **Never compare MI350X and MI355X by peak tables** | identical at equal clock; the difference is *sustained* clock |

The full measurement discipline (REPEATS=7, the ~0.5% noise band, 2-launch A/B) lives in
`common_methodology/profiling/measure_protocol.md`. This card is the hardware reason it exists.

## What it means for kernels

1. **Measure achieved FLOP/s from time**; treat peak clock as a ceiling, not an input.
2. **Warm up and take a median**, for DVFS lag and XCD variance.
3. **Prefer MI355X (1400 W)** for sustained compute-bound throughput; MI350X (1000 W) for air-cooled
   density.
4. **For bandwidth-bound work, cut bytes** — clock is irrelevant there.
5. **CPX/NPS partitioning** can localize power and thermals per XCD for many-small-job density
   (`mi350_chiplet.md`).

## Pitfalls
- **Using peak clock in an efficiency claim** — overstates utilization.
- **Comparing the two SKUs by peak tables** — they are identical at equal clock.
- **Cold-launch timing** — captures pre-ramp clock.
- **Attributing an XCD-variance-sized delta to your change** — 3–10% spread is the machine, not you.

## Verify
- `amd-smi metric --gpu <id>` during the kernel: sclk, mclk, power, temp, **throttle status**.
- `rocprof-compute`: achieved vs theoretical at the *measured* clock.

## Related
`mi350_overview.md` (peaks) · `mi350_chiplet.md` (per-XCD clock variance) ·
`common_methodology/profiling/measure_protocol.md` (the measurement protocol this underpins)
