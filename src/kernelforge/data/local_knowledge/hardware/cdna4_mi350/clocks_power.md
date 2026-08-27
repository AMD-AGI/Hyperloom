---
title: CDNA4 / MI350X / MI355X (gfx950) — clocks, power & thermals
kind: hardware
gens: [gfx950]
dtypes: []
regimes: [both]
updated: 2026-06-08
---

# CDNA4 / MI350X / MI355X (gfx950) — clocks, power & thermals

> Peak math uses the clocks here; matrix peaks in [peak_tables.md](peak_tables.md). Per-XCD clock
> variance behaves like CDNA3 — see [../cdna3_mi300/xcd_chiplet.md](../cdna3_mi300/xcd_chiplet.md).

## TL;DR
> Two SKUs differ mainly in cooling/power: **MI350X air-cooled, 1000 W**; **MI355X liquid-cooled,
> 1400 W**, the higher envelope holding higher sustained clocks (MI355X up to ~2400 MHz). N3P process,
> 185 B transistors. As always, **sustained < peak** — compute achieved FLOP/s from measured time.

## Concepts

### Clock & power table
| Param | MI350X | MI355X |
|---|---|---|
| Arch / ISA | CDNA4 / gfx950 | CDNA4 / gfx950 |
| Cooling | air | liquid |
| TDP | **1000 W** | **1400 W** |
| Peak engine clock | ~2.2–2.4 GHz | up to **~2400 MHz** |
| Process | TSMC **N3P** (XCD) + N6 (IOD) | N3P + N6 |
| Transistors | 185 B | 185 B |
| HBM | 288 GB HBM3E, 8 TB/s | 288 GB HBM3E, 8 TB/s |
| Rack density | up to 10U (air) | 5U (liquid) |

Both have the **same compute** (256 CU, identical per-CU matrix core); MI355X's higher power/cooling
sustains higher clocks under heavy AI load → higher realized throughput than MI350X for compute-bound
work, despite identical peak-FLOP tables at a given clock.

### Clock dynamics that bite kernels
- **Peak ≠ sustained.** Like CDNA3, sustained AI-load clock settles below boost; the higher 1400 W
  MI355X envelope keeps clock up longer. Always compute achieved TFLOP/s from wall time.
- **Per-XCD clock variance** (~3–10%, same mechanism as CDNA3): device-wide-synchronized kernels run
  at the slowest XCD; independent grids unaffected beyond load balance.
- **HBM bandwidth (8 TB/s)** is set by the memory data rate, independent of engine clock —
  bandwidth-bound kernels gain nothing from clock headroom, only from fewer bytes.
- **2× matrix throughput/CU** vs CDNA3 means compute-bound GEMM is more sensitive to clock throttling;
  on the 1000 W MI350X watch for power-capped clock under sustained FP8/FP16.

## The levers
1. **Measure achieved FLOP/s** from time; treat peak clock as a ceiling.
2. **Warm up + median-of-≥3** for DVFS lag and XCD variance.
3. **Prefer MI355X (1400 W)** for sustained compute-bound throughput; MI350X (1000 W) for air-cooled
   density.
4. **For bandwidth-bound work**, cut bytes — clock is irrelevant.
5. **CPX/NPS** can localize power/thermal per XCD for many-small-job density.

## Pitfalls
- **Using peak clock in efficiency claims** — overstates utilization.
- **Comparing MI350X vs MI355X by peak tables** — they're identical at equal clock; the difference is
  *sustained* clock under the power cap.
- **Cold-launch timing** captures pre-ramp clock.

## Verify
- `amd-smi metric --gpu <id>` (sclk/mclk/power/temp/throttle) during the kernel.
- `rocprof-compute` achieved vs theoretical at the measured clock.
