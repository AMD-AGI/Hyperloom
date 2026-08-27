---
title: CDNA3 / MI300X (gfx942) — clocks, power & thermals
kind: hardware
gens: [gfx942]
dtypes: []
regimes: [both]
updated: 2026-06-08
---

# CDNA3 / MI300X (gfx942) — clocks, power & thermals

> Companion to [peak_tables.md](peak_tables.md) (all peak numbers use the clocks here) and
> [xcd_chiplet.md](xcd_chiplet.md) (per-XCD clock variance).

## TL;DR
> Peak engine clock **2100 MHz**, TDP **750 W** (MI300X) / **1000 W** (MI325X). Peak FLOP math uses
> 2.1 GHz, but **sustained clock is lower under thermal/power limits**, and the **8 XCDs vary 3–10%**
> in clock — so benchmark with median-of-≥3 warm repeats and never assume all CUs run at 2.1 GHz.

## Concepts

### Clock & power table
| Param | MI300X | MI325X |
|---|---|---|
| Arch / ISA | CDNA3 / gfx942 | CDNA3 / gfx942 |
| Peak boost engine clock | **2100 MHz** | **2100 MHz** |
| TDP | **750 W** | **1000 W** |
| Recommended PSU | 1150 W | — |
| Process | TSMC N5 (XCD) + N6 (IOD) | N5 + N6 |
| HBM | 192 GB HBM3, 5.325 TB/s | 256 GB HBM3E, 6.0 TB/s |
| Form factor | OAM | OAM |

The two share the **same compute** (304 CU @ 2.1 GHz, identical peak FLOPS); MI325X differs only in
memory (more/faster HBM3E) and the higher 1000 W envelope.

### Clock dynamics that bite kernels
- **Peak ≠ sustained.** 2.1 GHz is the boost ceiling. Under sustained AI load the engine clock settles
  lower (power/thermal limited), which is one contributor to the **~45% of peak** sustained-FLOP
  reality (arXiv 2510.27583). Always compute achieved TFLOP/s from measured time, not from clock.
- **Per-XCD clock variance 3–10%** (see [xcd_chiplet.md](xcd_chiplet.md)): device-wide-synchronized
  kernels run at the slowest XCD; independent grids are unaffected beyond load balance.
- **DVFS lag.** A short kernel may run before the clock ramps; back-to-back warm repeats and a warmup
  launch give a stable measurement.

### Memory vs engine clock
- HBM bandwidth (5.325 TB/s) is set by the memory data rate (5.2 Gbps × 8192-bit bus), **independent**
  of the engine clock — bandwidth-bound kernels don't benefit from engine-clock headroom, only from
  fewer bytes.

## The levers
1. **Measure achieved TFLOP/s** from wall time; treat 2.1 GHz peak as a ceiling only.
2. **Warm up + median-of-≥3** to absorb DVFS lag and XCD variance.
3. **For compute-bound work**, ensure sustained clock isn't power-capped (check `amd-smi metric`); the
   1000 W MI325X holds clock better under heavy load.
4. **For bandwidth-bound work**, engine clock is irrelevant — cut bytes.
5. **CPX/NPS4** can raise effective clocks by localizing power/thermal per XCD.

## Pitfalls
- **Using 2.1 GHz in efficiency claims** — overstates utilization; use measured clock or measured time.
- **Cold-launch timing** — captures pre-ramp clock.
- **Assuming uniform clock across XCDs** — 3–10% spread.

## Verify
- `amd-smi metric --gpu <id>` (sclk, mclk, power, temperature, throttle status) during the kernel.
- `rocprof-compute` reports achieved vs theoretical at the measured clock.
