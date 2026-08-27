---
title: CDNA4 / MI350X / MI355X (gfx950) — peak throughput tables
kind: hardware
gens: [gfx950]
dtypes: [fp64, fp32, bf16, fp16, fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1, mxfp8, mxfp6, mxfp4, int8]
regimes: [both]
updated: 2026-06-09
---

# CDNA4 / MI350X / MI355X (gfx950) — peak throughput tables

> **All numbers are theoretical peak.** Sustained is well below (MI300X-class kernels see ~45%; expect
> similar fractions on CDNA4 pending broad measurement, arXiv 2510.27583). Quote *achieved*.

## TL;DR
> MI355X (256 CU, 1024 matrix cores): **FP16/BF16 2.5 PF**, **FP8 5 PF**, **FP6/FP4 10 PF**,
> FP32 157 TF, FP64 vector 78.6 TF. HBM3E **288 GB / 8 TB/s**. **2× FP16/FP8 vs CDNA3**; **TF32
> removed**; FP64 **matrix** halved.

## Matrix / vector peaks (per OAM, vendor-reported)
| Computation | MI355X peak | vs FP32 | vs CDNA3 (MI325X) |
|---|---|---|---|
| FP64 vector | 78.6 TFLOP/s | 0.5× | ~same vector / matrix halved |
| FP32 matrix | 157.3 TFLOP/s | 1× | ~same |
| FP16 / BF16 matrix | **2.5 PFLOP/s** | 16× | **2×** (1307→2500) |
| FP8 (OCP) matrix | **5 PFLOP/s** | 32× | **2×** (2615→5000) |
| FP6 matrix | **10 PFLOP/s** | 64× | new |
| FP4 matrix | **10 PFLOP/s** | 64× | new |
| MXFP8/6/4 | matches element rate | — | new |
| INT8 | ~5 POPS | 32× | 2× |
| TF32 | **removed** | — | gone |

FP6 and FP4 share the **10 PF** rate (FP6 effectively at FP4 rate). MI350X has the same per-CU peaks;
realized throughput differs by sustained clock (1000 W vs 1400 W) — see
[clocks_power.md](clocks_power.md).

## Memory & cache peaks
| Resource | Value |
|---|---|
| HBM3E capacity | **288 GB** (8 × 36 GB, 12-Hi) |
| HBM3E bandwidth | **8.0 TB/s** |
| Infinity Cache (L3/MALL) | 256 MiB |
| L2 | per-XCD |
| LDS | **160 KiB/CU**, **64 banks**, **256 B/clk** |
| L1 vector | 32 KiB/CU, 128 B line |
| Infinity Fabric (per card) | 1075 GB/s bidirectional aggregate (4th gen) |

FP16 roofline ridge ≈ 2.5 PF / 8 TB/s ≈ **312 FLOP/byte** (vs ~247 on MI300X) → relatively more
kernels are bandwidth-bound; byte-cutting matters even more.

## Peak formula cross-check
`peak = 2·M·N·K · 1024 cores · (clock / cycles)`. With 256 CU × 4 = 1024 cores and the **2× per-CU**
matrix rate, the FP16/FP8/FP6/FP4 peaks above follow from the CDNA4 cycle counts in
[matrix_core_blockscale.md](matrix_core_blockscale.md) (FP6/FP4 use the lower cycle count when neither
operand is FP8 → 2× the FP8 rate → 10 PF).

## Sustained reality
No large public sustained-of-peak dataset for CDNA4 yet (as of 2026-06). On MI300X the sustained
fraction is **~45%** across FP8/BF16/FP16 (arXiv 2510.27583); treat CDNA4 similarly until measured.
Record any measured number as `value @ MI355X gfx950, ROCm <ver>, <lib>@<ver>, <date>`.
