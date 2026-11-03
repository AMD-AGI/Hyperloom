---
title: CDNA3 / MI300X (gfx942) — peak throughput tables
kind: hardware
gens: [gfx942]
dtypes: [fp64, fp32, tf32, bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, int8]
regimes: [both]
updated: 2026-06-08
---

# CDNA3 / MI300X (gfx942) — peak throughput tables

> **All numbers below are theoretical peak.** Real kernels sustain **~45% of peak** across
> FP8/BF16/FP16 (arXiv 2510.27583). Quote *achieved*, never these.

## TL;DR
> MI300X @ 2.1 GHz, 304 CU (1216 matrix cores): **FP16/BF16 1307 TF**, **FP8/INT8 2615 TF/TOPS**,
> FP32/FP64-matrix 163 TF, TF32 654 TF. HBM3 **5.325 TB/s**, 192 GB. FP16 roofline ridge ≈ 247 FLOP/B.

## Per-CU FLOPs/clock (the basis of all peaks)
| Computation | FLOPs/clock/CU | MI300X peak (304 CU @ 2.1 GHz) |
|---|---|---|
| Vector FP64 | 128 | 81.7 TFLOP/s |
| Matrix FP64 | 256 | 163.4 TFLOP/s |
| Vector FP32 | 256 | 163.4 TFLOP/s |
| Matrix FP32 | 256 | 163.4 TFLOP/s |
| Vector TF32 (emulated) | 1024 | 653.7 TFLOP/s |
| Matrix FP16 / BF16 | 2048 | 1307.4 TFLOP/s |
| Matrix FP8 / INT8 | 4096 | 2614.9 TFLOP/s (TOPS) |

Peak formula: `FLOPs_per_clock_per_CU × 304 × 2.1e9`. Example FP16: `2048 × 304 × 2.1e9 ≈ 1307 TF`. ✓

> Vector FP32 and matrix FP32/FP64 are all ~163 TF → **8×** by dropping to FP16/BF16 MFMA, **16×** at
> FP8. For inference, push to the lowest viable precision.

## Memory & cache peaks
| Resource | Value |
|---|---|
| HBM3 capacity | 192 GB (8 × 24 GB) |
| HBM3 bandwidth | **5.325 TB/s** (5.2 Gbps × 8192-bit) |
| Infinity Cache (L3/MALL) | 256 MiB, ~11.9 TB/s measured (~17 theoretical), ~218 ns |
| L2 | 4 MiB **per XCD** |
| L1 vector | 32 KiB/CU, 128 B line |
| LDS | 64 KiB/CU, 32 banks, up to 128 B/clk |

## MFMA peak cross-check (per instruction)
`peak = 2·M·N·K · 1216 cores · (2.1e9 / cycles)`:
- FP16 `32x32x8` (32 cyc): `2·32·32·8·1216·(2.1e9/32) ≈ 1307 TF` ✓
- FP16 `16x16x16` (16 cyc): same 1307 TF ✓ (choose by register footprint, not peak)
- FP8 `16x16x32` (16 cyc): `2·16·16·32·1216·(2.1e9/16) ≈ 2615 TF` ✓

## Sustained reality (use these for expectations)
| Precision | Peak | Typical sustained | Source |
|---|---|---|---|
| FP16/BF16 GEMM | 1307 TF | ~45% (≈590 TF) | arXiv 2510.27583 |
| FP8 GEMM | 2615 TF | ~45% (≈1180 TF) | arXiv 2510.27583 |

Sustained depends on shape, library, ROCm version, clock throttling, and XCD balance. Always
benchmark; record `value @ MI300X gfx942, ROCm <ver>, <lib>@<ver>, <date>`.

## MI325X delta
Same compute/peak-FLOPS (304 CU @ 2.1 GHz); memory is **256 GB HBM3E @ 6.0 TB/s** (vs 192 GB / 5.3 TB/s),
TDP 1000 W. FP16 ridge shifts to ≈ 218 FLOP/B.
