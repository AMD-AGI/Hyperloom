---
title: CDNA3 / MI300X (gfx942) — peak throughput tables
kind: hardware
gens: [gfx942]
dtypes: [fp64, fp32, tf32, bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, int8]
regimes: [both]
updated: 2026-11-07
---

# CDNA3 / MI300X (gfx942) — peak throughput tables

> **All numbers below are theoretical peak.** One published study measured an average of **~45% of
> peak** on square GEMMs across FP8/BF16/FP16 [4]. Quote *achieved*, never these. Numbers in brackets
> point to the Sources at the end.

## TL;DR
> MI300X @ 2.1 GHz, 304 CU (1216 matrix cores) [1]: **FP16/BF16 1307.4 TF**, **FP8/INT8 2614.9
> TF/TOPS**, FP32/FP64-matrix 163.4 TF, TF32 653.7 TF [1]. HBM3 **5.3 TB/s**, 192 GB [1]. FP16 roofline
> ridge ≈ 247 FLOP/B (derived below).

## Per-CU FLOPs/clock (the basis of all peaks)
| Computation | FLOPs/clock/CU [2][3] | MI300X peak (304 CU @ 2.1 GHz) [1][2] |
|---|---|---|
| Vector FP64 | 128 | 81.7 TFLOP/s |
| Matrix FP64 | 256 | 163.4 TFLOP/s |
| Vector FP32 | 256 | 163.4 TFLOP/s |
| Matrix FP32 | 256 | 163.4 TFLOP/s |
| TF32 (MFMA `xf32` [5]) | 1024 | 653.7 TFLOP/s |
| Matrix FP16 / BF16 | 2048 | 1307.4 TFLOP/s |
| Matrix FP8 / INT8 | 4096 | 2614.9 TFLOP/s (TOPS) |

The white paper lists TF32 as a matrix rate [2]; the ROCm microarchitecture page labels the same 1024
FLOPs/clock row "Vector TF32" [3]. With 4:2 sparsity, FP16/BF16/FP8/INT8 double [1].

Peak formula: `FLOPs_per_clock_per_CU × 304 × 2.1e9`. Example FP16: `2048 × 304 × 2.1e9 ≈ 1307 TF`. ✓

> Vector FP32 and matrix FP32/FP64 are all ~163 TF → **8×** by dropping to FP16/BF16 MFMA, **16×** at
> FP8 (2048/256 and 4096/256; the white paper states the 16× [2]). For inference, push to the lowest
> viable precision.

## Memory & cache peaks
| Resource | Value |
|---|---|
| HBM3 capacity | 192 GB (8 × 24 GB) [2] |
| HBM3 bandwidth | **5.3 TB/s** [1] (8192-bit × 5.2 GT/s / 8 ≈ 5.32 TB/s, derived from [1]) |
| Infinity Cache (L3/MALL) | 256 MiB, 17.2 TB/s peak [2] |
| L2 | 4 MiB **per XCD** [2] |
| L1 vector | 32 KiB/CU, 128 B line [2] |
| LDS | 64 KiB/CU, 32 banks [5], 128 B/clk [6] |

FP16 ridge = 1307.4 TF / 5.3 TB/s ≈ 247 FLOP/B (derived).

## MFMA peak cross-check (per instruction)
`peak = 2·M·N·K · 1216 cores · (2.1e9 / cycles)`, with cycles from the ISA's MFMA table [5]:
- FP16 `32x32x8` (32 cyc): `2·32·32·8·1216·(2.1e9/32) ≈ 1307 TF` ✓
- FP16 `16x16x16` (16 cyc): same 1307 TF ✓ (choose by register footprint, not peak)
- FP8 `16x16x32` (16 cyc): `2·16·16·32·1216·(2.1e9/16) ≈ 2615 TF` ✓

## Sustained reality (use these for expectations)
| Precision | Peak | Typical sustained | Source |
|---|---|---|---|
| FP16/BF16 GEMM | 1307 TF | ~45% (≈590 TF) | [4], average across FP8/BF16/FP16 |
| FP8 GEMM | 2615 TF | ~45% (≈1180 TF) | [4], average across FP8/BF16/FP16 |

[4] measured square GEMMs (M=N=K) with its ROCm stack and recorded GEMM clocks of 1083–1217 MHz.
Sustained depends on shape, library, ROCm version, clock throttling, and XCD balance. Always
benchmark; record `value @ MI300X gfx942, ROCm <ver>, <lib>@<ver>, <date>`.

## MI325X delta
Same compute/peak FLOPS (304 CU @ 2.1 GHz) [2][7]; memory is **256 GB HBM3E @ 6.0 TB/s** (vs 192 GB /
5.3 TB/s) [2], board power 1000 W [7]. FP16 ridge shifts to ≈ 1307.4 / 6.0 ≈ 218 FLOP/B (derived).

## Sources
1. AMD, *AMD Instinct MI300X Accelerator* data sheet — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
2. AMD, *AMD CDNA 3 Architecture* white paper (Table 1; memory section) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
3. ROCm documentation, *AMD Instinct MI300 Series microarchitecture* (peak-performance table) — https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300.html
4. C. Ambati, T. Diep, *AMD MI300X GPU Performance Analysis*, arXiv:2510.27583 — https://arxiv.org/abs/2510.27583
5. AMD, *"AMD Instinct MI300" Instruction Set Architecture* (CDNA3 ISA; §2.2.1 LDS, §7.1 MFMA tables) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
6. HIP documentation, *Hardware implementation* (LDS bandwidth) — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
7. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* (architecture tables) — https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html
