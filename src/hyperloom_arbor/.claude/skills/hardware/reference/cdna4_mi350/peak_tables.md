---
title: CDNA4 / MI350X / MI355X (gfx950) — peak throughput tables
kind: hardware
gens: [gfx950]
dtypes: [fp64, fp32, bf16, fp16, fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1, mxfp8, mxfp6, mxfp4, int8]
regimes: [both]
updated: 2026-06-09
---

# CDNA4 / MI350X / MI355X (gfx950) — peak throughput tables

> **All numbers are peak theoretical** (AMD's term [1]). AMD publishes no sustained-of-peak figure for
> CDNA4; on MI300X, arXiv 2510.27583 measured about 45% of peak on average across FP8, BF16 and FP16 [6].
> Quote *achieved*. Numbers in brackets are the sources at the end.

## TL;DR
> MI355X (256 CU, 1,024 matrix cores) [1]: **FP16/BF16 2.5 PF**, **FP8 5 PF**, **MXFP6/MXFP4 10 PF**,
> FP32 157.3 TF, FP64 vector 78.6 TF [1][2]. HBM3E **288 GB / 8 TB/s** [1]. **1.9× FP16/FP8 vs CDNA3**
> (2× per CU per clock) [1]; **TF32 removed**; FP64 **matrix** halved [1].

## Matrix / vector peaks (per OAM, vendor-reported)
| Computation | MI355X peak [1][2] | vs FP32 (derived) | vs CDNA3 (MI300X/MI325X) [1][4] |
|---|---|---|---|
| FP64 vector | 78.6 TFLOP/s | 0.5× | 81.7 → 78.6 TF (~0.96×) |
| FP64 matrix | 78.6 TFLOP/s | 0.5× | 163.4 → 78.6 TF (~0.5×) |
| FP32 matrix | 157.3 TFLOP/s | 1× | 163.4 → 157.3 TF (~0.96×) |
| FP16 / BF16 matrix | **2.5 PFLOP/s** | 16× | **1.9×** (1.3 → 2.5 PF) |
| FP8 (OCP) matrix | **5 PFLOP/s** | 32× | **1.9×** (2.6 → 5.0 PF) |
| MXFP8 | **5 PFLOP/s** | 32× | new |
| MXFP6 | **10 PFLOP/s** | 64× | new |
| MXFP4 | **10 PFLOP/s** | 64× | new |
| INT8 | 5 POPS | 32× | 1.9× (2.6 → 5.0 POPS) |
| TF32 | **removed** (software emulation via BF16) | — | 653.7 TF on CDNA3 [4] |

MXFP6 and MXFP4 share the **10 PF** rate [1]. The unscaled f8f6f4 instructions have the same cycle counts
as the scaled ones [3], so plain FP6/FP4 run at that rate too (derived). Structured-sparsity peaks are 2×
these for FP16, BF16, FP8 and INT8 [1][2]. MI350X has the same per-CU, per-clock rates; its peaks are lower
by its 2,200 MHz peak clock: FP16/BF16 2.3 PF, FP8 4.6 PF, MXFP6/MXFP4 9.2 PF, FP32 144.2 TF, FP64 72.1 TF
[1][2]. See [clocks_power.md](clocks_power.md).

## Memory & cache peaks
| Resource | Value |
|---|---|
| HBM3E capacity | **288 GB** (8 × 36 GB, 12-Hi) [1] |
| HBM3E bandwidth | **8.0 TB/s** [1] |
| Infinity Cache (memory-side) | 256 MB [1] |
| L2 | 4 MB per XCD, 16-way [1] |
| LDS | **160 KiB/CU**, **64 banks** [3], **256 B/clk** [1] |
| L1 vector | 32 KiB/CU, 128 B line [1] |
| Infinity Fabric (per card) | 7 links × 144 GB/s = 1,008 GB/s peer-to-peer [1][5]; the white paper's Figure 5 and [4] give 1,075 GB/s (38.4 Gbps links), [2] 36 Gbps links |

FP16 roofline ridge ≈ 2.5 PF / 8 TB/s ≈ **312 FLOP/byte** (vs 1307.4 TF [7] / 5.3 TB/s [4] ≈ 247 on MI300X)
→ relatively more kernels are bandwidth-bound; byte-cutting matters even more (derived).

## Peak formula cross-check
`peak = 2·M·N·K · matrix cores · (clock / cycles)` [7]. With 1,024 matrix cores at 2,400 MHz [1] and the
CDNA4 cycle counts [3] (see [matrix_core_blockscale.md](matrix_core_blockscale.md)):
- FP16 16×16×32 at 16 cycles → 2.52 PF; f8f6f4 16×16×128 at 32 cycles with an FP8 operand → 5.03 PF, at
  16 cycles with FP6/FP4 only → 10.07 PF. These match AMD's peaks [1].
- The classic shapes give less: FP16 16×16×16 (16 cycles) → 1.26 PF, FP8 16×16×32 (16 cycles) → 2.52 PF.
  The peaks need the new shapes (derived).

## Sustained reality
AMD's documents give no sustained-of-peak figure for CDNA4. On MI300X, arXiv 2510.27583 measured about
**45%** of peak on average across FP8, BF16 and FP16 [6]; measure on gfx950 rather than assume it.
Record any measured number as `value @ MI355X gfx950, ROCm <ver>, <lib>@<ver>, <date>`.

## Sources
1. AMD, *Introducing AMD CDNA 4 Architecture* (white paper): Table 1 (FLOPs per clock per CU, MI300X vs
   MI355X peaks), memory, Figure 5, Table 2 (MI350X and MI355X specifications, P2P 1,008 GB/s).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf>
2. ROCm documentation, *AMD Instinct MI350 Series microarchitecture* (peak theoretical performance;
   Infinity Fabric links at 36 Gbps).
   <https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html>
3. AMD, *CDNA4 Instruction Set Architecture Reference Guide*, 5 Aug 2025: §2.2.1 (LDS banks), §7.1.2
   Table 28 (MFMA cycles).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf>
4. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* (MI300X, MI325X,
   MI350X and MI355X comparison tables).
   <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html>
5. AMD, *AMD Instinct MI355X GPU* datasheet (7 × 144 GB/s Infinity Fabric links).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf>
6. C. Ambati, T. Diep, *AMD MI300X GPU Performance Analysis*, arXiv:2510.27583.
   <https://arxiv.org/abs/2510.27583>
7. ROCm Blogs, *Matrix Core Programming on AMD CDNA3 and CDNA4 architecture* (peak formula; 1307.4 TF
   FP16 on MI325X, the same as MI300X).
   <https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html>
