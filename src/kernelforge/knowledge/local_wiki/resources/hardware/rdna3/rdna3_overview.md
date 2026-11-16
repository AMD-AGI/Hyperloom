---
title: Radeon PRO W7900 — overview, peaks and porting deltas
kind: hardware
topic: overview
gens: [gfx1100]
updated: 2026-11-16
---

# Overview, peaks and porting deltas

One-screen cheat sheet for the W7900. Read this first; the other cards in `hardware/rdna3/` hold the detail.

## The chip

| | W7900 (gfx1100) | Source |
|---|---|---|
| Architecture | RDNA 3, Navi 31 (5 nm GCD + 6 nm MCDs) | AMD product page |
| Compute | 96 CU = 48 WGP; 4 SIMD32 per WGP; 6144 stream processors | AMD product page, ISA §2.3 |
| Memory | 48 GB GDDR6 ECC, 384-bit, **864 GB/s** | AMD product page |
| Caches | L0 32 KiB/CU, L1 256 KiB per shader array, L2 6 MiB, Infinity Cache 96 MiB | ROCm GPU specs |
| Board power | 295 W | AMD product page |

## Peaks (vendor, dense)

| Precision | Peak | Path |
|---|---|---|
| FP16 / BF16 | **122.6 TFLOPS** | WMMA (BF16 runs at the FP16 rate) |
| FP32 | **61.3 TFLOPS** | VALU with dual issue |
| INT8 | 122.6 TOPS | WMMA (iu8) |
| INT4 | 245 TOPS | WMMA (iu4) |
| FP8 / FP4 | — | not supported on gfx11 |

**W7900D:** the same 96 CUs, 48 GB and 864 GB/s at a lower engine clock (rocminfo reports a 1760 MHz maximum).
AMD publishes no peaks for it; scale the W7900's compute peaks by the clock ratio as an estimate, and treat any
compute-bound conclusion on a W7900D as needing a measured ceiling. Its bandwidth ridge is lower in proportion.

AMD does not publish W7900 clocks. Third-party databases list a ~2.5 GHz boost, which is consistent with the
FP32 figure (6144 × 2 FMA × 2 dual-issue × 2.495 GHz ≈ 61.3 TFLOPS). Do not quote a clock as a fact; measure it.

## Roofline ridges (peak FLOP/s ÷ 864 GB/s)

| | Ridge | Meaning |
|---|---|---|
| FP16 WMMA | ≈ **142 FLOP/byte** | below this a kernel is bandwidth-bound |
| FP32 VALU | ≈ **71 FLOP/byte** | |

MI355X's FP16 ridge is ≈ 312 FLOP/byte. A GEMM shape that is compute-bound on Instinct can be memory-bound here;
re-derive the bound for this chip before choosing a direction. The 96 MiB Infinity Cache can lift the effective
bandwidth for working sets that fit in it, so a measured bandwidth above 864 GB/s is not by itself a bug.

## Porting deltas from CDNA (MI300X / MI355X)

| CDNA habit | On gfx1100 |
|---|---|
| wave64 everywhere | wave32 by default for HIP; wave64 is a build option that issues VALU/VMEM twice |
| MFMA (`v_mfma_*`), many shapes | WMMA (`v_wmma_*`), 16×16×16 only, A/B replicated across half-waves |
| AGPR accumulators, `v_accvgpr_*` | no AGPRs; accumulators are VGPRs |
| 512 VGPR+AGPR per lane budget | ≤256 VGPRs per wave; 1536 per SIMD in wave32, 24-register granule |
| LDS 64 KiB (MI300X) / 160 KiB (MI355X) per CU | 128 KiB per WGP, **64 KiB max per work-group** |
| HBM, multi-TB/s | GDDR6, 864 GB/s, 96 MiB Infinity Cache |
| FP8 (FNUZ / OCP), FP4, sparsity | none of them |
| XCD-aware tile swizzle | one die; no XCD locality to manage |
| rocprof-compute reports, wait/MFMA ratio | rocprofv3 counters only (`rdna3_measure.md`) |
