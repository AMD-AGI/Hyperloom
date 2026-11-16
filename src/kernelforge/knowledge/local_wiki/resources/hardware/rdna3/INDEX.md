---
title: Radeon PRO W7900 (RDNA 3, gfx1100) hardware — knowledge map
kind: index
scope: hardware
gens: [gfx1100]
updated: 2026-11-16
---

# Radeon PRO W7900 (RDNA 3, gfx1100) hardware — knowledge map

Entry index for `hardware/rdna3/`. Backend-neutral facts about the chip a gfx1100 campaign runs on: what it is,
what the numbers are, and what each subsystem does to a kernel.

> **Convention (KernelForge standard):** a knowledge folder that contains an `INDEX.md` is navigated
> **through this file** — load it whole.

## Scope: gfx1100 only

Target parts are the **Radeon PRO W7900** and **W7900D** (Navi 31, RDNA 3, ISA **gfx1100**). The W7900D is the
same silicon at a lower boost clock, so every count and size here holds for it; only its peaks are lower
(`rdna3_overview.md`). The RX 7900 XTX is the same die;
other Navi 3x chips (gfx1101, gfx1102) have a smaller register file and fewer CUs, so their occupancy numbers
differ. The MI350 cards in the parent `hardware/` folder are **wrong for this chip** — open them only for the
contrast, never for a number.

## Layout

| File | Subsystem |
|---|---|
| **`rdna3_overview.md`** | **START HERE** — cheat sheet, peaks, roofline ridges, the CDNA → RDNA 3 porting deltas |
| `rdna3_execution.md` | WGP/CU/SIMD32 hierarchy, wave32 vs wave64, dual-issue VALU (VOPD), VGPR file, occupancy formula |
| `rdna3_wmma.md` | WMMA instructions, operand lane layout, HIP builtins, hazards, throughput vs VALU |
| `rdna3_memory.md` | LDS (128 KiB/WGP, 64 KiB/work-group cap, CU vs WGP mode), cache ladder, GDDR6 bandwidth |
| `rdna3_measure.md` | rocprofv3 on gfx1100, what rocprof-compute cannot do here, clock control, library support |

## Cards elsewhere that cite `hardware/mi350_*.md`

Methodology and language cards were written against the MI350 map and cite its files by name. On gfx1100 read
the RDNA 3 card for the same topic instead:

| Cited MI350 card | Read instead |
|---|---|
| `mi350_overview.md`, `mi350_clocks.md` | `rdna3_overview.md`, `rdna3_measure.md` |
| `mi350_execution.md`, `mi350_isa.md` | `rdna3_execution.md` |
| `mi350_matrix_core.md`, `mi350_dtypes.md` | `rdna3_wmma.md` |
| `mi350_lds.md`, `mi350_memory.md`, `mi350_chiplet.md` | `rdna3_memory.md` |

## Constants you will look up most

| | |
|---|---|
| 96 CU = 48 WGP · 2 CU and 4 SIMD32 per WGP | wave32 default · wave64 issues VALU/VMEM twice |
| 1536 VGPRs/SIMD in wave32 (768 in wave64) · alloc granule 24 (12 in wave64) | ≤256 VGPRs per wave · **no AGPRs** · max 16 waves/SIMD |
| LDS **128 KiB per WGP**, 64 banks × 4 B · **≤64 KiB per work-group** | L0 32 KiB/CU · L1 256 KiB/shader array · L2 6 MiB · Infinity Cache 96 MiB |
| GDDR6 384-bit **864 GB/s** · 48 GB | FP16/BF16 WMMA **122.6 TFLOPS** · FP32 VALU **61.3 TFLOPS** |
| FP16 ridge ≈ **142 FLOP/byte** (HBM-class MI355X: ~312) | WMMA is 16×16×16 only; **no FP8/FP4, no sparsity** |

## Portable golden rules

- **Wave size is a build choice, not a constant.** HIP compiles gfx1100 for wave32 unless `-mwavefrontsize64` is
  passed, and Triton 3.8 does too; check the kernel's own build (`.amdhsa_wavefront_size32`) before doing any
  lane, shuffle, ballot or reduction arithmetic.
- **Every MFMA idea needs translating.** WMMA has one shape (16×16×16), needs its A/B operands replicated across
  half-waves, and accumulates in ordinary VGPRs.
- **Occupancy is a VGPR budget of 1536 per SIMD** (wave32), rounded to 24-register blocks; 96 VGPRs or fewer
  keeps all 16 wave slots.
- **Bandwidth is GDDR6, not HBM.** The ridge is under half of an Instinct part's, so many kernels that are
  compute-bound on MI300X/MI355X are memory-bound here.
- **No hardware-counter dashboard.** rocprof-compute does not support gfx1100; use rocprofv3 directly
  (`rdna3_measure.md`).

## Sources

- AMD, *RDNA 3 Instruction Set Architecture Reference Guide* — https://gpuopen.com/download/rdna3-shader-instruction-set-architecture-feb-2023.pdf
- LLVM, *User Guide for AMDGPU Backend* — https://llvm.org/docs/AMDGPUUsage.html
- ROCm, *GPU hardware specifications* — https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
- GPUOpen, *How to accelerate AI applications on RDNA 3 using WMMA* — https://gpuopen.com/learn/wmma_on_rdna3/
- AMD, *Radeon PRO W7900* — https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7900.html
- AMD, *Radeon PRO W7900D* (Chinese) — https://www.amd.com/zh-cn/products/graphics/workstations/radeon-pro/w7900d.html
