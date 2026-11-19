---
title: RDNA 3 (gfx1100) — execution model, registers, occupancy
kind: hardware
topic: execution
gens: [gfx1100]
updated: 2026-11-16
---

# Execution model, registers and occupancy

The card behind every "how many waves fit?" and "is this lane math right?" question on gfx1100.

## Hierarchy

- A **WGP** (work-group processor) holds **2 CUs** and **4 SIMD32s**. The W7900 has 48 WGPs (96 CUs).
- A work-group can run on any of the WGP's 4 SIMD32s; a WGP holds up to 32 work-groups of up to 1024
  work-items each. (ISA §2.3)
- **WGP mode is the default; CU mode is opt-in** (`-mcumode`). In WGP mode a work-group's waves can spread over
  both CUs and see one contiguous LDS, but the two CUs keep separate L0 caches, so work-group synchronization
  needs explicit L0 maintenance; the compiler emits it. In CU mode a work-group stays on one CU and sees only
  that CU's 64 KiB half of the LDS. (LLVM AMDGPUUsage, memory model GFX10–GFX11; ISA §3.3.4)

## Wave size

- gfx1100 runs **wave32 and wave64**; a program is compiled for one. **wave32 is the default**: LLVM's
  `wavefrontsize64` target feature is off unless `-mwavefrontsize64` is passed. (LLVM AMDGPUUsage)
- In wave64, VALU and VMEM instructions issue **twice** (low half, then high half); scalar ops, branches and
  messages issue once. (ISA §2.1)
- Triton compiles gfx1100 kernels for wave32 too: a Triton 3.8 (ROCm 10.1) build on a W7900D emits
  `.amdhsa_wavefront_size32 1`. An older or patched Triton may differ, so read that directive in the dump before
  doing lane arithmetic.
- Consequence: `__shfl*`, ballots, `warpSize`-based reductions and any lane-index arithmetic written for CDNA's
  64 lanes are wrong in wave32. Use `warpSize` or the wave size the build actually chose.

## Dual-issue VALU (VOPD)

- Two independent VALU operations packed into one instruction; **wave32 only** (wave64 does not use VOPD).
  (ISA §7.6)
- Constraints: 4 VGPR banks selected by `SRC[1:0]`; SRCX0 and SRCY0 must sit in different banks, as must VSRCX1
  and VSRCY1; at most one literal (or both ops share it).
- It is how the FP32 peak (61.3 TFLOPS) is reached. A kernel whose VALU stream the compiler cannot pair runs at
  about half that. Look for `v_dual_*` in the disassembly to see whether pairing happened.

## Registers

| | gfx1100 | Source |
|---|---|---|
| VGPRs per wave | ≤ 256 | ISA §3.3.2.1 |
| Physical VGPRs per SIMD | **1536 in wave32**, 768 in wave64 | LLVM `getTotalNumVGPRs` |
| Allocation granule | **24 VGPRs in wave32**, 12 in wave64 | ISA §3.3.2.1, LLVM `getVGPRAllocGranule` |
| AGPRs | **none** | ISA register tables list only VGPRs and SGPRs |
| SGPRs per wave | 106 (+16 trap TTMPs); never limit occupancy on GFX10+ | ISA Table 4, LLVM |
| Max waves per SIMD | 16 | LLVM `getMaxWavesPerEU` |

The kernel descriptor encodes VGPR counts in its own unit (8 in wave32, 4 in wave64); that is not the
allocation granule. Read the compiler's `.vgpr_count` / `NumVgprs`, not the descriptor field.

## Occupancy (wave32)

```
waves per SIMD = min(16, floor(1536 / roundup(vgprs, 24)))
```

| VGPRs used | Rounded | Waves/SIMD |
|---|---|---|
| ≤ 96 | 96 | **16** (full) |
| 128 | 144 | 10 |
| 168 | 168 | 9 |
| 192 | 192 | 8 |
| 256 | 264 | 5 |

- 96 VGPRs is the full-occupancy line. Above it occupancy falls in steps, but a wave can use up to 256 VGPRs
  without spilling, so trading occupancy for registers is often right for GEMM-like kernels.
- LDS can also cap occupancy: 128 KiB per WGP shared by its work-groups (`rdna3_memory.md`).
- For wave64 builds use 768 and a granule of 12 in the formula.
