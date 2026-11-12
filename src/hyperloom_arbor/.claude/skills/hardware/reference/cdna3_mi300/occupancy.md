---
title: CDNA3 / MI300X (gfx942) — occupancy math & tuning
kind: hardware
gens: [gfx942]
dtypes: []
regimes: [both]
updated: 2026-11-07
---

# CDNA3 / MI300X (gfx942) — occupancy math & tuning

> Cross-gen model in [../shared/wavefront_simd_vgpr_agpr.md](../shared/wavefront_simd_vgpr_agpr.md).
> This file is the MI300X numbers and worked examples. Numbers in brackets point to the Sources at the
> end.

## TL;DR
> Occupancy = resident waves/SIMD, capped at **8** (32/CU) [3]. It is the **minimum** of the VGPR limit
> (`floor(512/N)`), the LDS limit (`floor(65536/L)`), and the wave-slot cap [1][5]. More resident waves
> hide more memory latency [8], but a tuned MFMA GEMM can run at 4 waves per CU (one per SIMD) and
> hide latency with a pipelined, double-buffered LDS instead [1].

## Concepts

### Inputs
1. **N** = VGPRs/wave (ISA `.vgpr_count`), rounded up to the allocation granule: groups of **8** per the
   CDNA3 ISA [2]; AMD's workload guide uses blocks of 16 in its MI300X example [1].
2. **L** = LDS bytes/workgroup (allocated in 512 B blocks [2]).
3. **nW** = waves/workgroup = `ceil(threads_per_block / 64)`.

### Limits (MI300X)
```
occ_vgpr (waves/SIMD)    = floor(512 / N)            # 512 per EU [1] (512 KiB file/CU [4]); cap 8 [3]
occ_lds  (workgroups/CU) = floor(65536 / L)          # 64 KiB LDS [1]
wave_slots               = 8/SIMD = 32/CU            # hard cap [3]
wg_from_vgpr             = floor(occ_vgpr * 4 / nW)  # 4 SIMDs/CU [1]
wg_per_CU                = min(wg_from_vgpr, occ_lds, floor(32 / nW))
waves_per_CU             = wg_per_CU * nW
```
The guide's own formula is `occ = min(floor(occ_vgpr * 4 / nW), occ_lds) * nW / 4` [1][5]; the
`floor(32 / nW)` term adds the wave-slot cap.

### Worked examples
**A — VGPR-limited GEMM.** N=176, threads=256 (nW=4), L=32 KiB.
```
occ_vgpr=floor(512/176)=2 ; wg_from_vgpr=floor(2*4/4)=2 ; occ_lds=floor(65536/32768)=2
wg_per_CU=min(2,2,8)=2 -> 8 waves/CU
```
Dropping N to 128 gives occ_vgpr=4 but `occ_lds=2` now binds → still 2 wg/CU. Reduce LDS to gain.

**B — LDS-limited attention/softmax.** N=64, threads=512 (nW=8), L=48 KiB.
```
occ_vgpr=floor(512/64)=8 ; wg_from_vgpr=floor(8*4/8)=4 ; occ_lds=floor(65536/49152)=1  <-- binds
wg_per_CU=min(4,1,4)=1 -> 8 waves/CU
```
Halving LDS → occ_lds=2 → 2 wg/CU → 16 waves/CU.

**C — fully occupied bandwidth kernel.** N=48, threads=256 (nW=4), L=8 KiB.
```
occ_vgpr=floor(512/48)=10->cap 8 ; wg_from_vgpr=floor(8*4/4)=8 ; occ_lds=floor(65536/8192)=8
wave-slot cap floor(32/4)=8 -> wg_per_CU=8 -> 32 waves/CU (max)
```

### The VGPR granule trap
170 VGPRs rounds to 176, and since 176 × 3 > 512 the kernel gets 2 waves per SIMD, not 3 [1]. With the
ISA's 8-register granule, `floor(512/N)` changes tier at N = 64 (8 waves), 72, 80, 96, 128, 168, 256
(2 waves) and 512 (1) (derived) — shaving a few VGPRs across a boundary can jump an occupancy tier.

## The levers
1. **Cut VGPRs** (smaller tile; Triton `waves_per_eu` [1]; the **AGPR escape hatch**
   `amdgpu-mfma-vgpr-form=false` and `amdgpu-agpr-alloc=256`, at ~5% of the loop for epilogue copies [1]).
2. **Cut LDS** (less buffering, packing) when LDS binds (example B).
3. **`buffer_load_to_lds`** removes staging VGPRs (~100 per wave in AMD's guide) [1] — see
   [memory_hierarchy.md](memory_hierarchy.md).
4. **`__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)`** caps VGPRs so that many
   waves fit per SIMD [6]; the compiler-level form is the `amdgpu-waves-per-eu` attribute [7].
5. **Occupancy vs pipelining**: raise occupancy for latency-bound kernels [8]; for MFMA GEMM, AMD's
   guidance is 4 waves per CU with a deep software pipeline [1].

## Tuning checklist (from Omniperf symptoms)
| Symptom | Cause | Fix |
|---|---|---|
| Low waves/CU, high VGPR | register pressure | shrink tile, waves_per_eu, AGPR, buffer_load_to_lds [1] |
| Low waves/CU, high LDS | LDS pressure | smaller tiles, less buffering, pack |
| LDS bank-conflict stalls | column access / stride a multiple of 32 banks [2] | pad the leading dim, swizzle, ds_*_b128 |
| Low HBM BW | uncoalesced/scalar loads | global_load_dwordx4 [1], cache-line alignment [8] |
| Memory-latency-bound | too few waves | raise occupancy [8]; prefetch |
| MFMA underfed | LDS read conflicts / no double-buffer | padded or swizzled LDS, async copy + double buffer [1] |

**AMD's GEMM guidance (CDNA3/CDNA4, Gluon):** 256×256 output tile, 2-stage prefetch, **384–448 VGPR**
budget at 4 waves per CU, `buffer_load_to_lds`, double-buffered padded LDS [1]; `mfma_16x16` on
MI300X [1].

## Pitfalls
- **Maximizing occupancy blindly** — GEMM can win at low occupancy with deep prefetch [1].
- **Ignoring the LDS limiter** (example B).
- **Forgetting granule rounding** when budgeting VGPRs [1][2].

## Verify
- ISA `.vgpr_count` and the kernel's LDS size; the `occ.sh` script from ROCm/triton turns them into
  occupancy [1].
- `rocprof-compute` occupancy panel: resident vs theoretical waves, which resource binds.

## Sources
1. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* (`waves_per_eu`, "Compute the occupancy of a kernel", Gluon register pressure) — https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html
2. AMD, *"AMD Instinct MI300" Instruction Set Architecture* (CDNA3 ISA; §3.6.4 VGPR allocation, §3.6.5 LDS allocation, §11 LDS banks) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
3. ROCm Compute Profiler documentation, *Pipeline descriptions* (8 wavefront slots per SIMD, 32 per CU) — https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.4.1/conceptual/pipeline-descriptions.html
4. ROCm documentation, *AMD GPU specifications* — https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
5. ROCm 6.1.1 documentation, *Optimizing Triton kernels* (kernel occupancy) — https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
6. HIP documentation, *HIP C++ language extensions* (`__launch_bounds__`) — https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html
7. LLVM, *User Guide for AMDGPU Backend* (`amdgpu-waves-per-eu`) — https://llvm.org/docs/AMDGPUUsage.html
8. HIP documentation, *Hardware implementation* (register pressure and occupancy; coalescing) — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
