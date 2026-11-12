---
title: Wavefront / SIMD / VGPR / AGPR execution model (CDNA cross-gen)
kind: hardware
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: []
regimes: [both]
updated: 2026-11-07
---

# Wavefront / SIMD / VGPR / AGPR execution model

> Numbers in brackets point to the Sources at the end.

## TL;DR
> CDNA is **wave64**: a wavefront = **64 work-items** [2][3]; each CU has **4 SIMDs** (= 4 EUs) [1][9]
> and 4 Matrix Cores (see [matrix_core_mfma_smfmac.md](matrix_core_mfma_smfmac.md)). Occupancy is per-SIMD and is gated by the **minimum** of the VGPR limit, the
> LDS limit, and the **8-waves/SIMD** hard cap (CDNA2–4) [1][10]. Register usage limits how many
> waves fit; **AGPRs** let matmul keep big accumulators without spending the architected VGPR budget
> [5][6].

## Concepts

### Wave64 (no wave32 on CDNA)
- A wavefront is 64 lanes. On a 16-wide SIMD the issue physically spans **4 cycles** (16 lanes ×4)
  [1][8], but the programming model is "64 lanes, one instruction." Branch divergence uses the 64-bit
  `EXEC` mask [5]; cross-lane ops (DPP, `ds_swizzle`, `ds_bpermute`, HIP `__shfl`) work on the 64-lane
  wave, so all shuffle/ballot/reduction math is **mod 64**, not mod 32 (a frequent CUDA-port bug).
- All CDNA generations (gfx908/90a/942/950) keep wave64 [2][3]. MFMA is a wave-level op — all 64
  lanes cooperate on one `D=A·B+C` [5].

### CU → SIMD → wave slots
| Gen | SIMD/CU | Wave slots/SIMD | Wave slots/CU | Wave size |
|---|---|---|---|---|
| CDNA1 gfx908 | 4 [1] | 10 [1] | 40 | 64 [2] |
| CDNA2 gfx90a | 4 [1][9] | 8 [1][9] | 32 | 64 [2] |
| CDNA3 gfx942 | 4 [1][7] | 8 [1] | 32 | 64 [2] |
| CDNA4 gfx950 | 4 [7][10] | 8 [10] | 32 | 64 [2] |

(On CDNA1, other limits generally cap occupancy at 32 waves per CU anyway [1].) All waves of a
**workgroup run on one CU** [5].

### The register files: VGPR, SGPR, AGPR
| File | CDNA1 (gfx908) | CDNA2/3/4 (gfx90a/942/950) | Access |
|---|---|---|---|
| VGPR (architected) | V0–V255, a 256 KiB file per CU [2][3] | **512 per SIMD lane, shared with the AGPRs** (512 KiB file per CU) [2][4][5][6] | all VALU |
| AGPR (accumulation) | AV0–AV255, a separate file [2][3] | up to **256** of the 512 [4][5][6] | MFMA + `v_accvgpr_read/write_b32` [5]; CDNA2+ memory ops can also load into them [4] |
| SGPR (scalar) | 12.5 KiB per CU [2] | 12.5 KiB per CU [2]; a wave allocates 16–102 [5] | scalar unit |

- **CDNA1 introduced AGPRs** with its Matrix Core: the accumulation VGPRs are a separate file [3][1].
  **CDNA2 merged the architected and accumulation VGPRs into one unified pool**: a wave may have up to
  512 VGPRs, at most 256 of each type [4].
- **VGPR allocation granule**: groups of 4 on CDNA1 [3] and groups of 8 on CDNA2, CDNA3 and CDNA4 per
  the ISA guides [4][5][6]. AMD's workload guide instead says blocks of 16 in its MI300X example: 170
  VGPRs round to 176, and 176 × 3 > 512 limits the kernel to 2 waves per SIMD [7]. Either way the
  rounding alone can drop an occupancy tier.
- **AGPRs** are the "escape hatch": park large FP32 matmul accumulators there so they don't consume
  the architected VGPR budget. AMD's guide prices the epilogue AGPR→VGPR copies at ~5% of the loop
  for compute-bound kernels with large K [7].

### Occupancy math (per-SIMD)
```
occ_vgpr (waves/SIMD)    = floor(VGPR_per_SIMD / N)     # cap at wave-slot limit (8, or 10 on CDNA1)
                                                        # CDNA2-4: floor(512/N); CDNA1: floor(256/N)
occ_lds  (workgroups/CU) = floor(LDS_per_CU / L)        # 65536 (CDNA1-3) or 163840 (CDNA4)
nW                       = ceil(threads_per_block / 64) # waves per workgroup
wg_from_vgpr             = floor(occ_vgpr * 4 / nW)     # 4 SIMDs/CU
wg_per_CU                = min(wg_from_vgpr, occ_lds, floor(wave_slots_per_CU / nW))
waves_per_CU             = wg_per_CU * nW
```
(the guide's formula [7][15], with the wave-slot cap [1] added; 512 and 256 from the register files
above) Worked examples per generation live in each gen's `occupancy.md`.

## The levers
1. **Cut VGPRs to gain occupancy** — but watch the granule rounding and the LDS limit, which can bind
   first (see the occupancy examples).
2. **`__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)`** [11], the
   `amdgpu-waves-per-eu` attribute [12], or Triton's `waves_per_eu` [7] tell the compiler to cap VGPRs
   so N waves fit per EU.
3. **AGPR escape hatch** for GEMM: `amdgpu-mfma-vgpr-form=false`, `amdgpu-agpr-alloc=256` [7].
4. **Direct global→LDS** (`buffer_load ... lds`) removes staging VGPRs — ~100 VGPRs per wave in AMD's
   guide [7]; CDNA4 widens it to 128 b/lane [6]. See
   [memory_model_lds_bank.md](memory_model_lds_bank.md).
5. **Occupancy vs pipelining**: more resident waves hide more latency [13]; AMD's GEMM guidance instead
   runs 4 waves per CU with a deep, double-buffered LDS pipeline [7].

## Pitfalls
- **Assuming wave32.** All warp-width constants (shuffle masks, ballot, `__activemask`) must be 64.
- **The "512" double-meaning.** 512 = VGPRs **per SIMD lane** (per EU) [7], the occupancy-math number;
  the CU's VGPR file is 512 KiB across 4 SIMDs [2]. Don't conflate count with bytes.
- **Ignoring AGPR eligibility.** The calculator's `--detail-instruction` reports, per MFMA, whether
  each matrix can use ArchVGPRs and/or AccVGPRs [14].
- **CDNA1 256-VGPR ceiling.** gfx908 has 256 architected VGPRs (plus a separate AGPR file) [3] — port
  tile sizes down accordingly.

## Verify
- ISA dump `.vgpr_count` / `.agpr_count` / `.sgpr_count` [12], or
  `-Rpass-analysis=kernel-resource-usage` [9].
- `rocprof-compute` occupancy/wave panels: resident waves/CU vs theoretical, and which resource binds.
- The `occ.sh` script from ROCm/triton turns VGPR/LDS into occupancy [7].

## Sources
1. ROCm Compute Profiler documentation, *Pipeline descriptions* (VALU: 16-wide SIMDs, wavefront slots; AGPRs) — https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.4.1/conceptual/pipeline-descriptions.html
2. ROCm documentation, *AMD GPU specifications* (wavefront size, VGPR and SGPR files) — https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
3. AMD, *"AMD Instinct MI100" Instruction Set Architecture* (CDNA1 ISA; VGPR allocation, AccVGPRs) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi100-cdna1-shader-instruction-set-architecture.pdf
4. AMD, *"AMD Instinct MI200" Instruction Set Architecture* (CDNA2 ISA; feature changes, VGPR allocation) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf
5. AMD, *"AMD Instinct MI300" Instruction Set Architecture* (CDNA3 ISA; §3.1 state, §3.6.2 SGPRs, §3.6.4 VGPRs, §4.3 work-groups, §7.1 MFMA) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
6. AMD, *AMD Instinct CDNA4 Instruction Set Architecture* (§3.6.4 VGPRs, §9.1.9 load to LDS) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
7. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* — https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html
8. AMD, *AMD CDNA Architecture* white paper (MI100; compute units) — https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-white-paper.pdf
9. ROCm Blogs, *Register pressure in AMD CDNA2 GPUs* — https://rocm.blogs.amd.com/software-tools-optimization/register-pressure/README.html
10. ROCm Blogs, *Occupancy Math on the AMD MI355X GPU (CDNA4)* — https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html
11. HIP documentation, *HIP C++ language extensions* (`__launch_bounds__`) — https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html
12. LLVM, *User Guide for AMDGPU Backend* (`amdgpu-waves-per-eu`; code object metadata) — https://llvm.org/docs/AMDGPUUsage.html
13. HIP documentation, *Hardware implementation* (register pressure and occupancy) — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
14. ROCm, *AMD Matrix Instruction Calculator* README — https://github.com/ROCm/amd_matrix_instruction_calculator
15. ROCm 6.1.1 documentation, *Optimizing Triton kernels* (kernel occupancy) — https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
