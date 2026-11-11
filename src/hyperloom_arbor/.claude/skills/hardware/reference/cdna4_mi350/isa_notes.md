---
title: CDNA4 / MI350 (gfx950) — ISA notes for kernel authors
kind: hardware
gens: [gfx950]
dtypes: []
regimes: [both]
updated: 2026-06-08
---

# CDNA4 / MI350 (gfx950) — ISA notes for kernel authors

> Just the gfx950 ISA deltas vs gfx942 that change how you write/read kernels. MFMA detail in
> [matrix_core_blockscale.md](matrix_core_blockscale.md); memory ops in [memory.md](memory.md). Numbers in
> brackets are the sources at the end.

## TL;DR
> Target **`--offload-arch=gfx950`** [3], wave64 [1]. New since gfx942: **`v_mfma_scale_*_f8f6f4`**
> (block-scaled MFMA), **FP6/FP4** opcodes and new FP16/BF16 large shapes [1], **96/128-bit
> GLOBAL_LOAD_LDS** [1], **read-with-transpose `ds` loads** [1], **160 KiB LDS** (1280-byte, 320-DWORD
> allocation blocks) [1]. Removed: **TF32 hardware** [4] (the gfx942 XF32 MFMAs have no gfx950
> counterpart [1][5]); the FP64 matrix rate is halved [4]. ROCm **7.0+** for the scaled intrinsics [2].

## Concepts

### Target & toolchain
| Item | Value |
|---|---|
| ISA target | `gfx950` [3] |
| Compile | `--offload-arch=gfx950` |
| Wave size | **64** [1] |
| Matrix Instruction Calculator | covers CDNA1–CDNA3, **not CDNA4** [6]; use the ISA guide's tables [1] |
| Min ROCm for scaled MFMA | **7.0** [2] |
| Sibling | `gfx942` (CDNA3) — see [../cdna3_mi300/isa_notes.md](../cdna3_mi300/isa_notes.md) |

### New / changed instruction families vs gfx942
| Area | Change |
|---|---|
| Matrix | + `v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4` (E8M0 block scale); + unscaled f8f6f4 FP8/FP6/FP4; + FP16/BF16 16×16×32, 32×32×16; + INT8 16×16×64, 32×32×32 [1]; **TF32 removed**; FP64 matrix halved [4] |
| FP8 | **OCP** (E4M3FN/E5M2) instead of FNUZ [2][7] |
| Direct g→LDS | `global_load_lds_*` / `buffer_load_* ... lds` move 1, 2, 4, 12 or 16 bytes per lane (`ubyte`/`sbyte`, `ushort`/`sshort`, `dword`, `dwordx3`, `dwordx4`): the 96/128-bit forms are new [1][5] |
| LDS | 160 KiB, 64 banks, **read-with-transpose `ds`** loads (`ds_read_b64_tr_b4/b8/b16`, `ds_read_b96_tr_b6`); 1280-byte (320-DWORD) allocation blocks [1] |
| Carryover | `v_smfmac_*`, `v_accvgpr_read/write_b32`, count-based `s_waitcnt`, `buffer_*`/`global_*`/`ds_*` [1] |

### Scaled MFMA call (the headline)
```cpp
// ROCm 7.0+, gfx950. codes: 0=E4M3 1=E5M2 2=E2M3 3=E3M2 4=E2M1 ; scale = E8M0 -> 2^(scale-127)
acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
          a, b, acc, Atype, Btype, /*opsel_a*/0, scale_a, /*opsel_b*/0, scale_b);
```
A/B types independent; scales applied after the dot product, before accumulate [2]. Headers: `hip_fp8.h`
(`__hip_fp8_*`), `hip_ext_ocp.h` (`__amd_fp8_*`, `__amd_fp4x2_storage_t`, `__amd_create_fp4x2`) [2][7].

### Direct global→LDS (128-bit)
```asm
global_load_lds_dwordx4 ...        ; 16 B/lane straight into LDS (gfx950)
buffer_load_dwordx4 ... lds        ; descriptor form
```
4× the CDNA3 width [1][5]; the data bypasses VGPRs, so no `ds_write` or staging VGPRs [1]. Combine with
read-with-transpose to feed MFMA without a transpose pass. See [memory.md](memory.md).

### Reading the ISA dump
```bash
hipcc --offload-arch=gfx950 -S -o - kern.hip
llvm-objdump -d --arch-name=amdgcn --mcpu=gfx950 kern.o
```
Confirm `v_mfma_scale_*` (not emulated), the `dwordx3`/`dwordx4` `global_load_lds`, and `.lds_size`
against the 160 KiB budget. gfx950 has no TF32 (XF32) MFMA [1], so none is emitted; BF16/FP32 is the
fallback [4].

## The levers
1. **Use `v_mfma_scale_*`** for MXFP4/6/8 (ROCm ≥7.0).
2. **Emit 128-bit `global_load_lds`** for tile staging.
3. **Read-with-transpose `ds`** for an MFMA operand.
4. **OCP FP8** in both quant and kernel.
5. **Drop TF32**; emulate with BF16 or run FP32 if needed.
6. **Fine-grained `s_waitcnt`**, AGPR accumulators, `ds_*_b128` (AMD's guidance is `_b128` LDS accesses
   [8]) — unchanged best practices.

## Pitfalls
- **Targeting gfx942 opcodes / FNUZ FP8** on gfx950.
- **Expecting TF32 or full-rate FP64 matrix** — removed/halved.
- **Old ROCm (<7.0)** lacking the scaled intrinsics.
- **32-bit-only direct-to-LDS** leaving the 128-bit width unused.

## Verify
- The gfx950 MFMA set: CDNA4 ISA §7.1.2 Table 28 and §7.2.1 [1].
- Disassemble and confirm scaled MFMA, 128-bit direct-to-LDS, and LDS within 160 KiB.

## Sources
1. AMD, *CDNA4 Instruction Set Architecture Reference Guide*, 5 Aug 2025: §1.1 (wavefront of 64
   work-items), §2.2.1 (LDS), §3.6.5 (1280-byte LDS blocks), chapter 7 (MFMA, SMFMAC, `V_ACCVGPR_*`),
   §9.1.9 and §10 (loads to LDS), §11.4 (transpose loads), instruction lists (`S_WAITCNT`).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf>
2. ROCm Blogs, *Matrix Core Programming on AMD CDNA3 and CDNA4 architecture* (scaled intrinsic, ROCm 7.0
   or later, `hip_ext_ocp.h`, OCP vs FNUZ).
   <https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html>
3. ROCm documentation, *AMD GPU specifications* (LLVM target name gfx950).
   <https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html>
4. AMD, *Introducing AMD CDNA 4 Architecture* (white paper): TF32 moved to software via BF16; FP64 matrix
   FLOPs per clock per CU halved.
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf>
5. AMD, *AMD Instinct MI300 / CDNA3 Instruction Set Architecture Reference Guide* (XF32 MFMAs; loads to
   LDS up to a dword).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>
6. AMD Matrix Instruction Calculator (supported architectures: CDNA1–CDNA3, RDNA3, RDNA4).
   <https://github.com/ROCm/amd_matrix_instruction_calculator>
7. HIP documentation, *Low precision floating point types*.
   <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html>
8. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* (use `_b128` LDS
   loads and stores).
   <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html>
