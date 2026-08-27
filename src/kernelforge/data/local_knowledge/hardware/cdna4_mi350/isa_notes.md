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
> [matrix_core_blockscale.md](matrix_core_blockscale.md); memory ops in [memory.md](memory.md).

## TL;DR
> Target **`--offload-arch=gfx950`**, wave64. New since gfx942: **`v_mfma_scale_*_f8f6f4`**
> (block-scaled MFMA), **FP6/FP4** opcodes and new FP16/BF16 large shapes, **96/128-bit
> GLOBAL_LOAD_LDS**, **read-with-transpose `ds` loads**, **160 KiB LDS** (320-DWORD alloc). Removed:
> **TF32 hardware** and full-rate FP64 matrix. ROCm **7.0+** for the scaled intrinsics.

## Concepts

### Target & toolchain
| Item | Value |
|---|---|
| ISA target | `gfx950` |
| Compile | `--offload-arch=gfx950` |
| Wave size | **64** |
| Calculator keyword | `cdna4` |
| Min ROCm for scaled MFMA | **7.0** |
| Sibling | `gfx942` (CDNA3) — see [../cdna3_mi300/isa_notes.md](../cdna3_mi300/isa_notes.md) |

### New / changed instruction families vs gfx942
| Area | Change |
|---|---|
| Matrix | + `v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4` (E8M0 block scale); + classic f8f6f4 FP6/FP4; + FP16/BF16 16×16×32, 32×32×16; **TF32 removed**; FP64 matrix halved |
| FP8 | **OCP** (E4M3FN/E5M2) instead of FNUZ |
| Direct g→LDS | `global_load_lds` / `buffer_load ... lds` accept **1/2/4/12/16 DWORD** (96/128-bit added) |
| LDS | 160 KiB, 64 banks, **read-with-transpose `ds`** loads; 320-DWORD alloc granularity |
| Carryover | `v_smfmac_*`, `v_accvgpr_read/write_b32`, count-based `s_waitcnt`, `buffer_*`/`global_*`/`ds_*` |

### Scaled MFMA call (the headline)
```cpp
// ROCm 7.0+, gfx950. codes: 0=E4M3 1=E5M2 2=E2M3 3=E3M2 4=E2M1 ; scale = E8M0 -> 2^(scale-127)
acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
          a, b, acc, Atype, Btype, /*opsel_a*/0, scale_a, /*opsel_b*/0, scale_b);
```
A/B types independent; scales applied after the dot product, before accumulate. Headers: `hip_fp8.h`
(`__hip_fp8_*`), `hip_ext_ocp.h` (`__amd_fp8_*`, `__amd_fp4x2_storage_t`, `__amd_create_fp4x2`).

### Direct global→LDS (128-bit)
```asm
global_load_lds_dwordx4 ...        ; 16 B/lane straight into LDS (gfx950)
buffer_load_dwordx4 ... lds        ; descriptor form
```
4× the CDNA3 width; eliminates `ds_write` + staging VGPRs. Combine with read-with-transpose to feed
MFMA without a transpose pass. See [memory.md](memory.md).

### Reading the ISA dump
```bash
hipcc --offload-arch=gfx950 -S -o - kern.hip
llvm-objdump -d --arch-name=amdgcn --mcpu=gfx950 kern.o
```
Confirm `v_mfma_scale_*` (not emulated), the 12/16-DWORD `global_load_lds`, and `.lds_size` against the
160 KiB budget. **No TF32** path will be emitted; BF16/FP32 is the fallback.

## The levers
1. **Use `v_mfma_scale_*`** for MXFP4/6/8 (ROCm ≥7.0).
2. **Emit 128-bit `global_load_lds`** for tile staging.
3. **Read-with-transpose `ds`** for the B operand.
4. **OCP FP8** in both quant and kernel.
5. **Drop TF32**; emulate with BF16 or run FP32 if needed.
6. **Fine-grained `s_waitcnt`**, AGPR accumulators, `ds_*_b128` — unchanged best practices.

## Pitfalls
- **Targeting gfx942 opcodes / FNUZ FP8** on gfx950.
- **Expecting TF32 or full-rate FP64 matrix** — removed/halved.
- **Old ROCm (<7.0)** lacking the scaled intrinsics.
- **32-bit-only direct-to-LDS** leaving the 128-bit width unused.

## Verify
- `amd_matrix_instruction_calculator --architecture cdna4 --list-instructions` for the gfx950 MFMA set.
- Disassemble and confirm scaled MFMA, 128-bit direct-to-LDS, and LDS within 160 KiB.
