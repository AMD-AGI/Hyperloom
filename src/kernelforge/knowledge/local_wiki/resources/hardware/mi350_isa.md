---
title: MI350X — gfx950 ISA, toolchain, reading the disassembly
kind: hardware
topic: isa
gens: [gfx950]
updated: 2026-08-28
---

# gfx950 ISA and toolchain

What changes at the instruction level, and how to confirm the compiler actually emitted it.

## Target and toolchain

| Item | Value |
|---|---|
| ISA target | **`gfx950`** |
| Compile | `--offload-arch=gfx950` |
| Wave size | **64** |
| Matrix-calculator keyword | `cdna4` |
| Min ROCm for scaled MFMA | **7.0** |

## Instruction families that changed

| Area | gfx950 |
|---|---|
| **Matrix** | + `v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4` (E8M0 block scale) · + classic f8f6f4 FP6/FP4 · + FP16/BF16 **16×16×32, 32×32×16** · **TF32 removed** · FP64 matrix halved |
| **FP8** | **OCP** (E4M3FN / E5M2) instead of FNUZ |
| **Direct g→LDS** | `global_load_lds` / `buffer_load ... lds` accept **1/2/4/12/16 DWORD** (96- and 128-bit added) |
| **LDS** | 160 KiB, **64 banks**, **read-with-transpose `ds`** loads, 320-DWORD alloc granularity |
| **Carryover** | `v_smfmac_*`, `v_accvgpr_read/write_b32`, count-based `s_waitcnt`, `buffer_*` / `global_*` / `ds_*` |

## Wait counters

`s_waitcnt <counter>(N)` means **"wait until ≤ N outstanding"**, not "wait N instructions".
Counters: `vmcnt` (VMEM), `lgkmcnt` (LDS/SMEM). Count-based waiting is what makes deep prefetch
overlap expressible — wait only for the loads you need right now.

## The scaled MFMA call

```cpp
// ROCm 7.0+, gfx950. Type codes: 0=E4M3 1=E5M2 2=E2M3 3=E3M2 4=E2M1
// scale = E8M0 -> factor 2^(scale-127); 127 = no scaling.
acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
          a, b, acc, Atype, Btype, /*opsel_a*/0, scale_a, /*opsel_b*/0, scale_b);
// also: __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
```

A/B types are independent; scales apply after the dot product, before accumulate.
Headers: `hip_fp8.h` (`__hip_fp8_*`, the older FNUZ path) and **`hip_ext_ocp.h`** (`__amd_fp8_*`,
`__amd_fp4x2_storage_t`, `__amd_create_fp4x2` — hardware-accelerated on gfx950).

## Direct global→LDS at 128 bit

```asm
global_load_lds_dwordx4 ...        ; 16 B/lane straight into LDS
buffer_load_dwordx4 ... lds        ; descriptor form
```

4× the CDNA3 width. Eliminates `ds_write` and the staging VGPRs. Combine with read-with-transpose to
feed MFMA without a transpose pass (`mi350_lds.md`).

## Reading the disassembly

```bash
hipcc --offload-arch=gfx950 -S -o - kern.hip
llvm-objdump -d --arch-name=amdgcn --mcpu=gfx950 kern.o
```

For Triton, dump AMDGCN via the cache / `AMDGCN_ENABLE_DUMP`.

**The checklist for any "did my change land?" question:**

| Look for | Pass |
|---|---|
| `v_mfma_scale_*` | present and native, not emulated |
| MFMA shape | the 16×16 form you asked for |
| `global_load_lds` | **12/16-DWORD** form, not 1/2/4 |
| `ds_read_b128` / `ds_write_b128` | wide forms in the hot loop, not `b32` |
| `.vgpr_count` / `.agpr_count` | matches your budget, below the tier boundary you targeted |
| **scratch `buffer_load`/`buffer_store`** | **none** in the hot loop — any spill is a bug |
| `.lds_size` | within the 160 KiB budget after 320-DWORD rounding |
| TF32 | **no** TF32 path will be emitted — BF16/FP32 is the fallback |

A "win" whose ISA is byte-identical to the baseline is measurement noise, every time.

## What it means for kernels

1. **Use `v_mfma_scale_*`** for MXFP4/6/8 (ROCm ≥ 7.0).
2. **Emit 128-bit `global_load_lds`** for tile staging.
3. **Use read-with-transpose `ds`** for the B operand.
4. **OCP FP8** in both the quantizer and the kernel.
5. **Drop TF32 code paths** — emulate with BF16 or run FP32.
6. Unchanged best practice: fine-grained `s_waitcnt`, AGPR accumulators, `ds_*_b128`.

## Pitfalls
- **Targeting gfx942 opcodes or FNUZ FP8** on gfx950.
- **Expecting TF32 or full-rate FP64 matrix** — removed / halved.
- **ROCm < 7.0** — the scaled intrinsics are not there.
- **32-bit-only direct-to-LDS** — leaves the 128-bit width unused.
- **Trusting source over ISA** — the compiler silently narrows loads it cannot prove aligned.

## Verify
- `amd_matrix_instruction_calculator --architecture cdna4 --list-instructions` for the full gfx950
  MFMA set; `--detail-instruction` for cycles and register layout.
- Disassemble and walk the checklist above.

## Related
`mi350_matrix_core.md` (shapes, cycles, the scaled intrinsic) · `mi350_lds.md` (direct-to-LDS) ·
`mi350_dtypes.md` (OCP FP8, MXFP)
