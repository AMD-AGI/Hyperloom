---
title: CDNA4 / MI350 (gfx950) — Matrix Core & block-scaled MFMA
kind: hardware
gens: [gfx950]
dtypes: [fp64, fp32, bf16, fp16, fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1, mxfp8, mxfp6, mxfp4, int8]
regimes: [both]
updated: 2026-06-08
---

# CDNA4 / MI350 (gfx950) — Matrix Core & block-scaled MFMA

> Cross-gen concepts in [../shared/matrix_core_mfma_smfmac.md](../shared/matrix_core_mfma_smfmac.md);
> FP6/FP4/MXFP numerics in [fp4_fp6_microscaling.md](fp4_fp6_microscaling.md). This file is the CDNA4
> instruction table, the **block-scaled** intrinsic, and the layout it expects.

## TL;DR
> 256 CU × 4 = **1024 Matrix Cores**, each **2× the FP16/FP8 rate** of CDNA3 → FP16/BF16 **2.5 PF**,
> FP8 **5 PF**, FP6/FP4 **10 PF**. The headline new op is **`v_mfma_scale_f32_32x32x64_f8f6f4`**:
> A and B may be any of FP8/FP6/FP4 (chosen independently) with **per-32-element E8M0 scales** applied
> after the dot product. **FP8 is OCP** here; **TF32 is gone**.

## Concepts

### What changed from CDNA3
| Feature | CDNA3 | CDNA4 |
|---|---|---|
| FP8 variant | FNUZ | **OCP** (E4M3FN/E5M2) |
| TF32 | present (654 TF) | **removed** (emulate w/ BF16 or run FP32) |
| FP6 (E2M3/E3M2), FP4 (E2M1) | ✗ | ✓ |
| Block-scaled MXFP8/6/4 | ✗ | ✓ (`v_mfma_scale_*`, E8M0) |
| New FP16/BF16 shapes | — | **16×16×32, 32×32×16** |
| New f8f6f4 shapes | — | **16×16×128, 32×32×64** |
| Read-with-transpose LDS for MFMA | ✗ | ✓ |
| FP64 matrix rate | 1× | **0.5×** |

### CDNA4 MFMA shape/cycle table (inference-relevant)
| Type (out←in) | Shapes | Cycles |
|---|---|---|
| FP64←FP64 | 16×16×4 | 64 |
| FP32←FP32 | 32×32×2 / 16×16×4 | 64 / 32 |
| FP32←FP16/BF16 | 32×32×8, 16×16×16, **+ 32×32×16, 16×16×32** | 32 / 16 |
| FP32←FP8 (OCP) | 16×16×32, 32×32×16 | 16 / 32 |
| FP32←{FP8/FP6/FP4} (f8f6f4) | **16×16×128, 32×32×64** | 16 or 32 / 32 or 64 |
| FP32←{MXFP8/6/4} (scaled) | **16×16×128, 32×32×64** | 16 or 32 / 32 or 64 |
| INT32←INT8 | 16×16×32, 32×32×16 | 16 / 32 |

> Cycle rule for f8f6f4/scaled: the **lower** count applies when neither A nor B is FP8 (i.e.
> FP6/FP4-only); the **higher** count applies when **either matrix is FP8**. This is why FP6/FP4 reach
> 10 PF while FP8 tops at 5 PF. SMFMAC (4:2 sparse) carries over from CDNA3.

### The block-scaled intrinsic
```cpp
// gfx950, ROCm 7.0+. Type codes: 0=E4M3(fp8) 1=E5M2(bf8) 2=E2M3(fp6) 3=E3M2(bf6) 4=E2M1(fp4)
// scale_a/scale_b are E8M0 -> factor 2^(scale-127); 127 = no scaling.
acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
          a_reg, b_reg, acc,
          /*Atype*/Acode, /*Btype*/Bcode,
          /*OPSEL_A*/0, scale_a,
          /*OPSEL_B*/0, scale_b);
// also: __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
```
Scales are applied **after the normal dot product and prior to accumulation**. A and B types are
**independent** (e.g. A=FP4, B=FP6). Classic (non-scaled) FP8/FP6/FP4 use
`v_mfma_f32_*_f8f6f4` without the scale operands.

### Block-scaled layout (32×32×64 example)
A = 32×64, **Ax (scales) = 32×2**, B = 64×32, **Bx = 2×32**, C = 32×32. Per-thread (wave64): **32 A,
1 Ax, 32 B, 1 Bx, 16 C**. Each E8M0 scale covers a **32-element block** of the K dimension. FP4 packs
2 values/byte (`__amd_fp4x2_storage_t`); the scaled intrinsic wants its first two operands 256-bit
wide, so 32 FP4 (128 bit) pad the upper 128 bits with zero. See
[fp4_fp6_microscaling.md](fp4_fp6_microscaling.md).

### Numerics
- **FP8 = OCP** (E4M3FN bias 7, ±448, ±0, NaN; E5M2 bias 15, with inf). Re-cast any FNUZ checkpoint.
- **Subnormals fully supported**; accumulate in FP32/INT32.
- Headers: `hip_fp8.h` (`__hip_fp8_*`) and `hip_ext_ocp.h` (`__amd_fp8_*` / `__amd_fp4*`,
  hardware-accelerated on gfx950). Detail: [../shared/dtype_numerics.md](../shared/dtype_numerics.md).

## The levers
1. **`16x16` over `32x32`** still wins on register footprint.
2. **MXFP4/6 block-scaled** for weight-heavy layers (32×/64× FP32) with a task-accuracy gate.
3. **Independent A/B types** — mix FP4 weights with FP6/FP8 activations as accuracy demands.
4. **OCP FP8** everywhere; drop TF32 paths.
5. **Read-with-transpose `ds`** to feed B without an explicit transpose.

## Pitfalls
- **FNUZ→OCP bit-copy** corrupts FP8.
- **Wrong E8M0 scale layout** (Ax/Bx) — the calculator's `--get-register` gives the exact map.
- **Expecting TF32** — removed.
- **Assuming FP6 is slower than FP4** — they share the 10 PF rate.

## Verify
- `amd_matrix_instruction_calculator --architecture cdna4 --instruction
  v_mfma_scale_f32_32x32x64_f8f6f4 --detail-instruction` (cycles, FLOPs/CU/cycle, scale layout).
- `--get-register --A-matrix`/`--Ax` for exact scale/operand placement before wiring scales.
