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
> instruction table, the **block-scaled** intrinsic, and the layout it expects. Numbers in brackets are the
> sources at the end.

## TL;DR
> 256 CU × 4 = **1024 Matrix Cores** [3], each CU at **2× the FP16/FP8 FLOPs per clock** of CDNA3 [3] →
> FP16/BF16 **2.5 PF**, FP8 **5 PF**, MXFP6/MXFP4 **10 PF** on MI355X [3]. The headline new op is
> **`v_mfma_scale_f32_32x32x64_f8f6f4`**: A and B may be any of FP8/FP6/FP4 (chosen independently) with
> **per-32-element E8M0 scales** applied after the dot product [1][2]. **FP8 is OCP** here [2][5];
> **TF32 is gone** [3].

## Concepts

### What changed from CDNA3
| Feature | CDNA3 | CDNA4 |
|---|---|---|
| FP8 variant | FNUZ [2][5] | **OCP** (E4M3FN/E5M2) [2][5] |
| TF32 | present (653.7 TF on MI300X) [4] | **removed** (software emulation via BF16 [3], or run FP32) |
| FP6 (E2M3/E3M2), FP4 (E2M1) | ✗ [7] | ✓ [1] |
| Block-scaled MXFP8/6/4 | ✗ [4] | ✓ (`v_mfma_scale_*`, E8M0) [1] |
| New FP16/BF16 shapes | — | **16×16×32, 32×32×16** [1][2] |
| New f8f6f4 shapes | — | **16×16×128, 32×32×64** [1][2] |
| Read-with-transpose LDS loads for MFMA | ✗ [7] | ✓ [1] |
| FP64 matrix rate per CU | 256 FLOPs/clock | **128** (0.5×) [3] |

### CDNA4 MFMA shape/cycle table (inference-relevant) [1][2]
| Type (out←in) | Shapes | Cycles |
|---|---|---|
| FP64←FP64 | 16×16×4 | 64 |
| FP32←FP32 | 32×32×2 / 16×16×4 | 64 / 32 |
| FP32←FP16/BF16 | 32×32×8, 16×16×16, **+ 32×32×16, 16×16×32** | 32 / 16 |
| FP32←FP8 (OCP) | 16×16×32, 32×32×16 | 16 / 32 |
| FP32←{FP8/FP6/FP4} (f8f6f4) | **16×16×128, 32×32×64** | 16 or 32 / 32 or 64 |
| FP32←{MXFP8/6/4} (scaled) | **16×16×128, 32×32×64** | 16 or 32 / 32 or 64 |
| INT32←INT8 | 16×16×32, 32×32×16, **+ 16×16×64, 32×32×32** | 16 / 32 |

> Cycle rule for f8f6f4/scaled: the **larger** count applies when **either matrix is FP8**, the smaller
> one when neither is (FP6/FP4 only) [1]. This is why MXFP6/MXFP4 reach 10 PF while FP8 tops at 5 PF.
> The peaks need the new shapes: with 1024 matrix cores at 2.4 GHz, the classic FP16 16×16×16 (16 cycles)
> gives 1.26 PF against 2.52 PF for 16×16×32, and the classic FP8 16×16×32 gives 2.52 PF against 5.03 PF
> for f8f6f4 16×16×128 with FP8 (derived from [1][3]; formula in [peak_tables.md](peak_tables.md)).
> SMFMAC (2:4 structured sparsity) carries over from CDNA3 [1].

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
(Signature, type codes and ROCm 7.0 requirement from [2]; the instructions from [1].) Scales are applied
**after the normal dot product and prior to accumulation** [2]. A and B types are **independent** (e.g.
A=FP4, B=FP6) [1]. Classic (non-scaled) FP8/FP6/FP4 use `v_mfma_f32_*_f8f6f4` without the scale
operands [1].

### Block-scaled layout (32×32×64 example)
A = 32×64, **Ax (scales) = 32×2**, B = 64×32, **Bx = 2×32**, C = 32×32. Per-thread (wave64): **32 A,
1 Ax, 32 B, 1 Bx, 16 C** [2]. Each E8M0 scale covers a **32-element block** of the K dimension [1]. FP4
packs 2 values/byte (`__amd_fp4x2_storage_t`); the scaled intrinsic wants its first two operands 256-bit
wide, so 32 FP4 (128 bit) pad the upper 128 bits with zero [2]. See
[fp4_fp6_microscaling.md](fp4_fp6_microscaling.md).

### Numerics
- **FP8 = OCP**: E4M3 bias 7, max ±448, ±0, NaN, no infinity; E5M2 bias 15, max ±57344, with ±inf [1][2].
  Re-cast any FNUZ checkpoint: the encodings differ [2].
- **Denormals kept**: the MFMA rules ignore the MODE denorm control and keep input/output denorms [1];
  outputs are FP32/INT32 [1].
- Headers: `hip_fp8.h` (`__hip_fp8_*`) and `hip_ext_ocp.h` (`__amd_fp8_*` / `__amd_fp4*`,
  hardware-accelerated on gfx950) [2][5]. Detail: [../shared/dtype_numerics.md](../shared/dtype_numerics.md).

## The levers
1. **`16x16` vs `32x32`:** AMD reports 16×16 ahead on MI300X; for MI350X it says to benchmark both [4].
2. **MXFP4/6 block-scaled** for weight-heavy layers (64× the FP32 matrix peak [3]) with a task-accuracy gate.
3. **Independent A/B types** — mix FP4 weights with FP6/FP8 activations as accuracy demands.
4. **OCP FP8** everywhere; drop TF32 paths.
5. **Use the new shapes** (16×16×32 FP16/BF16, f8f6f4 for FP8) to reach the peaks (above).
6. **Read-with-transpose `ds`** to feed an operand without an explicit transpose [1].

## Pitfalls
- **FNUZ→OCP bit-copy** corrupts FP8.
- **Wrong E8M0 scale layout** (Ax/Bx) — the ISA's scale data layouts give the exact map [1].
- **Expecting TF32** — removed.
- **Assuming FP6 is slower than FP4** — they share the 10 PF matrix rate [3].

## Verify
- Disassemble (`llvm-objdump --mcpu=gfx950`) and confirm `v_mfma_scale_*` and the new shapes are emitted.
- Operand and scale placement: CDNA4 ISA §7.1.4–§7.2.1 [1] and the worked layouts in [2]. AMD's Matrix
  Instruction Calculator covers CDNA1–CDNA3, not CDNA4 [6].

## Sources
1. AMD, *CDNA4 Instruction Set Architecture Reference Guide*, 5 Aug 2025: chapter 7 introduction (2:4
   sparse outer products), §7.1.2 (Table 28, dense MFMA shapes and cycles, and the MFMA rules), §7.1.4
   (F8F6F4 rules and type codes), §7.2.1 (block exponent scaling, one scale per 32 K-values, scale layouts), SMFMA Table 33,
   §7.3 Table 30 (small float formats).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf>
2. ROCm Blogs, *Matrix Core Programming on AMD CDNA3 and CDNA4 architecture* (format table, MFMA table,
   scaled intrinsic and layouts, ROCm 7.0 or later).
   <https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html>
3. AMD, *Introducing AMD CDNA 4 Architecture* (white paper): Table 1 (FLOPs per clock per CU), Table 2
   (1,024 matrix cores, peaks), TF32 moved to software.
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf>
4. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* (TF32 653.7 TF on
   MI300X, MXFP support by GPU, MI16x16 vs MI32x32).
   <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html>
5. HIP documentation, *Low precision floating point types* (`hip_fp8.h` types; OCP on CDNA4, FNUZ on
   gfx94x).
   <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html>
6. AMD Matrix Instruction Calculator (supported architectures: CDNA1–CDNA3, RDNA3, RDNA4).
   <https://github.com/ROCm/amd_matrix_instruction_calculator>
7. AMD, *AMD Instinct MI300 / CDNA3 Instruction Set Architecture Reference Guide* (no F8F6F4 or transpose
   loads from LDS).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>
