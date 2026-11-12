---
title: CDNA3 / MI300X (gfx942) — Matrix Core / MFMA
kind: hardware
gens: [gfx942]
dtypes: [fp64, fp32, tf32, bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, int8]
regimes: [both]
updated: 2026-11-07
---

# CDNA3 / MI300X (gfx942) — Matrix Core / MFMA

> Cross-gen concepts (naming, lane mapping, peak formula, AGPR) live in
> [../shared/matrix_core_mfma_smfmac.md](../shared/matrix_core_mfma_smfmac.md). This file is the
> CDNA3-specific instruction table, intrinsics, and numerics. Numbers in brackets point to the Sources
> at the end.

## TL;DR
> 304 CU × 4 = **1216 Matrix Cores** @ 2.1 GHz [2] → **1307.4 TF FP16/BF16**, **2614.9 TF FP8/INT8** [2].
> Use **`v_mfma_f32_16x16x16_*`** for FP16/BF16 GEMM and **`16x16x32`** for FP8 — each reaches the same
> peak as its 32×32 sibling with a smaller C-register footprint (derived below), and AMD's guide finds
> `mfma_16x16` typically faster on MI300X [4]. FP8 is **FNUZ** here [5].

## Concepts

### CDNA3 dense MFMA table (inference-relevant)
Cycles = per instruction on one matrix core, from the ISA's MFMA opcode table [1]. A/B/C = entries per
lane (wave64), i.e. M·K/64, K·N/64 and M·N/64 (derived from the shape).

| Instruction | M×N×K | A,B in | C,D out | Cycles | A/B/C entries | Notes |
|---|---|---|---|---|---|---|
| `v_mfma_f64_16x16x4_f64` | 16×16×4 | FP64 | FP64 | 32 | 1/1/4 | HPC |
| `v_mfma_f32_32x32x2_f32` | 32×32×2 | FP32 | FP32 | 64 | 1/1/16 | |
| `v_mfma_f32_16x16x4_f32` | 16×16×4 | FP32 | FP32 | 32 | 1/1/4 | |
| `v_mfma_f32_32x32x8_f16` | 32×32×8 | FP16 | FP32 | 32 | 4/4/16 | |
| `v_mfma_f32_16x16x16_f16` | 16×16×16 | FP16 | FP32 | 16 | 4/4/4 | **preferred FP16** [4] |
| `v_mfma_f32_32x32x8_bf16` | 32×32×8 | BF16 | FP32 | 32 | 4/4/16 | |
| `v_mfma_f32_16x16x16_bf16` | 16×16×16 | BF16 | FP32 | 16 | 4/4/4 | **preferred BF16** [4] |
| `v_mfma_f32_32x32x16_fp8_fp8` | 32×32×16 | FP8 E4M3 | FP32 | 32 | 8/8/16 | FNUZ |
| `v_mfma_f32_16x16x32_fp8_fp8` | 16×16×32 | FP8 | FP32 | 16 | 8/8/4 | **preferred FP8** [4] |
| `v_mfma_f32_*_fp8_bf8` / `bf8_fp8` / `bf8_bf8` | same | E4M3/E5M2 mix | FP32 | same | same | A,B dtype independent [1] |
| `v_mfma_i32_32x32x16_i8` | 32×32×16 | INT8 | INT32 | 32 | 8/8/16 | quantized |
| `v_mfma_i32_16x16x32_i8` | 16×16×32 | INT8 | INT32 | 16 | 8/8/4 | **preferred INT8** [4] |

CDNA3 has **no** native FP6/FP4 or block-scaled (MX) MFMA — CDNA4 only [5][7] (see
[../cdna4_mi350/matrix_core_blockscale.md](../cdna4_mi350/matrix_core_blockscale.md)). CDNA3 also
supports **SMFMAC** 4:2 structured sparsity, e.g. `v_smfmac_f32_16x16x32_f16` [1], which doubles the
matrix throughput [3].

### Two instructions, same peak
FP16 `16x16x16`@16cyc and `32x32x8`@32cyc both reach 1307.4 TF (the peak formula with the cycles
above; see [peak_tables.md](peak_tables.md)). Choose by **register/LDS pressure and tile fit**:
`16x16x16` carries **4 C-regs/lane** vs **16** for `32x32x8`, so it preserves occupancy. AMD's guide:
for GEMM on MI300X `mfma_16x16` typically outperforms `mfma_32x32`, even for large tiles, and its
Tensile notes credit 16×16's better power efficiency [4].

### Intrinsics (real HIP)
The builtins follow `__builtin_amdgcn_mfma_<CDfmt>_<MxNxK><ABfmt>(a, b, c, cbsz, abid, blgp)` [6].
```cpp
// FP16/BF16 workhorse — each call advances K by 16
using fp16x4 = __attribute__((__vector_size__(4*sizeof(__half)))) __half;
using fp32x4 = __attribute__((__vector_size__(4*sizeof(float)))) float;
fp32x4 acc = {0,0,0,0};
acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag, acc, 0, 0, 0);
// BF16: __builtin_amdgcn_mfma_f32_16x16x16bf16 with bf16x4 operands

// FP8 32x32x16 — A/B packed into 64-bit (cast to long), FNUZ storage
acc16 = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
            *reinterpret_cast<long*>(&a8), *reinterpret_cast<long*>(&b8), acc16, 0,0,0);
// mixed: ..._fp8_bf8 (A=E4M3, B=E5M2). INT8: __builtin_amdgcn_mfma_i32_16x16x32_i8
```
`cbsz/abid/blgp` = broadcast/lane-group modifiers; 0 means no broadcast and the normal layout [6].
Keep `acc` in **AGPRs** for large tiles: AMD's guide sets `amdgpu-mfma-vgpr-form=false` and
`amdgpu-agpr-alloc=256`, at a cost of about 5% of the loop for the epilogue AGPR→VGPR copies [4].

### Numerics (CDNA3-specific)
- **FP8 is FNUZ** [5]: E4M3 bias 8, max ±240, no inf, NaN = `0x80`; E5M2 (BF8) bias 16, max ±57344
  [1]. Re-cast OCP checkpoints; never bit-copy. See
  [../shared/dtype_numerics.md](../shared/dtype_numerics.md).
- A bit-accurate study found that CDNA3's mixed-precision MFMAs (TF32/BF16/FP16/FP8 inputs) round
  down (RD) in their internal summation, a small negative bias that grows with long K [8].
- CDNA3 MFMAs ignore the MODE denorm control and keep input/output denorms [1]; the same study found
  CDNA2's FP16 MFMA flushing input subnormals [8]. Accumulate in FP32/INT32 only.
- **TF32 is present on CDNA3** as MFMA `xf32` (1024 FLOPs/clock/CU, 653.7 TF) [1][2][3] — it moved to
  software emulation via BF16 on CDNA4 [7].

## The levers
1. **`16x16` over `32x32`** for FP16/BF16/FP8/INT8 [4].
2. **FP8 `16x16x32`** for max throughput where accuracy allows.
3. **AGPR accumulators** for big tiles [4].
4. **Swizzle LDS** to the lane map so `ds_read_b128` feeds MFMA conflict-free →
   [memory_hierarchy.md](memory_hierarchy.md).
5. **SMFMAC** only for genuinely 4:2-sparse weights.

## Pitfalls
- **OCP↔FNUZ bit-copy** corrupts FP8 silently (different bias and range [1][7]).
- **Down-converting the accumulator** inside the K-loop.
- **Picking 32×32 for "bigger tiles"** — worse C-register footprint, lower occupancy.

## Verify
- `amd_matrix_instruction_calculator --architecture cdna3 --instruction <name> --detail-instruction`
  → opcode, register usage, execution cycles, FLOPs/CU/cycle, VALU co-execution, ArchVGPR/AccVGPR
  eligibility [9].
- `--get-register` for the exact register and lane of any A/B/C/D element [9].

## Sources
1. AMD, *"AMD Instinct MI300" Instruction Set Architecture* (CDNA3 ISA; §7.1 MFMA tables, §7.2 BF8/FP8 formats) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
2. AMD, *AMD Instinct MI300X Accelerator* data sheet — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
3. AMD, *AMD CDNA 3 Architecture* white paper (Table 1; sparsity) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
4. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* — https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html
5. HIP documentation, *Low precision floating point types* (FP8 support per architecture; FP4/FP6 need gfx950) — https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html
6. AMD GPUOpen, *AMD matrix cores* (AMD lab notes) — https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
7. AMD, *AMD CDNA 4 Architecture* white paper — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
8. P. Xie et al., *Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units* (MMA-Sim), arXiv:2511.10909 (§VI-B) — https://arxiv.org/abs/2511.10909
9. ROCm, *AMD Matrix Instruction Calculator* README — https://github.com/ROCm/amd_matrix_instruction_calculator
