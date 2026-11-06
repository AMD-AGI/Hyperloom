---
title: CDNA3 / MI300X (gfx942) — Matrix Core / MFMA
kind: hardware
gens: [gfx942]
dtypes: [fp64, fp32, tf32, bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, int8]
regimes: [both]
updated: 2026-06-08
---

# CDNA3 / MI300X (gfx942) — Matrix Core / MFMA

> Cross-gen concepts (naming, lane mapping, peak formula, AGPR) live in
> [../shared/matrix_core_mfma_smfmac.md](../shared/matrix_core_mfma_smfmac.md). This file is the
> CDNA3-specific instruction table, intrinsics, and numerics.

## TL;DR
> 304 CU × 4 = **1216 Matrix Cores** @ 2.1 GHz → **1307 TF FP16/BF16**, **2615 TF FP8/INT8**.
> Use **`v_mfma_f32_16x16x16_*`** for FP16/BF16 GEMM and **`16x16x32`** for FP8 — both hit peak with
> a smaller C-register footprint than the 32×32 variants. FP8 is **FNUZ** here.

## Concepts

### CDNA3 dense MFMA table (inference-relevant)
Cycles = per-instruction on one matrix core. A/B/C regs = entries per lane (wave64).

| Instruction | M×N×K | A,B in | C,D out | Cycles | A/B/C regs | Notes |
|---|---|---|---|---|---|---|
| `v_mfma_f64_16x16x4_f64` | 16×16×4 | FP64 | FP64 | 64 | 1/1/4 | HPC |
| `v_mfma_f32_32x32x2_f32` | 32×32×2 | FP32 | FP32 | 64 | 1/1/16 | |
| `v_mfma_f32_16x16x4_f32` | 16×16×4 | FP32 | FP32 | 32 | 1/1/4 | |
| `v_mfma_f32_32x32x8_f16` | 32×32×8 | FP16 | FP32 | 32 | 4/4/16 | |
| `v_mfma_f32_16x16x16_f16` | 16×16×16 | FP16 | FP32 | 16 | 4/4/4 | **preferred FP16** |
| `v_mfma_f32_32x32x8_bf16` | 32×32×8 | BF16 | FP32 | 32 | 4/4/16 | |
| `v_mfma_f32_16x16x16_bf16` | 16×16×16 | BF16 | FP32 | 16 | 4/4/4 | **preferred BF16** |
| `v_mfma_f32_32x32x16_fp8_fp8` | 32×32×16 | FP8 E4M3 | FP32 | 32 | 8/8/16 | FNUZ |
| `v_mfma_f32_16x16x32_fp8_fp8` | 16×16×32 | FP8 | FP32 | 16 | 8/8/4 | **preferred FP8** |
| `v_mfma_f32_*_fp8_bf8` / `bf8_fp8` / `bf8_bf8` | same | E4M3/E5M2 mix | FP32 | — | — | A,B dtype independent |
| `v_mfma_i32_32x32x16_i8` | 32×32×16 | INT8 | INT32 | 32 | — | quantized |
| `v_mfma_i32_16x16x32_i8` | 16×16×32 | INT8 | INT32 | 16 | — | **preferred INT8** |

CDNA3 has **no** native FP6/FP4 or block-scaled (MX) MFMA — CDNA4 only (see
[../cdna4_mi350/matrix_core_blockscale.md](../cdna4_mi350/matrix_core_blockscale.md)). CDNA3 also
supports **SMFMAC** 4:2 structured sparsity (~2× throughput), e.g. `v_smfmac_f32_16x16x32_f16`.

### Two instructions, same peak
FP16 `16x16x16`@16cyc and `32x32x8`@32cyc both reach 1307 TF. Choose by **register/LDS pressure and
tile fit**: `16x16x16` carries **4 C-regs/lane** vs **16** for `32x32x8`, so it preserves occupancy.
On MI300X, 16×16 wins more often.

### Intrinsics (real HIP)
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
`cbsz/abid/blgp` = broadcast/lane-group modifiers; 0 for ordinary GEMM. Keep `acc` in **AGPRs** for
large tiles (`-mllvm -amdgpu-mfma-vgpr-form=false -mllvm -amdgpu-agpr-alloc=256`).

### Numerics (CDNA3-specific)
- **FP8 is FNUZ** (E4M3FNUZ bias 8, ±240, no inf; E5M2FNUZ bias 16). Re-cast OCP checkpoints; never
  bit-copy. See [../shared/dtype_numerics.md](../shared/dtype_numerics.md).
- FP16/BF16/TF32 MFMA use an asymmetric **round-down (RD)** accumulate path (small systematic bias on
  long K); the **FP8** path was adjusted to mitigate. CDNA3 **fully supports subnormals** (unlike
  CDNA2). Accumulate in FP32/INT32 only.
- **TF32 is present on CDNA3** (653.7 TF, vector/emulated path) — but **removed on CDNA4**.

## The levers
1. **`16x16` over `32x32`** for FP16/BF16/FP8/INT8.
2. **FP8 `16x16x32`** for max throughput where accuracy allows.
3. **AGPR accumulators** for big tiles.
4. **Swizzle LDS** to the lane map so `ds_read_b128` feeds MFMA conflict-free →
   [memory_hierarchy.md](memory_hierarchy.md).
5. **SMFMAC** only for genuinely 4:2-sparse weights.

## Pitfalls
- **OCP↔FNUZ bit-copy** corrupts FP8 silently.
- **Down-converting the accumulator** inside the K-loop.
- **Picking 32×32 for "bigger tiles"** — worse C-register footprint, lower occupancy.

## Verify
- `amd_matrix_instruction_calculator --architecture cdna3 --instruction <name> --detail-instruction`
  → opcode, cycles, FLOPs/CU/cycle, VALU co-exec, GPR counts, AGPR eligibility (authoritative).
- `--get-register` for exact `Vx{lane}.sub` of any A/B/C/D element.
