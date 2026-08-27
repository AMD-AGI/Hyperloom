---
title: MFMA intrinsics — __builtin_amdgcn_mfma_* and the block-scaled CDNA4 variant
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1, int8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://github.com/ROCm/amd_matrix_instruction_calculator
  - https://github.com/llvm/llvm-project/pull/116723
---

# MFMA intrinsics

## TL;DR
MFMA = **Matrix Fused Multiply-Add**, `D = A·B + C`, a **wave-level** op: all 64 lanes cooperate to
compute one M×N×K block with A/B/C/D fragments scattered across lane registers. Always accumulate in
32-bit (f32/i32/f64). Prefer the **intrinsic** over inline asm (the SW pipeliner only recognizes
intrinsic MFMA). On CDNA4, a new **block-scaled** family (`__builtin_amdgcn_mfma_scale_f32_*_f8f6f4`)
adds per-32-element E8M0 scaling for MXFP4/6/8. Default to **16×16×16** (higher achievable FLOPs than
32×32 on MI300X). See [register_alloc.md](register_alloc.md) for fragment placement.

## Core concepts — intrinsic form
```
d_reg = __builtin_amdgcn_mfma_<ODType>_<M>x<N>x<K><InDType>(a_reg, b_reg, c_reg, cbsz, abid, blgp);
```
`ODType` = accum type (f32/i32/f64). `InDType` = A/B type. `cbsz, abid, blgp` = broadcast/block-select
flags — set `0,0,0` for plain GEMM. (For SMFMAC, `cbsz/abid` instead select the compression-index half —
see [raw_asm.md](raw_asm.md) §SMFMAC.)

### CDNA3 dense MFMA table (LLM-relevant)
| Intrinsic | M×N×K | A/B→C | cycles | A regs/lane | B | C | note |
|---|---|---|---|---|---|---|---|
| `..._f32_32x32x8f16` | 32×32×8 | f16→f32 | 32 | 4 (fp16x4) | 4 | 16 | |
| `..._f32_16x16x16f16` | 16×16×16 | f16→f32 | 16 | 4 | 4 | 4 | **prefer** |
| `..._f32_32x32x8bf16` | 32×32×8 | bf16→f32 | 32 | 4 | 4 | 16 | |
| `..._f32_16x16x16bf16` | 16×16×16 | bf16→f32 | 16 | 4 | 4 | 4 | **prefer** |
| `..._f32_32x32x16_fp8_fp8` | 32×32×16 | fp8→f32 | 32 | 8 (fp8x8) | 8 | 16 | 2× bf16 K-density |
| `..._f32_16x16x32_fp8_fp8` | 16×16×32 | fp8→f32 | 16 | 8 | 8 | 4 | 2× bf16 K-density |
| `..._i32_16x16x32_i8` | 16×16×32 | i8→i32 | 16 | 8 | 8 | 4 | int |
| `..._f64_16x16x4f64` | 16×16×4 | f64→f64 | 64 | 1 | 1 | 4 | HPC |

fp8 e4m3/e5m2 can be **mixed**: `..._fp8_bf8` / `..._bf8_fp8`. fp8 operands are cast to `long` (64-bit)
for the intrinsic:
```cpp
c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8((long)a, (long)b, c, 0,0,0);
```
MI300X fp8 is the gfx942-only **fnuz** (NANOO) encoding (different bias from OCP) — match dequant scale.

> **Critical MI300X fact:** `mfma_16x16x16` usually **beats** `mfma_32x32x8` even at large GEMM sizes.
> The 32×32 op has higher *software* efficiency (bigger payload, fewer instructions) but draws more power
> → clocks lower → lower *max-achievable* FLOPs. Default to 16×16×16; only test 32×32×8 (ROCm
> Max-Achievable-FLOPs Part 2).

### CDNA4 block-scaled MFMA (gfx950, ROCm ≥7.0)
A new family with **block exponent scaling** for MXFP formats — different syntax from classic MFMA:
```
d = __builtin_amdgcn_mfma_scale_f32_MxNxK_f8f6f4(
        a, b, c, Atype, Btype, OPSEL_A, scale_a, OPSEL_B, scale_b);
```
`Atype`/`Btype` codes: `0`=E4M3(fp8), `1`=E5M2(bf8), `2`=E2M3(fp6), `3`=E3M2(bf6), `4`=E2M1(fp4) — the
`f8f6f4` suffix means inputs may **mix** these formats. Scale matrices `scale_a`/`scale_b` are **E8M0**
(one 8-bit exponent per 32-element block); the scale is applied after the dot product, before
accumulate. Shapes: `16x16x128` and `32x32x64`. At LLVM level: `llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4`
(VOP3PX encoding bundling the pre-scale `v_mfma_ld_scale_b32`). The compiler shrinks the operand vector
when the format needs <8 bit (e.g. fp6 → v6i32) — PR #117047.

**Scale-layout caveat:** the scale layout global memory delivers is **not** the layout the scaled-MFMA
consumes, and no instruction reads scales register→MFMA-layout directly. You need a 3-step pipeline:
Global Read scales → LDS Write (convert layout) → LDS Read → feed scaled MFMA.

## The levers
- **Pick the smallest MFMA** (16×16) for scheduling flexibility *and* power-limited FLOPs; HipKittens
  also defaults register tiles to the smallest MFMA for maximal scheduling control, mixing 16×16×32 and
  32×32×16 only where the algorithm benefits.
- **fp8/fp4**: double/quadruple K-density per MFMA → fewer instructions per K, but watch operand reg
  count (8 regs/lane fp8).
- **`blgp`/`cbsz`/`abid`** for A-broadcast tricks (e.g. broadcasting one A block across N) — rare; check
  the calculator before using (they conflict with SMFMAC index selection).

## Pitfalls
- Putting MFMA in inline asm → `SchedGroupMask` can't see it → no SW pipelining. Use the intrinsic.
- Wrong fragment placement → silent wrong answer. Use the calculator (below), never guess lane order.
- fp8 fnuz vs OCP encoding mismatch → garbage. CDNA3 = fnuz; CDNA4 OCP fp8/MXFP.
- Block-scaled MFMA on gfx942 → unsupported (gfx950 + ROCm ≥7.0 only).

## Verify
```bash
./matrix_calculator.py --architecture cdna3 --instruction v_mfma_f32_16x16x16_bf16 --detail-instruction
./matrix_calculator.py --architecture cdna3 --instruction v_mfma_f32_32x32x8_f16 --register-layout --A-matrix
./matrix_calculator.py --architecture cdna4 --instruction v_mfma_scale_f32_16x16x128_f8f6f4 --detail-instruction
# output Vx{y}.z : x=reg offset, y=lane, .z=sub-register
```

## Sources
- AMD CDNA3 ISA §7 Matrix Arithmetic (MFMA encodings, cycles): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- AMD CDNA4 ISA §7 (block-scaled MFMA, E8M0, f8f6f4): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- Matrix Core Programming CDNA3/CDNA4 (intrinsic forms, fp8 examples, mfma_scale syntax, scale pipeline): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- amd_matrix_instruction_calculator (per-instruction M/N/K, cycles, register/lane layouts): https://github.com/ROCm/amd_matrix_instruction_calculator
- LLVM PR #116723 (define v_mfma_f32_{16x16x128|32x32x64}_f8f6f4) & #117047 (shrink regs by format): https://github.com/llvm/llvm-project/pull/116723 ; https://github.com/llvm/llvm-project/pull/117047
- ROCm Blog — Measuring Max-Achievable FLOPs Part 2 (16×16 vs 32×32 power/clock): https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html
