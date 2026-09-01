---
title: HIP low-precision dtype API — fp8/fp6/fp4/bf16/fp16 headers & MFMA vector types
kind: api_reference
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3, fp8_e5m2, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp6_e2m3, fp6_e3m2, fp4_e2m1, bf16, fp16]
regimes: [both]
status: sota
updated: 2026-07-09
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/cpp_language_extensions.html
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# HIP low-precision dtype API

The device headers/classes for the reduced-precision formats CDNA3/CDNA4 matrix cores consume. **Which
encoding is legal on which arch is a correctness gate** (FNUZ on gfx942 vs OCP on gfx950) — mixing them
is a silent ~2× error, not a crash (see
[../skills/optimize/hip_levers/hip_traps.md](../skills/optimize/hip_levers/hip_traps.md)).

## FP8 (E4M3 / E5M2)
```cpp
#include <hip/hip_fp8.h>
// gfx950 (OCP):   __hip_fp8_e4m3,       __hip_fp8_e5m2
// gfx942 (FNUZ):  __hip_fp8_e4m3_fnuz,  __hip_fp8_e5m2_fnuz
// E4M3 = higher precision (inference weights); E5M2 = wider range (gradients)
```
**gfx942 fp8 is FNUZ** (exponent bias differs from OCP by 1). Feeding OCP `e4m3fn` into a gfx942 MFMA is
wrong/unlowerable — normalize to fnuz first.

## FP6 (E2M3 / E3M2) — CDNA4 only
```cpp
#include <hip/hip_fp6.h>
// __hip_fp6_e2m3 (higher precision), __hip_fp6_e3m2 (wider range) + vector variants
```

## FP4 (E2M1) — CDNA4 only
```cpp
#include <hip/hip_fp4.h>
// __hip_fp4_e2m1, __hip_fp4x2_e2m1, __hip_fp4x4_e2m1
// __hip_cvt_float_to_fp4(), __hip_cvt_fp4_to_halfraw()
// saturation via __hip_saturation_t / __HIP_SATFINITE
```

## BF16 / FP16
```cpp
#include <hip/hip_bf16.h>   // __hip_bfloat16, __hip_bfloat162
#include <hip/hip_fp16.h>   // __half, __half2
// __float2bfloat16(), __bfloat162float(), __float2half(), __half2float()
```

## Microscaling (block-scaled MXFP, gfx950)
```cpp
// Scale type: __amd_scale_t (E8M0)
// Storage:    __amd_fp8x2_storage_t, __amd_fp8x8_storage_t, __amd_fp4x2_storage_t
// Scale-aware convert: __amd_cvt_fp8x2_to_floatx2_scale()
// Stochastic rounding: *_sr APIs (require a seed)
// OCP C++ structs: __hipext_ocp_fp8_e4m3, __hipext_ocp_fp8x2_e4m3, __hipext_ocp_fp6x32_e2m3
```
MXFP block-scaled MFMA uses a 32-element E8M0 scale per block — the `v_mfma_scale_*_f8f6f4` path (see
[../skills/optimize/hip_levers/hip_builtins.md](../skills/optimize/hip_levers/hip_builtins.md)).

## MFMA operand vector types
The matrix-core intrinsics take per-lane vectors declared with `vector_size`:
```cpp
using fp32x4 = __attribute__((vector_size(16))) float;   // 4× fp32 — MFMA accumulator (AGPR)
using int32x4 = __attribute__((vector_size(16))) int;    // 4× i32 — buffer SRD
using int32x8 = __attribute__((vector_size(32))) int;    // 8× i32 — packed fp8 MFMA operand
using fp16x4 = __attribute__((vector_size(8)))  _Float16; // 4× fp16 MFMA operand
```
Keep the accumulator in a **stable** `fp32x4` variable across the K-loop so it stays in AGPRs (avoids the
`v_accvgpr_*` spill — LLVM #131954). Use the AMD Matrix Instruction Calculator for the exact
lane→element map per instruction.

## Sources
- HIP C++ extensions (fp8/fp6/fp4/bf16/fp16 headers, conversions): https://rocm.docs.amd.com/projects/HIP/en/latest/reference/cpp_language_extensions.html
- Matrix Core programming (per-lane MFMA operand layout, E8M0 block scale, FNUZ vs OCP): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- amd_matrix_instruction_calculator: https://github.com/ROCm/amd_matrix_instruction_calculator
