---
title: elementwise — numerics & parity
kind: operator_overview
operator: elementwise
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://pytorch.org/docs/stable/generated/torch.clamp.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# elementwise — numerics & parity

Elementwise ops have **no cross-element reduction**, so order-of-summation parity issues (the bane of
`../reduction/numerics.md`) don't apply. The risks are per-element: dtype
rounding, NaN/Inf propagation, denormals, and (for cast) fp8 saturation.

## 1. bf16/fp16 round-trip
PyTorch (and torch's eager kernels) compute pointwise math by **upcasting to fp32**, applying the op,
then rounding to the output dtype. Match this: do arithmetic in fp32 even when in/out are bf16 (Triton's
`tl.bfloat16` is auto-promoted to fp32 in arithmetic; in HIP, load → `float` → compute → `__float2bfloat16`).
A kernel that accumulates a chain in bf16 will drift from the torch reference. `-ffast-math` /
`-fgpu-flush-denormals-to-zero` change rounding/denormal handling — re-check parity if used.

## 2. NaN / Inf and `clamp` / `min` / `max`
- `clamp(x, lo, hi)`: torch is `min(max(x, lo), hi)`. With NaN inputs the result is implementation-order
  dependent (`max(NaN, lo)` propagates NaN in IEEE `fmax`? — AMD `v_max_f32` returns the non-NaN operand,
  unlike `std::max`). If parity on NaN matters, replicate torch's exact `min(max(...))` op order, not a
  single fused `med3`.
- `where(cond, a, b)` is NaN-safe (selects, doesn't compute both) — prefer it over arithmetic masking
  (`a*m + b*(1-m)`) which produces `NaN*0 = NaN`.

## 3. fp8 / int8 saturation (cast-adjacent)
Writing to fp8 saturates to max-normal on overflow (no Inf in e4m3); on **gfx942 use FNUZ**
(`fp8_e4m3_fnuz`, exponent bias 8, no signed zero), on **gfx950 OCP** (`fp8_e4m3`). The wrong dialect is
off by ~2× in scale — a correctness bug, not a rounding one. int8 elementwise must define rounding
(round-half-to-even vs truncate) and clamp to `[-128, 127]`. Detail in
`../cast_fill_copy/numerics.md`.

## 4. Division & reciprocal
`div` via `v_rcp_f32` + Newton step is ~1 ULP; a bare `v_rcp_f32` (`-ffast-math`) is faster but
~2–3 ULP — fine for activations, not for anything feeding a comparison/argmax downstream.

## 5. Fusion changes nothing here (but check the neighbor)
Fusing add/mul/clamp into a GEMM or norm epilogue doesn't change the elementwise math, but it **does**
change where the upcast/rounding boundary sits — a fused `+bias` happens in the GEMM's fp32 accumulator
*before* the bf16 round (more accurate), whereas a separate add rounds twice. This is a parity *gain*, but
note it when matching a reference that does it the unfused way.

## Parity gate
For a standalone elementwise op: bitwise/`atol` parity vs torch eager at the op level is achievable and
expected (no reduction nondeterminism). If it fails, the cause is almost always (a) bf16 not upcasted,
(b) NaN/clamp order, or (c) wrong fp8 dialect — not "GPU nondeterminism."

## Sources
- FNUZ (gfx942) vs OCP (gfx950) fp8 dialects: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- torch.clamp = min(max(x,lo),hi) semantics: https://pytorch.org/docs/stable/generated/torch.clamp.html
- HIP `v_max_f32`/`__float2bfloat16`, denormal flags: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
