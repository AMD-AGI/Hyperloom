---
title: reduction — numerics & parity
kind: operator_overview
operator: reduction
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://github.com/triton-lang/triton/issues/3017
  - https://arxiv.org/pdf/2510.27583
---

# reduction — numerics & parity

Reductions are where **summation order** becomes a correctness concern: floating-point add is **not
associative**, so a tree reduce, a sequential reduce, and an atomic reduce give *different* low-order
bits. This is the central numerics fact for this operator.

## 1. Accumulate in fp32 (always)
bf16/fp16 inputs must accumulate in **fp32** — a bf16 accumulator over `hidden=5120` loses precision fast
and diverges hard from torch. Triton auto-promotes `tl.bfloat16` to fp32 in `tl.sum`; in HIP load → fp32
→ reduce → round. `mean = sum/n` and `L2 = sqrt(Σx²)` finalize in fp32 then round to out dtype.

## 2. Order-dependence → nondeterminism
- **Tree (wave-shuffle + LDS)**: pairwise, well-conditioned, but a *different* order than torch's. Stable
  run-to-run for a fixed launch config.
- **Atomic split** (`atomicAdd`): blocks finish in **nondeterministic order** → bf16/fp32 LSB varies
  *run to run*. Acceptable for sum/mean if a downstream task gate passes; **not** for anything feeding a
  comparison/argmax (a flipped LSB can flip the argmax → different token). Use the **two-call**
  deterministic path when parity is required.
- Don't expect **bitwise** parity vs torch across reduction backends — gate on fp32 `atol`/`rtol`, and a
  task-level eval for the serving path.

## 3. Mixing reduce + scan / two reductions in one kernel
Triton issue #3017: using `tl.sum` and `tl.cumsum` in the *same* kernel produced wrong results vs a torch
reference (observed on multiple archs). Lesson: validate any kernel that combines a reduction with a scan
or a second reduction against torch; don't assume composition is bug-free. Welford (mean+var one pass) is
the safe way to get two stats from one load.

## 4. `max`/`min` and NaN
AMD `v_max_f32`/`v_min_f32` return the **non-NaN** operand (unlike `std::max`/torch which propagate NaN in
`amax`). If a row can contain NaN and you must match torch's NaN-propagating `max`, add an explicit
`isnan` check — otherwise a NaN silently disappears from the reduce. For softmax row-max this is usually
fine (NaN there is already a bug upstream), but flag it.

## 5. The 45%-of-peak reality (context)
Reductions are bandwidth-bound; MI300X sustains ~81% of HBM peak on a clean reduce but real fused
norms/softmax see less. Don't quote 5.3 TB/s as achievable; the BabelStream-class ceiling is ~4.3 TB/s,
and arithmetic-heavy reductions (L2 with sqrt, Welford) sit below that.

## Parity gate
fp32 `atol` vs torch eager; for the serving path that includes an atomic-split sum feeding logits, run a
greedy/temp=0 task eval — a nondeterministic-order sum can flip a borderline argmax. Prefer the
deterministic two-call path for anything on the logits/sampling critical path.

## Sources
- fp add non-associativity, wave reduce order, v_max NaN behavior: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- tl.sum + tl.cumsum same-kernel wrong results: https://github.com/triton-lang/triton/issues/3017
- ~45% of peak FLOPs / BW reality on MI300X: https://arxiv.org/pdf/2510.27583
