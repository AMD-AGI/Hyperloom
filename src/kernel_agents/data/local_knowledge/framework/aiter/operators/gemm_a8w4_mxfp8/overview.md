---
title: gemm_a8w4_mxfp8 — overview
kind: operator_overview
operator: gemm_a8w4_mxfp8
gens: [gfx1250]
dtypes: [mxfp8_e4m3, mxfp4_e2m1, e8m0, bf16]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/gemm_op_a8w4.py
  - ROCm/aiter@b467ce342:csrc/py_itfs_cu/asm_mxfp8fp4gemm.cu
---

# gemm_a8w4_mxfp8

## TL;DR
`gemm_a8w4_mxfp8` is a **gfx1250 (MI450) ASM** GEMM where the **activation is MXFP8 (e4m3)** and the **weight is
MXFP4 (e2m1)** — "a8w4" = 8-bit activation × 4-bit weight. Both operands carry **OCP microscaling e8m0 block
scales, one per 32 K-elements** (`gemm_op_a8w4.py:4-8`). It computes `D[M,N] bf16 = A @ Bᵀ` with the scales applied
inside the matrix core, and auto-selects the kernel variant from `M/N/K` unless an explicit `kernelName` is given.
It is the a8w4 sibling of the a8w8 MXFP8×MXFP8 path that shares the same `.cu` dispatcher.

## What it is
The A8W4 concretization of block-scaled low-precision GEMM (see scaled_quant_gemm/overview.md for the general
block-scaled MFMA contract). The activation is packed 1 byte/elem MXFP8; the weight is packed **2 elems/byte**
MXFP4, so `B` has shape `[N, K/2]`. Per-operand e8m0 scales have shape `[·, K/32]`. Output is bf16. The kernel
uses AMD's **kernarg-preload** ASM launch mode with a packed 76-byte `KernelArgs` struct that must stay
bit-identical to the POC host layout (`asm_mxfp8fp4gemm.cu:35-52`).

## Entry points (API)
| symbol | path:line | signature (abridged) | purpose |
|---|---|---|---|
| `gemm_a8w4_mxfp8` | `aiter/ops/gemm_op_a8w4.py:35` | `(A, B, ScaleA, ScaleB, dtype=bf16, a_preshuffle=True, kernelName="") -> Tensor` | public wrapper; allocates `out[M,N]`, derives `M=A.shape[0]`, `N=B.shape[0]`, calls the ASM op |
| `_mxfp8_mxfp4_gemm_asm` | `aiter/ops/gemm_op_a8w4.py:24` | `(A, B, ScaleA, ScaleB, out, kernelName=None, a_preshuffle=1) -> None` | `@compile_ops("module_mxfp8fp4gemm_asm", fc_name="mxfp8_mxfp4_gemm_asm", ffi_type="ctypes")` |

Operand contract (from the wrapper's own annotations, `gemm_op_a8w4.py:25-49`):
- `A`: `[M, K]` MXFP8 e4m3 (preshuffled when `a_preshuffle=1`).
- `B`: `[N, K/2]` MXFP4 e2m1, **always preshuffled**.
- `ScaleA`: `[M, K/32]` e8m0 (shuffled); `ScaleB`: `[N, K/32]` e8m0 (shuffled).
- `out`: `[M, N]` bf16. `K` is taken from `A` (`A.shape[1] == K`).

`gemm_a8w4_mxfp8` is star-imported into the top-level namespace (`aiter/__init__.py:95` → `aiter.gemm_a8w4_mxfp8`).

## Dispatch / backends
- Backend is **raw ASM HSACO** registered under `module_mxfp8fp4gemm_asm` via ctypes FFI (not a torch-bound op).
- `get_heuristic_kernel(M, N, K, arch_id, b_intype, a_preshuffle, cfgs)` (`asm_mxfp8fp4gemm.cu:56`) picks the best
  registered variant for the shape; passing `kernelName` bypasses the heuristic.
- The `.cu` hosts **two entry points** sharing config/dispatch: `mxfp8_mxfp8_gemm_asm` (a8w8) and
  `mxfp8_mxfp4_gemm_asm` (a8w4), selected by the `b_type` weight-dtype flag (`B_DTYPE_FP8=0`, `B_DTYPE_FP4=1`,
  `asm_mxfp8fp4gemm.cu:27-28`).
- Persistent + cluster shaders do their own tile scheduling; the host only supplies `M/N/K/batch` and launches on
  a fixed cluster grid (`asm_mxfp8fp4gemm.cu:14-17`), block dim 128 (4 waves × 32 threads, `:264`).

## Config / knobs
| knob | where | effect / constraint |
|---|---|---|
| `kernelName` | wrapper arg | force a specific registered variant; `""`/`None` → heuristic |
| `a_preshuffle` | wrapper arg (default `True`) | whether `A` is pre-shuffled to the MFMA layout (weight `B` is always preshuffled) |
| `dtype` | wrapper arg (default bf16) | output dtype of the allocated `out` |
| MX block | `.cu` constant | `MX_SCALE_BLOCK = 32` — one e8m0 scale per 32 K-elems (`asm_mxfp8fp4gemm.cu:29`) |
| K alignment | `.cu` constant | `K_ALIGN = 128` — POC requires `K % 128 == 0` (`asm_mxfp8fp4gemm.cu:30`) |
| `AITER_GPU_ARCHS` | build/env | if no kernel is registered the dispatcher errors, hinting `AITER_GPU_ARCHS=gfx1250` (`asm_mxfp8fp4gemm.cu:183`) |

## Numerics / parity
- e8m0 block scales are applied per 32-K group inside the matrix core; accumulate to fp32, emit bf16 (the wrapper
  allocates `out` as bf16 by default). No perf numbers are stated in-repo for this op.
- Weight is 4-bit (e2m1) so it is the lossy operand — gate accuracy end-to-end against a bf16 dense reference; the
  MX 32-element shared exponent bounds per-block dynamic range.

## Pitfalls
- **gfx1250 only**: this is a POC-silicon ASM path; on any other arch no kernel is registered and the dispatcher
  raises (`asm_mxfp8fp4gemm.cu:183`).
- `B` is `[N, K/2]` (packed 2 MXFP4/byte) and **always preshuffled** — passing an unshuffled or `[N,K]` weight is
  wrong (`gemm_op_a8w4.py:26`, `:37`).
- `K % 128 == 0` is required (`K_ALIGN`, `:30`); scale tensors must be `[·, K/32]` e8m0, not per-tensor floats.
- The 76-byte `KernelArgs` layout is asserted `static_assert(sizeof(KernelArgs)==76)` (`:52`) and must match the
  POC host struct — do not reorder fields.

## Cross-links
- General block-scaled low-precision GEMM contract (E8M0, scaled MFMA, FP4/FP6 rates) →
  operators/scaled_quant_gemm/overview.md (this op is the a8w4 / gfx1250 concretization).
- Input activation quant to MXFP8 and weight pack to MXFP4 → operators/quant_fp4_mxfp/aiter.md.
- a8w8 sibling (MXFP8×MXFP8) shares the same `.cu` → operators/scaled_quant_gemm/aiter.md,
  operators/dense_gemm/backends/opus.md.

## Sources
- on-box `ROCm/aiter@b467ce342`: `aiter/ops/gemm_op_a8w4.py` (`gemm_a8w4_mxfp8:35`, `_mxfp8_mxfp4_gemm_asm:24`,
  operand shape/dtype annotations `:25-49`), `csrc/py_itfs_cu/asm_mxfp8fp4gemm.cu` (header contract `:1-17`,
  `B_DTYPE_*`/`MX_SCALE_BLOCK`/`K_ALIGN` `:27-30`, `KernelArgs` `:35-52`, `get_heuristic_kernel:56`, arch error
  `:183`), `aiter/__init__.py:95`.
