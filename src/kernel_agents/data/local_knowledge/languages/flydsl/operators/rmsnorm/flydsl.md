---
title: rmsnorm on flydsl — SOTA card
kind: sota_card
operator: rmsnorm
backend: flydsl
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: competitive
updated: 2026-06-08
---

# rmsnorm × flydsl

## TL;DR
FlyDSL does **not** ship a standalone RMSNorm kernel as a product — instead it provides the **wave64
block-reduce primitive** (`kernels/reduce.py`) that softmax / layernorm / rmsnorm-style kernels are built
from, and folds RMSNorm-style reductions into its **fused** kernels (e.g. `silu_and_mul_fq`, MoE stages).
Use FlyDSL for RMSNorm only when you need instruction-level control (direct-to-LDS, hand-scheduled
reduce) *and* you're fusing it into a larger FlyDSL kernel; for a plain norm, [aiter.md](aiter.md) CK/asm
or [triton.md](triton.md) is the path.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `reduce.py` `make_block_reduce` (wave64 reduce primitive) | `aiter/ops/flydsl/kernels/reduce.py` | gfx942/950 | wave64-aware (`NUM_WAVES`, `ds_bpermute`/swizzle LDS exchange) | building a fused norm inside a FlyDSL kernel |
| RMSNorm-style reduce inside `silu_and_mul_fq` / MoE-2stage | `aiter/ops/flydsl/kernels/silu_and_mul_fq.py`, `moe_gemm_2stage.py` | gfx942/950, fp8/fp4 out | part of the Kimi-K2.5 fused-MoE **+162% throughput** (vendor) | norm folded into MoE/activation epilogue |
| (no standalone `flydsl_rmsnorm`) | — | — | — | use aiter CK/asm or Triton |

## Config space / knobs
- `make_block_reduce(NUM_WAVES, compute_type, ...)` → returns a `block_reduce(val, op)`; the reduce uses
  per-wave `__shfl`-equivalent then an LDS exchange across `NUM_WAVES` (XOR-swizzle / `ds_bpermute`).
- Inherited FlyDSL knobs: `num_warps`, `waves_per_eu`, `maxnreg`, `SmemAllocator` budget
  (65536 B gfx942 / 163840 B gfx950), `use_async_copy` (gfx950 direct-to-LDS).
- fp32 compute_type for the Σx² accumulate (pass `compute_type=f32`).

## Numerics / parity
fp32 reduce via `compute_type`; γ promote in the FLIR layout. Reduction order is hand-built (wave then
LDS) → differs from CK/Triton; greedy re-gate. fp8 output uses the FlyDSL quant path (fnuz on gfx942). See
[../numerics.md](../numerics.md).

## Integration (rebind seam)
- Not a serving rebind seam on its own — it engages only when a FlyDSL kernel that *contains* a norm
  reduce is selected (e.g. fused-MoE via aiter `tuned_gemm` `flydsl` libtype, or `silu_and_mul_fq`).
- Author path: import the reduce builder from `aiter.ops.flydsl.kernels.reduce` and compose with
  `SmemAllocator` + ROCDL ops. See [[languages/flydsl/patterns]].

## Pitfalls & anti-patterns
- ⚠ Reaching for FlyDSL for a *plain* RMSNorm = over-engineering; the op is bandwidth-bound and CK/Triton
  already hit the floor. FlyDSL pays off only when fusing into a compute kernel.
- gfx942: async-copy/direct-to-LDS is off by default (`KERNEL_ASYNC_COPY = arch != gfx942`) — the reduce
  uses the synchronous LDS path there.

## How to verify
Confirm via `AITER_LOG_MORE=1` that the dispatched kernel is a `flydsl`-libtype kernel containing the
fused norm; rocprofv3 kernel name; greedy parity.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [hip.md](hip.md) ·
[[languages/flydsl/kernel_families]] (silu_and_mul_fq, reduce) · [[backends/aiter/flydsl_path]].
