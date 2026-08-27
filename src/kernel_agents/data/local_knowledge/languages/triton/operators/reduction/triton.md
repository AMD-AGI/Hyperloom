---
title: reduction on Triton — SOTA card
kind: sota_card
operator: reduction
backend: triton
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://triton-lang.org/main/python-api/triton.language.html
  - https://github.com/triton-lang/triton/issues/3017
---

# reduction × triton

## TL;DR
`tl.sum`/`tl.max`/`tl.min` lower to a **wave64 shuffle reduce + LDS combine** automatically — Triton is
the SOTA authoring path and Inductor's codegen target. Set `BLOCK = next_pow2(axis)` so the reduced dim
fills the 64-lane wave, `num_warps` to control how many waves combine via LDS, `num_stages=1`. For
**low-output-count** shapes, do a 2-D grid + `tl.atomic_add` (the reduction analogue of GEMM `SPLIT_K`) to
fill 304 CUs.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| row reduce (`tl.sum/max`, one program per row) | [`../../../triton_levers/patterns.md`] §5 | gfx942/950, bf16→fp32 | bandwidth-bound, **~3.5–4.3 TB/s** input read @ MI300X | many rows (`[tokens, hidden]`) |
| split reduce (2-D grid + `tl.atomic_add`) | this card | gfx942/950 | fills CUs at low output count (CU util ≫ naive) | few output rows / huge axis |
| Inductor-fused reduce | [`../../../backends/pytorch_inductor/overview.md`] | gfx942/950 | auto-fused with pointwise | `torch.compile` graphs |

```python
@triton.autotune([triton.Config({}, num_warps=nw, num_stages=1) for nw in (2,4,8)], key=["n"])
@triton.jit
def row_sum(x_ptr, out_ptr, sr, n, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + r*sr + cols, mask=cols < n, other=0.0).to(tl.float32)  # fp32 acc
    tl.store(out_ptr + r, tl.sum(x, 0))          # -> wave64 reduce + LDS combine
# grid = (rows,);  BLOCK = next_pow2(n)
```

## Config space / knobs
- `BLOCK = next_pow2(axis)`: reduced dim must be wave-full (a dim < 64 wastes lanes).
- `num_warps`: 2/4 typical; more waves → more LDS combine steps. Memory-bound → not 8.
- `num_stages=1` (no K-loop). `waves_per_eu=3/4` to hide load latency.
- **split**: 2-D grid `(rows, splits)`, each program reduces a chunk of the axis, `tl.atomic_add` into the
  output (zero-init first). Use when `rows < ~1024`.
- `knobs.amd.use_buffer_ops=1` for cheap masked tail loads.

## Numerics / parity
fp32 accumulate (bf16 auto-promoted). `tl.atomic_add` split = **nondeterministic order** → bf16 LSB
varies run-to-run; use single-program reduce or a 2nd-kernel combine for parity-critical paths.
⚠ **Don't mix `tl.sum` and `tl.cumsum` in one kernel** (issue #3017 — wrong results). See [numerics.md](numerics.md).

## Integration (rebind seam)
Usually you let **Inductor generate** the reduce (fused with pointwise). Authored kernel matters for a
custom split strategy or a fused reduce the compiler won't form; register as a torch custom op to keep it
opaque.

## Pitfalls & anti-patterns
- `BLOCK < axis` (axis doesn't fit one program) → need a loop over chunks + carry, or split — a single
  program can't reduce an arbitrarily long axis.
- Reduced dim not pow2 / < 64 → wasted lanes.
- Atomic split on a parity-critical reduce → flipped downstream argmax.
- `num_warps=8` on a BW reduce → spill, no gain.

## How to verify
`TRITON_PRINT_AUTOTUNING=1`; GB/s = `input_bytes/time` vs ~4.3 TB/s; for split, rocprof CU utilization
≫ naive; fp32 atol vs torch.

## Alternatives / cross-links
hip (explicit wave/LDS/atomic) · [composable_kernel.md](ck.md)
(library instances) · [tuning.md](tuning.md) · [fusion.md](fusion.md).

## Sources
- fused softmax/reduce wave template, BLOCK=next_pow2: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- tl.sum/tl.max/tl.cumsum semantics: https://triton-lang.org/main/python-api/triton.language.html
- tl.sum + tl.cumsum same-kernel bug: https://github.com/triton-lang/triton/issues/3017
