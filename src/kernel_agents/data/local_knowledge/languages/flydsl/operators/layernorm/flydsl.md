---
title: layernorm on flydsl — SOTA card
kind: sota_card
operator: layernorm
backend: flydsl
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: experimental
updated: 2026-06-09
---

# layernorm × flydsl

## TL;DR
The on-box FlyDSL source provides a **reduce PRIMITIVE**, not a standalone tuned LayerNorm op — no
`flydsl_layernorm` kernel is confirmed in the source. LayerNorm needs **two row statistics** (μ and σ²),
and the relevant building block is `make_block_reduce_add2` in
[`kernels/reduce.py`](/sgl-workspace/aiter/aiter/ops/flydsl/kernels/reduce.py): it reduces **two
independent scalars** (e.g. `Σx` and `Σx²`) across a wave64 block while **paying the cross-wave barriers
only once**. For an actual LayerNorm use [aiter.md](aiter.md) (CK/asm `layernorm2d_fwd`) /
[triton.md](triton.md) / [hip.md](hip.md); reach for FlyDSL only when hand-authoring a fused FLIR kernel
whose epilogue contains a LayerNorm-style mean+var reduce.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `make_block_reduce_add2(Σx, Σx², s0, s1)` — two-statistic block reduce (μ + σ²) | `aiter/ops/flydsl/kernels/reduce.py` | gfx942/950 | "NOT pair-reduce: reduces two independent scalars but shares the same cross-wave sync so we only pay the barriers once" (source docstring) | mean+var in one barrier set inside a fused FlyDSL kernel |
| `make_block_reduce_add(val_f32, scratch)` — single block sum | `aiter/ops/flydsl/kernels/reduce.py` | gfx942/950 | wave64 reduce; single-wave fast path skips LDS | two-pass mean-then-var if not using `_add2` |
| (no standalone `flydsl_layernorm`) | — | — | no isolated flydsl number | use aiter CK/asm / Triton / HIP |

Real excerpt — `make_block_reduce_add2` reduces both scalars with one set of barriers (single-wave
fast path returns both wave-reductions directly):

```python
def block_reduce_add2(val0_f32, val1_f32, scratch0_memref, scratch1_memref):
    # Single-wave block: no LDS/no barrier, just two wave reductions.
    if RED_SLOTS == 1:
        return _wave_reduce_add(val0_f32), _wave_reduce_add(val1_f32)
    ...
    # lane0 writes per-wave partials into LDS for both sums.
    if is_lane0:
        wave_idx = arith_ops.IndexCastOp(T.index(), wave_i32).result
        red_idx = flir.crd2idx(flir.make_coord(wave_idx), layout_red)
        scratch0_tv[red_idx] = w0
        scratch1_tv[red_idx] = w1
    gpu.barrier()
```
— `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/reduce.py` (`make_block_reduce_add2`).

For a LayerNorm row you would feed `val0 = Σx` and `val1 = Σx²`, then finalize
`μ = Σx/N`, `σ² = Σx²/N − μ²`, `y = (x−μ)·rsqrt(σ²+ε)·γ + β` in the enclosing kernel (the finalize math
is **not** in the reduce primitive — only the two-sum reduction is).

## Config space / knobs
| knob | where | effect |
|---|---|---|
| `WARP_SIZE` | builder arg (64) | wave64 shuffle width |
| `RED_SLOTS` | builder arg (wave count) | `==1` → no-LDS fast path; `>1` → 2 LDS scratch buffers + barriers |
| `compute_type` / `f32` operands | call | fp32 μ/σ² accumulate |
| `fm_fast` | builder arg | fastmath on the reduce adds |
| `scratch0_memref`, `scratch1_memref` | call | one LDS buffer per statistic |
| inherited FlyDSL knobs | — | `num_warps`, `SmemAllocator` budget — [[languages/flydsl/knobs]] |

## Performance / parity
**No isolated flydsl LayerNorm number exists.** This is a reduce primitive folded into fused
aiter/FlyDSL kernels, not a benchmarked standalone LayerNorm. For LayerNorm perf use the aiter/Triton/HIP
cards.

## Numerics
fp32 accumulate for both μ and σ² (callers pass `*_f32`, cross-wave neutral is f32 `0.0`). The
two-statistic (`_add2`) path is mathematically a two-pass-style mean/var built from Σx and Σx² — watch
catastrophic cancellation in `σ² = Σx²/N − μ²` (Welford is the safer alternative but is **not** in the
on-box reduce primitive). Reduction order is the fixed wave→LDS→wave tree → bitwise-different from
CK/Triton; compare with fp32 tolerance. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- **Not a directly-dispatched standalone op.** No `flydsl` libtype LayerNorm is confirmed in source; the
  primitive engages only inside a hand-authored fused FLIR kernel that calls it.
- Author path: `from aiter.ops.flydsl.kernels.reduce import make_block_reduce_add2`, build with `tid`,
  `gpu`, `arith`, `flir`, `T`, two LDS scratch memrefs, then call inside the kernel body and finalize the
  norm math yourself. See [[languages/flydsl/patterns]].

## Pitfalls & anti-patterns
- ⚠ There is no tuned standalone FlyDSL LayerNorm in the source — do not claim one. For a plain LayerNorm,
  aiter CK/asm or Triton is the path; LayerNorm is bandwidth-bound and they hit the floor.
- `_add2` is **not** a pair/2-vector reduce of one tensor — it reduces two *separate* scalars; feed Σx and
  Σx², not a width-2 vector.
- Allocate **two** LDS scratch buffers (`scratch0`, `scratch1`), each ≥ `RED_SLOTS` fp32 slots.
- Normalize operands to raw MLIR `Value` (`arith.as_value`) as the source does.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [hip.md](hip.md) · [vllm_kernels.md](vllm_kernels.md) ·
[miopen.md](miopen.md) · [[operators/rmsnorm/backends/flydsl]] ·
[[operators/reduction/backends/flydsl]] · [[operators/softmax/backends/flydsl]] ·
[[languages/flydsl/kernel_families]] · [[languages/flydsl/knobs]].
