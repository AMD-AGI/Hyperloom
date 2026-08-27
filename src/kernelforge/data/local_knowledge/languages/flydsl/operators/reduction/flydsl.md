---
title: reduction on flydsl — SOTA card
kind: sota_card
operator: reduction
backend: flydsl
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: experimental
updated: 2026-06-09
---

# reduction × flydsl

## TL;DR
FlyDSL does **not** ship a standalone, separately-dispatched reduction op. What the on-box source
provides is the **block-reduce PRIMITIVE** `make_block_reduce_add` / `make_block_reduce_add2` in
[`kernels/reduce.py`](/sgl-workspace/aiter/aiter/ops/flydsl/kernels/reduce.py) — a **wave64**
(`WARP_SIZE = 64`) intra-wave XOR-shuffle reduce, then an **LDS scratch** exchange across `RED_SLOTS`
waves, finished by a second shuffle in wave0. It is the substrate that FlyDSL's fused norm/softmax
kernels are built from (see [[operators/rmsnorm/backends/flydsl]]). Use it only when authoring a fused
FlyDSL kernel that needs a block-wide `Σ`; for a plain reduce use [triton.md](triton.md) /
[hip.md](hip.md) / [ck.md](ck.md).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `make_block_reduce_add(val_f32, scratch)` — general block sum | `aiter/ops/flydsl/kernels/reduce.py` | gfx942/950 | wave64 reduce; **single-wave fast path** (`RED_SLOTS == 1`) skips LDS + barrier | one block-wide Σ inside a fused FlyDSL kernel |
| `make_block_reduce_add2(a,b, scratch0, scratch1)` — two independent sums, one set of barriers | `aiter/ops/flydsl/kernels/reduce.py` | gfx942/950 | reduces two scalars sharing the cross-wave sync (e.g. mean+var) | two reductions over the same row (LayerNorm μ/σ²) |
| `make_block_reduce(val, "max"/"add")` — generic op | `aiter/ops/flydsl/kernels/reduce.py` | gfx942/950 | same wave→LDS→wave shape, neutral = `-inf`/`0` | softmax-style max-or-add reduce |
| (no standalone `flydsl_reduce`) | — | — | no isolated flydsl number | use Triton / HIP / CK |

Real excerpt — the wave64 intra-wave XOR-shuffle reduce that all three builders share
(`make_block_reduce_add`, single-wave fast path):

```python
# Fast path: single-wave block (RED_SLOTS==1) needs no LDS and no barrier.
# After xor-shuffle reduction, all lanes hold the same reduced value.
if RED_SLOTS == 1:
    width_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
    w = arith.as_value(val_f32)
    for sh in [32, 16, 8, 4, 2, 1]:
        off = arith.as_value(arith.constant(sh, type=T.i32()))
        peer = arith.as_value(
            gpu.ShuffleOp(arith.as_value(w), off, width_i32, mode="xor").shuffleResult
        )
        w = arith.as_value(
            arith_ops.AddFOp(arith.as_value(w), peer, fastmath=fm_fast).result
        )
    return w
```
— `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/reduce.py` (`make_block_reduce_add`).

The multi-wave path: lane0 of each wave writes its partial to `scratch_tv[wave_id]` (LDS), a
`gpu.barrier()`, then wave0 loads the `NUM_WAVES = RED_SLOTS` partials (out-of-range lanes selected to
`0.0`), reduces again with the same shuffle ladder, and lane0 writes `scratch[0]`; final `gpu.barrier()`
and all threads read `scratch[0]`.

## Config space / knobs
| knob | where | effect |
|---|---|---|
| `WARP_SIZE` | builder arg (passed `64`) | wave width of the shuffle ladder; AMD wave64 |
| `RED_SLOTS` | builder arg (= number of waves / LDS slots) | `==1` → no-LDS no-barrier fast path; `>1` → LDS scratch + 2 barriers |
| `compute_type` | builder arg | accumulation type; pass `f32` for the sum (see numerics) |
| `fm_fast` | builder arg | fastmath flag on the `AddFOp` reduce steps |
| `scratch_memref` | call arg | LDS scratch tensor (`shape=(RED_SLOTS,)`) for cross-wave exchange |
| inherited FlyDSL knobs | — | `num_warps`, `waves_per_eu`, `SmemAllocator` budget — see [[languages/flydsl/knobs]] |

(`make_block_reduce` uses `BLOCK_SIZE` and derives `NUM_WAVES = ceil(BLOCK_SIZE/64)` itself rather than
taking `RED_SLOTS`.)

## Performance / parity
**No isolated flydsl reduction number exists** — this is a reduce primitive folded into fused aiter/FlyDSL
kernels, not a benchmarked standalone op. Bench it only via the kernel that contains it. For standalone
reduction perf use the Triton/HIP/CK cards.

## Numerics
fp32 accumulate: callers pass `val_f32` and a `compute_type=f32`, and the cross-wave neutral is
`arith.constant(0.0, type=T.f32())`. Reduction order is **wave-shuffle then LDS then wave-shuffle** (a
fixed tree) → bitwise-different from CK/Triton/torch sequential order; compare with fp32 tolerance, not
exact. `fm_fast` enables fastmath on the adds. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- **Not a directly-dispatched standalone op.** It engages only when an enclosing FlyDSL kernel that calls
  the builder is selected (e.g. a fused norm/softmax/MoE stage).
- Author path: `from aiter.ops.flydsl.kernels.reduce import make_block_reduce_add` (and `_add2`,
  `make_block_reduce`), build the closure with `tid`, `gpu`, `arith`, `flir`, `T`, an LDS scratch memref,
  then call `block_reduce_add(val_f32, scratch)` inside your kernel body. See [[languages/flydsl/patterns]].

## Pitfalls & anti-patterns
- ⚠ Reaching for FlyDSL for a *plain* standalone reduce is over-engineering — it is bandwidth-bound and
  Triton/HIP/CK already hit the floor. FlyDSL pays off only when fusing the reduce into a compute kernel.
- The `scratch_memref` must hold ≥ `RED_SLOTS` fp32 slots and be the correct LDS allocation, else the
  cross-wave exchange corrupts.
- Operands must be raw MLIR `Value`s — the source defensively wraps everything in `arith.as_value(...)`
  because `ShuffleOp`/`AddFOp` reject wrapper objects; reuse that pattern.
- `WARP_SIZE` is hard-wired to 64 by callers; do not pass 32.

## Alternatives / cross-links
[triton.md](triton.md) · [hip.md](hip.md) · [ck.md](ck.md) ·
[[operators/rmsnorm/backends/flydsl]] (same reduce primitive, norm context) ·
[[operators/layernorm/backends/flydsl]] · [[operators/softmax/backends/flydsl]] ·
[[languages/flydsl/kernel_families]] · [[languages/flydsl/knobs]].
