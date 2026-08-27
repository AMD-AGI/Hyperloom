---
title: softmax on flydsl — SOTA card
kind: sota_card
operator: softmax
backend: flydsl
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: experimental
updated: 2026-06-09
---

# softmax × flydsl

## TL;DR
The on-box FlyDSL source ships a softmax **reduce PRIMITIVE**, not a standalone tuned softmax op. Softmax
needs a row **max** and a row **sum**; the building blocks in
[`kernels/reduce.py`](/sgl-workspace/aiter/aiter/ops/flydsl/kernels/reduce.py) are the per-lane vector
reductions `reduce_vec_max` (`vector.reduction(..., "maxnumf", ...)`) and `reduce_vec_sum`
(`vector.reduction(..., "add", ...)`) over `VEC_WIDTH`, plus the block-wide `make_block_reduce(val, "max"
/ "add")` which combines lanes across a wave64 block (intra-wave XOR shuffle → LDS → wave0 shuffle, with
neutral = `-inf` for max, `0` for add). These are the max/sum substrate of an online-softmax body. For an
actual softmax use [triton.md](triton.md) (aiter online-softmax) / [hip.md](hip.md); reach for FlyDSL only
when fusing softmax into a hand-authored FLIR kernel.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `reduce_vec_max(vec, VEC_WIDTH, compute_type, vector)` — per-lane max over a vector | `aiter/ops/flydsl/kernels/reduce.py` | gfx942/950 | `vector.reduction("maxnumf")`; `VEC_WIDTH==1` shortcut extracts elem 0; bf16 max avoids fastmath | the `m = max(x)` step of softmax |
| `reduce_vec_sum(vec, VEC_WIDTH, compute_type, vector, fm_fast)` — per-lane Σ over a vector | `aiter/ops/flydsl/kernels/reduce.py` | gfx942/950 | `vector.reduction("add", fastmath=fm_fast)` | the `Σ exp(x−m)` step |
| `make_block_reduce(val, "max")` / `(val, "add")` — block-wide combine | `aiter/ops/flydsl/kernels/reduce.py` | gfx942/950 | wave64 shuffle → LDS → wave0; neutral `-inf`/`0` | combine per-lane partials across the block row |
| (no standalone `flydsl_softmax`) | — | — | no isolated flydsl number | use Triton / HIP |

Real excerpt — the two per-lane vector reductions that feed the online-softmax max/sum:

```python
def reduce_vec_max(vec_val, *, VEC_WIDTH, compute_type, vector):
    if VEC_WIDTH == 1:
        return vector.extract(vec_val, static_position=[0], dynamic_position=[])
    # Avoid fastmath on bf16 max reduction; some backends can fail to select.
    ...
    return vector.reduction(compute_type, "maxnumf", vec_val)


def reduce_vec_sum(vec_val, *, VEC_WIDTH, compute_type, vector, fm_fast):
    if VEC_WIDTH == 1:
        return vector.extract(vec_val, static_position=[0], dynamic_position=[])
    ...
    return vector.reduction(compute_type, "add", vec_val, fastmath=fm_fast)
```
— `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/reduce.py` (`reduce_vec_max`, `reduce_vec_sum`).

The block-wide combine uses `make_block_reduce(val, "max"/"add")`: per-lane partials go through the wave64
XOR-shuffle ladder, lane0 of each wave writes to `s_red[wave_id]` (LDS), then wave0 reduces `NUM_WAVES`
partials with the same ladder using `c_neg_inf` (max) or `c_zero` (add) as the out-of-range neutral.

## Config space / knobs
| knob | where | effect |
|---|---|---|
| `VEC_WIDTH` | `reduce_vec_*` arg | per-lane vector width; `==1` extracts element 0 (no reduction) |
| `compute_type` | `reduce_vec_*` arg | dtype of the `vector.reduction` result (fp32 for stable exp/accum) |
| `vector` | `reduce_vec_*` arg | the MLIR vector-dialect handle used to emit the reduction |
| `fm_fast` | `reduce_vec_sum` / block builders | fastmath on the add reduce (explicitly **off** for bf16 max) |
| `BLOCK_SIZE` | `make_block_reduce` | derives `NUM_WAVES = ceil(BLOCK_SIZE/64)` for the LDS exchange |
| `c_neg_inf` / `c_zero` | `make_block_reduce` args | neutral element for max / add across waves |
| inherited FlyDSL knobs | — | `num_warps`, `SmemAllocator` — [[languages/flydsl/knobs]] |

## Performance / parity
**No isolated flydsl softmax number exists.** This is a reduce primitive (vector reductions + block
combine) folded into fused aiter/FlyDSL kernels, not a benchmarked standalone softmax. For standalone
softmax perf use the Triton/HIP cards; for the dominant case bench attention (softmax lives inside FMHA).

## Numerics
Max-subtraction is the caller's job; `make_block_reduce` supplies the row `max` and row `sum` and uses
`-inf`/`0` neutrals so partial/out-of-range lanes don't perturb the result. `compute_type` should be fp32
for the exp/accumulate. **bf16 max deliberately avoids fastmath** (source comment: "some backends can fail
to select"). Reduction order is the fixed wave→LDS→wave tree → bitwise-different from Triton/torch;
compare with fp32 tolerance. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- **Not a directly-dispatched standalone op.** The primitives engage only inside a hand-authored fused
  FLIR softmax/attention kernel that calls them.
- Author path: `from aiter.ops.flydsl.kernels.reduce import reduce_vec_max, reduce_vec_sum,
  make_block_reduce`; compute per-lane max/sum over the vector, then combine across the block with
  `make_block_reduce(..., "max")` / `(..., "add")`. See [[languages/flydsl/patterns]].

## Pitfalls & anti-patterns
- ⚠ No tuned standalone FlyDSL softmax exists in the source — do not claim one. For routing/sampling/vocab
  softmax use Triton/HIP; the dominant softmax is fused inside attention.
- Keep `fm_fast` **off** for the bf16 `maxnumf` reduction (matches the source guard) or codegen may fail
  to select.
- The vector reductions expect a raw MLIR `Value` — the source calls `arith.as_value(vec_val)` before
  `vector.reduction`; reuse that.
- `make_block_reduce` must be passed the correct neutral (`c_neg_inf` for max, `c_zero` for add) and an LDS
  scratch sized for `NUM_WAVES`.

## Alternatives / cross-links
[triton.md](triton.md) · [hip.md](hip.md) · [aiter.md](aiter.md) ·
[[operators/rmsnorm/backends/flydsl]] · [[operators/reduction/backends/flydsl]] ·
[[operators/layernorm/backends/flydsl]] · [[attention_prefill_fmha]] ·
[[languages/flydsl/kernel_families]] · [[languages/flydsl/knobs]].
