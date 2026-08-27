---
title: reduction — overview
kind: operator_overview
operator: reduction
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int32]
regimes: [prefill, decode, training, both]
updated: 2026-06-08
---

# reduction  (`out = ⊕_i x[i]` along an axis — sum/mean/max/min/L2)

## TL;DR
Reduction collapses an axis with an associative op: **sum, mean, max, min, L2-norm** (Σx², used in
RMSNorm/LayerNorm), and `prod`. It's memory-bound like elementwise but adds a **cross-lane combine**, so
the AMD-specific levers are: a **wave64 shuffle reduce** (`__shfl_down` over 64 lanes, *not* 32) →
**LDS** to combine waves within a block → and, when the reduced axis is huge or there are too few output
rows to fill 304 CUs, a **split / multi-block reduction** that finishes with `atomicAdd` or a second
"reduce-the-partials" kernel. The dominant LLM reduction is the **row reduce** in RMSNorm/LayerNorm/softmax
(reduce `hidden`, keep `tokens`) — there the reduction lives *inside* the norm/softmax kernel; a standalone
reduction is mostly torch/`sum`/`max` glue and Inductor fuses it.

## Math contract
- **Full reduce**: `out = ⊕_{i} x[i]` → scalar.
- **Axis reduce**: `out[r] = ⊕_{c} x[r, c]` (reduce last dim, the LLM "row reduce") or `out[c] = ⊕_r x[r,c]`
  (reduce leading dim — strided, harder).
- ops: `sum`, `mean` (=sum/n), `max`, `min`, `L2 = sqrt(Σx²)`, `prod`, `any/all`.
- dtype: **accumulate in fp32** even for bf16/fp16 inputs (parity + overflow); `mean`/`L2` finalize in
  fp32 then round. `argmax`/`argmin` (index-returning) → [`../argmax_topk/overview.md`](../argmax_topk/overview.md).

## Shape regimes
- **Row reduce `[tokens, hidden]` → `[tokens]`**: `hidden ∈ {4096,5120,8192}`, `tokens` 1..16k. One block
  (or a few waves) per row; the dominant shape (norm/softmax). Many rows ⇒ grid fills naturally.
- **Few-row / huge-axis reduce** (e.g. global norm, a single large vector): **few output elements** ⇒ a
  naive one-block-per-output leaves 303 CUs idle ⇒ **split reduction** required.
- **Leading-dim / column reduce**: strided, anti-coalesced — transpose-aware or atomic split.

## Where it matters (Amdahl)
As a *standalone* op, small. But the reduce **inside RMSNorm/LayerNorm/softmax** is on the critical path
of every layer; its efficiency *is* the norm/softmax kernel's efficiency (see those operators). The
honest standalone-reduction win is **split reduction for low-output-count shapes** (turn 1 busy CU into
304) and **fusing the reduce into its producer**.

## Backend landscape
Competitive positioning for this operator; this folder's card is [flydsl.md](flydsl.md). Other backends
live in their own folders (`framework/aiter/`, `languages/{triton,ck,asm,hip}/`).

| backend | status |
|---|---|
| triton | 🟢 sota (authoring + Inductor target; `tl.sum/max` wave reduce) |
| hip | 🟢 sota (peak; explicit wave-shuffle + LDS + atomic split) |
| composable_kernel | 🟢 sota (DeviceReduce multi-block/atomic/two-call instances) |
| pytorch_inductor | 🟢 sota (fuses reduce into pointwise chain) |
| aiter | 🟡 (reduce lives inside fused norm/quant ops) |
| flydsl | 🧪 experimental (wave64 block-reduce **primitive** `make_block_reduce_add`/`_add2`, not a standalone op) |

## Fusion neighbors
Reduce-then-elementwise (norm: `Σx²` → `rsqrt` → `×x`), elementwise-then-reduce (`clamp`/`abs` → `sum`),
two reductions sharing a load (mean+var = Welford / one pass). Prime host for elementwise fusion. See
[fusion.md](fusion.md) and [`../rmsnorm/overview.md`](../rmsnorm/overview.md),
[`../layernorm/overview.md`](../layernorm/overview.md), [`../softmax/overview.md`](../softmax/overview.md).

## Numerics
fp32 accumulation; **order-dependent** (tree vs sequential vs atomic) → bf16 LSB nondeterminism; `max`
NaN handling; `mean` divide; L2 `sqrt` epsilon. See [numerics.md](numerics.md).

## How to bench
Isolated: time the reduce at the exact `[rows, axis]`, dtype; achieved GB/s = `input_bytes / time` vs
~4.3 TB/s (output is tiny). For split reduction, also report CU utilization (rocprof) — the whole point is
filling CUs at low output count. Parity: fp32 atol vs torch (exact bitwise won't hold across reduction
orders).
