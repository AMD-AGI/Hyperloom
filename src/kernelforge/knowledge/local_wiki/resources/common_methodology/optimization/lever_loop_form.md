---
title: loop form and pipelining — a data-dependent trip count silently disables num_stages and async copy
kind: technique
gens: [gfx942, gfx950]
dtypes: [any]
regimes: [prefill, decode, both]
updated: 2026-08-23
---

# loop form and pipelining — the compiler optimizes the loop you wrote, not the work you meant

## TL;DR
`num_stages` software pipelining and direct-to-LDS asynchronous copies both require the loop's **trip
count and addresses to be shape-static** — derivable from tensor *shapes*, constexprs and the launch
grid, never from tensor *contents*. A `while` loop whose bound comes from a tensor load, or a
`tl.load` whose address came out of another `tl.load` in the same iteration, disables both
**silently**: no error, no warning, no diagnostic — only a slower kernel. So on a sparse or gather
kernel the question is not *how many blocks does this loop visit* but *can the compiler pipeline it*,
and rewriting a data-dependent gather as **"iterate a larger shape-static range and select with a
mask"** can win on net while doing strictly more nominal work. Measured once, on one kernel: the
static form visited **2–4× more blocks and still ran ~2× faster** (0.1164 ms vs 0.2336 ms; see
"The evidence"). The mistake this card exists to prevent is pricing a loop restructuring on **work
volume alone** — the arithmetic can be right and the conclusion still wrong, because the payoff is not
the work you saved.

## What "shape-static" does and does not mean
It does **not** mean compile-time constant. Every dense GEMM K-loop
(`for k in range(0, tl.cdiv(K, BLOCK_K))`, with `K` a runtime kernel argument) is shape-static and is
pipelined at `num_stages=2` — the bound is loop-invariant and the compiler can build a schedule around
it without knowing its value. It means the compiler can answer two questions **without reading device
memory**:

1. **Does iteration `i+S` execute?** — needed to issue its loads `S` iterations early (the prologue).
2. **What address does iteration `i+S` load from?** — needed to form that load at all.

Anything that makes either answer depend on the *values* in a tensor pushes the loop off the
pipelined path. The two forms that do it in practice are a `while` whose exit test reads memory (the
compiler cannot prove iteration `i+S` exists) and an indirect load — index in, data out, inside one
iteration (the address for `i+S` is not available until iteration `i+S` runs).

## Why it costs so much (the mechanism)
- **Stream pipelining** (`[[optimization/lever_prefetch.md]]`) is what overlaps
  `global_load` → `ds_write` → `ds_read` → consumer across iterations. Without it every iteration pays
  a full exposed global round trip: `s_waitcnt vmcnt(0)` immediately before the consumer, nothing in
  flight behind it. On a memory-bound gather that is not "a few percent of overhead", it is the entire
  latency-hiding structure of the kernel.
- **Direct-to-LDS async copy** (`global_load_lds` / `buffer_load ... lds`, `s_wait_asynccnt`) needs its
  destination LDS slot assigned before the data lands, against a statically known set of in-flight
  copies. An indirect gather cannot use it: the index load must complete before the data load can even
  be formed, so the iteration serializes into two dependent round trips, neither overlapped. The path
  also frees the staging VGPRs (`[[optimization/lever_occupancy.md]]`), so losing it costs
  registers as well as latency.
- Both losses are **silent**. The kernel compiles, the knobs are accepted, `num_stages=3` is a legal
  config that changes nothing. There is no signal in the build output — only in the ISA and the clock.

## The signature (how to recognise it in your own kernel)
Read the loop, not the comment above it. Any of these puts it on the unpipelined path:

- **`while` with a data-derived bound** — `n_sel = tl.load(counts + pid)` … `while blk < n_sel:`, or a
  `break` on a sentinel value read from memory. Any `while` at all is suspect: Triton pipelines `for` /
  `tl.range` loops, and a `while` is a different construct in the IR.
- **A pointer loaded inside the loop and dereferenced in the same iteration** — the paged-attention
  shape: `page = tl.load(block_table_row + blk)` then `tl.load(kv + page * stride + offs)`. This is the
  literal test: **`tl.load` on an address derived from another `tl.load`.**
- **An induction variable advanced by data** — a cursor read from a linked index, a next-pointer, a
  per-program top-k list walked to its own length.
- **A trip count that changes with the input's values rather than its shape** — two programs in the
  same launch iterating a different number of times because their *contents* differ.

**The check that needs no compiler:** *could I write down the exact sequence of addresses this loop
touches, knowing only the tensor shapes and the launch grid?* If no, it will not pipeline.

## The rewrite
> **Bound the iteration by a shape-static range; move the selection into a mask.**

1. **Find the smallest shape-static superset.** The tightest range you can compute from shapes alone
   that provably contains everything you must visit. For causal attention that is the visible block
   range, `0 … cdiv(q_block_end, BLOCK_N)`; for a per-sequence paged walk it is that sequence's page
   count; for a top-k union it is whatever window the top-k was drawn from.
2. **Iterate it with `for` / `tl.range`**, bound loop-invariant, no `break`, no `continue`, no early
   exit. Early exit re-introduces the data-dependent trip count you just removed.
3. **Turn membership into a value, not control flow.** Build the per-(row, element) predicate — a
   bitmask, a comparison against a stored index, a precomputed boolean tile — and apply it as data:
   `other=0.0` on the load, `tl.where` on the accumulate, `-inf` on the attention score.
4. **Keep the addresses affine in the induction variable.** Index the data by the loop counter
   directly wherever the layout allows. Where an indirection genuinely cannot be removed, at least
   hoist it out of the per-iteration dependence chain (load the index vector once, before the loop)
   rather than leaving an index load feeding a data load inside the body.
5. **Then re-tune `num_stages`** — it now does something. Start from the operator's usual value (fused
   attention wants `1`, single GEMM `2`; see the language card) and sweep; one point per command with
   `[[optimization/lever_cheap_sweeps.md]]`.

## What it costs — this is a trade, not a free win
The static walk does strictly more nominal work: `N_range / N_selected` times the block visits, the
loads and the dots. Masked-out loads still cost bandwidth; masked-out matrix work still costs issue
slots. You are trading **work volume** for **latency hiding**, and the trade reverses somewhere.

**From one measured point you cannot locate where.** What the one point says is that at **2–4×
redundancy** on a memory-bound attention body the trade was strongly positive — roughly 2× faster
while doing 2–4× the nominal work. Do **not** read that as "the break-even is above 4×"; a single
kernel at one shape gives one point, and the crossing depends on how latency-bound the body is and on
what fraction of the redundant work is bandwidth versus issue slots. Where it will clearly lose is
obvious enough to state without measuring it: when the range dwarfs the selected set — a long-context
decode where top-k picks 16 blocks out of 4096 — the static walk reads 256× the KV and no amount of
pipelining buys that back. **Measure both forms.** This card gives you a hypothesis worth the
experiment, not a conclusion you can adopt unmeasured.

Two further costs to price before you commit:
- **LDS and registers.** The static form stages a full tile per iteration and `num_stages>1`
  multiplies that footprint; overflowing LDS drops occupancy and gives the win straight back
  (`[[optimization/lever_lds_banks.md]]`, `[[optimization/lever_occupancy.md]]`).
- **The mask must be exactly right.** A skipped element must contribute *nothing* — `-inf` before the
  softmax maximum, not `0` after it — or you have traded a slow kernel for a wrong one
  (`[[optimization/lever_numerics.md]]`).

## The evidence, and what it does not cover
**The measurement.** MI355X (gfx950) kernel arena, `gqa` sparse-attention prefill, Triton 3.6,
2026-08-18 … 08-23. Two agents wrote the same kernel two ways over the same data:

| | loop form | blocks visited per tile | best large-case time |
|---|---|---|---|
| data-dependent | `while` over the union of the tile's per-query top-k lists; trip count and page pointer both from tensor loads | ~16–30 | **0.2336 ms** |
| shape-static | `tl.range` over the contiguous causally-visible block range, per-(query, block) bitmask | 64 (all visible) | **0.1164 ms** |

The static form visits 2–4× more blocks and is **2.0× faster**. Across six head-to-head runs the
dynamic form never came within 2× of the static form's ceiling and lost every one. Over the five batch
pairs where both produced a scored run, a paired one-sided t-test on `log(ratio)` gives **t = −2.93,
mean −15.2%, t_crit(95%) = 2.132** — separable from noise, which matters on this harness because
batch-to-batch variance is wide (the same build twelve hours apart moved individual kernels −7.0% to
+16.5%; a single-batch delta under ±17% carries no signal).

**What this rests on — state it before you generalize.** One kernel, one operator class, one
architecture (gfx950), one Triton version (3.6). The two loop forms were written by two different
agents, so this is *not* a controlled A/B with everything else held fixed: the attribution of the
margin to pipelining plus async copy is the best reading of the structural difference between the two
kernels, not an isolated measurement of loop form. If you have the budget, do the controlled thing —
write both forms yourself over the same data and time them in one session
(`[[profiling/measure_protocol.md]]`).

**Generation caveat.** The pipelining half is the same stream pipeliner on gfx942 and gfx950. The
async-copy half is not symmetric: `knobs.amd.use_async_copy` is **default on gfx950 and experimental on
gfx942**, so on CDNA3 the second mechanism may not be engaged in the first place and the margin should
be expected to be smaller. Not measured on gfx942.

## Verify
- **The cheapest probe: does `num_stages` do anything?** Sweep `num_stages ∈ {1,2,3}` on the loop in
  question. If latency is flat across all of them inside the noise band, the loop is **not being
  pipelined at all** — that flatness is the diagnostic, not a finding about the right depth
  (`[[optimization/lever_cheap_sweeps.md]]`).
- **ISA dump** (`AMDGCN_ENABLE_DUMP=1`,
  `[[languages/triton/skills/optimize/triton_levers/triton_isa_check.md]]`): a pipelined loop issues the
  *next* tile's global loads ahead of the current tile's consumer, with one `s_waitcnt` per stage
  boundary. A `vmcnt(0)` immediately before every use means nothing is in flight. `global_load_lds` /
  `buffer_load ... lds` and `s_wait_asynccnt` present ⇒ async copy engaged; absent on gfx950 where you
  expected it ⇒ the loop form blocked it.
- **Profiler**: exposed global latency shows as memory-wait stalls before the consumer with HBM far
  from peak and low matrix-core busy — the latency-bound signature, not the bandwidth-bound one
  (`[[profiling/measure_triage.md]]`, `[[optimization/lever_bottleneck_class.md]]`).
- **A/B both forms**, same session, non-overlapping, under the canonical protocol
  (`[[profiling/measure_protocol.md]]`).

## The reasoning failure, named
An analysis that rejects a loop restructuring on work volume alone has priced one dimension and closed
the axis on it. The case that produced this card ran the arithmetic correctly — *"two adjacent queries
share ≈7 of 16 blocks by chance … worth a few percent, not the headline lever"* — and reached the wrong
answer, because the winning form does not depend on sharing work at all. It depends on being a loop the
compiler can pipeline.

So: **whenever you reject a change to loop *structure*, write down which of the two dimensions you
priced.** If the answer is only "how much work does it do", the change is not yet priced, and a
constraint recorded on that basis will keep the axis closed for every iteration that inherits it.

## See also
- `[[optimization/lever_prefetch.md]]` — what the pipeline does once the loop form allows it;
  `num_stages`, prefetch distance, `global_load_lds`, CDNA3 vs CDNA4.
- `[[optimization/lever_bottleneck_class.md]]` — classify first; this lever is for latency-bound and
  memory-bound loops, not for a compute-bound body.
- `[[optimization/lever_cheap_sweeps.md]]` — one command per `num_stages` point.
- `[[optimization/lever_numerics.md]]` — the gate the mask has to survive.
- `[[languages/triton/skills/optimize/triton_levers/triton_lowering.md]]` — the AMD stream pipeliner and
  `knobs.amd.use_async_copy` in detail.
