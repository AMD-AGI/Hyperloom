---
title: FlyDSL — optimizing a GEMM you wrote yourself
kind: language
lever: flydsl_gemm_authoring
gens: [gfx950]
updated: 2026-08-28
---

# Optimizing a hand-written FlyDSL GEMM

> This is a how-to, not a ranking. Nothing here tells you which configuration wins — only on-box
> measurement does that. What it gives you is an order of operations that stops you tuning constants
> around a structural problem.

The GEMM-specific continuation of `flydsl_authoring_method.md`.

## Route here when
The conversation has already narrowed to GEMM structure. Concretely, when the open question is one of:

- how to pick `tile_m` / `tile_n` / `tile_k`
- which MFMA shape and repeat layout to use
- whether to add LDS ping-pong or deeper staging
- how to lay out LDS so the banks do not collide
- how to overlap global loads, LDS reads and MFMA
- what to pass to `sched_mfma` / `sched_vmem` / `sched_dsrd` / `sched_dswr`
- whether the epilogue should store directly or reorder first
- why VGPR pressure or occupancy is where it is
- what an ATT trace or ISA dump of the hot loop is telling you

**Somewhere else if:**

| Situation | Go to |
|---|---|
| Writing your first FlyDSL kernel | `../../../API_docs/flydsl-tile-programming.md` |
| The kernel is wrong, not slow | `../../bottleneck/debug-flydsl-kernel.md` |
| Not GEMM, or the bottleneck is still unidentified | `flydsl_authoring_method.md` |

**And two questions to settle before authoring anything:** does a shipped family already cover this
shape (`flydsl_kernel_library.md`), and is FlyDSL even the backend aiter will dispatch to
(`../../../../../framework/aiter/skills/optimize/aiter_levers/aiter_flydsl_libtype.md`)? Both are
cheaper to answer than a rewrite.

## Step 1 — establish that it really is a GEMM
Read both the device kernel and the host launcher, and answer these before touching anything. A kernel
with incidental MFMA in it is not a GEMM, and GEMM advice will not help it.

- What are the logical M, N and K?
- How do blocks and waves divide up the output tiles?
- Which operands come from global memory on every iteration?
- Which data gets re-read often enough that LDS staging pays for itself?
- Does the epilogue store straight out, or does it have to reorder fragments first?

## Step 2 — let the evidence pick the problem
Rank your evidence: runtime numbers plus shape sensitivity, an ATT trace (`vmcnt`, `lgkmcnt`,
`s_barrier`, `ds_*`, `buffer_load_*`, `v_mfma_*`), or an ISA dump you can count instructions in.

| What the evidence shows | What it means |
|---|---|
| `s_waitcnt vmcnt(0)` sitting in front of the MFMA | global-load latency is exposed — nothing is hiding it |
| `s_waitcnt lgkmcnt(0)` or visible `ds_*` stalls | LDS is the problem: latency or bank conflicts |
| Time going into `s_barrier` | too much synchronization |
| Few MFMA relative to everything else, with bubbles | the loop body or its schedule is wrong |
| Dense MFMA but disappointing wall time | tile shape, occupancy, or the store path |

That last row is the one people misread. Good MFMA density is necessary, not sufficient — a kernel can
keep the matrix core busy and still lose to a store path that writes uncoalesced.

## Step 3 — fix in this order
1. Tile strategy
2. LDS staging and overlap
3. MFMA loop scheduling
4. Epilogue and store strategy
5. Parameter tuning

The order is not arbitrary. Each level changes the pressure the next one operates under, so tuning
scheduler constants against a bad tile means re-tuning them after you fix the tile. **Do not touch
step 5 while steps 1–4 still have known problems.**

## Tiling
Constraints to satisfy before you consider anything else:

- `tile_m` divides evenly by the MFMA M dimension — **the atom is 16**.
- `tile_n` is big enough to keep the waves fed, and maps cleanly onto the wave and workgroup split.
- `tile_k · elem_bytes` lines up with how you are loading and packing operands.
- LDS per stage fits the budget: **160 KiB per workgroup on gfx950**.

Then the trade. Push `tile_k` up when there is enough compute to hide the memory latency and LDS has
room. Pull it down when LDS, register pressure or occupancy has become the binding constraint — and
note which one, because the fix differs.

For irregular shapes, pick a tile that holds up across the whole benchmarked range rather than the one
that wins on your favourite shape. A tile that is 10% better on one M and unusable on the next is not
an optimization. Aim the grid at **256 CUs**.

## LDS staging
Three questions, each with a different answer:

- Is an operand tile read many times by the MFMA loop? → stage it through LDS.
- Is global-load latency exposed? → prefetch it earlier.
- Does one LDS buffer sit idle while compute runs on the other? → consider ping-pong.

If LDS is already in use and something is wrong with it, separate the failure modes before reaching for
a fix:

| Mode | How it shows up |
|---|---|
| Capacity | LDS per workgroup is high enough to cap occupancy |
| Layout | bank conflicts caused by the stride or the access pattern |
| Timing | a `ds_write` too close to the `ds_read` that depends on it |

These need three different fixes. **Treating all LDS trouble as a swizzle problem is the standard way
to waste an afternoon** — check the write-to-read distance before you touch the layout.

## Bank conflicts: gfx950 has 64 banks
The bank index is `(byte_addr / 4) mod 64`.

**Any swizzle or padding you inherited was derived against 32 banks and is unverified here.** Re-derive
it. The larger 160 KiB budget also means plain padding is affordable in cases where it previously cost
too much LDS to consider.

Choose between the two:

- **XOR swizzle** when the access pattern is regular, the read and write transforms can be kept in
  lockstep, and LDS headroom is tight.
- **Padding** when swizzle arithmetic would clutter the address code and a small stride bump removes
  the conflict outright.

> Whichever you pick, **the producer and the consumer must agree**. A swizzled store paired with an
> unswizzled load is not a half-finished optimization — it is a correctness bug that will read
> plausible garbage.

## Prefetch and scheduling
Prefetch buys nothing unless there is independent work available to fill the latency it is hiding.
Things that qualify: next-tile global loads, address arithmetic, MFMA groups with no dependency on the
in-flight load, epilogue setup that does not touch data still in flight.

Watch the cost side. Prefetching means carrying more state in registers, and **prefetch that pushes you
into spills is a net loss** — re-check VGPR pressure and occupancy after adding it, not before.

For the scheduler hints, the counts have to come from *this* loop: how many MFMA, how many LDS reads,
how many VMEM operations per iteration, in the kernel in front of you. Constants lifted from another
kernel are actively worse than passing no hints, because they instruct the scheduler to interleave
around work that is not there.

## Epilogue
The epilogue is the mapping from accumulator fragments to output stores, and there are two shapes of
answer.

**Store directly** when the fragments are already reasonably coalesced and the mapping is simple.

**Reorder first** when the stores are poorly coalesced or the tile shape fragments the writes — but
only when the LDS traffic and barrier you are adding cost less than the store inefficiency you are
removing. That is a measurement, not a guess.

The shipped `use_cshuffle_epilog` argument on `flydsl_preshuffle_gemm_a8` is the library making this
same choice for you (`flydsl_knob_space.md`).

## Symptom table
| Symptom | Likely cause | First thing to try |
|---|---|---|
| `s_waitcnt vmcnt(0)` ahead of MFMA | global-load latency exposed | move next-tile loads earlier; revisit prefetch distance |
| `s_waitcnt lgkmcnt(0)` or `ds_*` stalls | LDS latency or conflicts | check layout, swizzle, padding, and write-read distance — in that order of cheapness |
| Time in `s_barrier` | too many synchronization points | collapse stage boundaries; merge dependent phases |
| Low MFMA ratio in the hot loop | schedule overhead, or loop shape | count MFMA against memory ops; simplify the body |
| Fast on one shape, slow on neighbours | the tile is brittle | re-check divisibility, occupancy, and edge handling |
| **Slower after adding prefetch** | register pressure crossed a threshold | carry less state, or stage more lightly |

## Correctness constraints
- Stay inside the LDS limit — **160 KiB on gfx950**.
- Keep tile packing and vector widths consistent with the operand layout.
- Check that accumulator and output type conversions cannot overflow.
- Apply swizzle or padding identically on the producer and the consumer side.
- Confirm edge masking still holds for shapes that do not divide the tile.

## Recurring mistakes
- Reaching for scheduler constants before the bottleneck has been proven.
- Lifting tile sizes from another kernel without checking that the work decomposes the same way.
- Adding LDS stages until occupancy collapses.
- Diagnosing every LDS symptom as a swizzle problem, without checking wait distance.
- Benchmarking one shape and tuning until it wins.
- Assuming a trace pattern from another repository transfers to this kernel.

## Verifying a change
1. Correctness first — a faster wrong kernel is not a result.
2. Re-measure the **same** shapes as the baseline, not a convenient subset.
3. If you had trace or ISA evidence, confirm **that specific stall moved**. Wall time dropping for some
   other reason means you have not learned anything and the next change will be a guess.
4. Check the win is not one shape improving while its neighbours regress.

## Related
`flydsl_authoring_method.md` (the general workflow this specializes) ·
`flydsl_knob_space.md` (the shipped kernels' arguments) ·
`flydsl_kernel_library.md` (check before authoring) ·
`../../../../../hardware/mi350_lds.md` · `../../../../../hardware/mi350_matrix_core.md`
