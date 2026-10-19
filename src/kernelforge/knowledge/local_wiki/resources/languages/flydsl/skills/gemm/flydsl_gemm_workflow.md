---
title: FlyDSL GEMM — how to work on one: order of operations, evidence, and the round discipline
kind: language
lever: flydsl_gemm_workflow
gens: [gfx950]
updated: 2026-10-19
sources:
  - AMD-AGI/Primus-Turbo@7615a5a2:agent/skills/kernel-optimize/SKILL.md
  - AMD-AGI/Primus-Turbo@7615a5a2:agent/rules/iteration_rules.mdc
---

# Working on a FlyDSL GEMM

> This card gives an order of operations and a discipline for each round. It does not say which
> configuration wins; only measurement on the box does that. The GEMM-specific continuation of
> `../optimize/flydsl_levers/flydsl_authoring_method.md`.

## Route here when
The question has narrowed to a dense GEMM, and you need to know where to start, what to change first,
or how to run the next round.

| The open question | Card |
|---|---|
| Loop design (8-wave pipeline or 4-wave whole loop), waits, inline-asm register plan, MFMA atom, K tail | `flydsl_gemm_mainloop.md` |
| LDS layout, swizzle, global→LDS DMA, descriptors, transpose reads, MX scales | `flydsl_gemm_datapath.md` |
| Wave placement, XCD/L2 tile order, scheduling hints and fences | `flydsl_gemm_schedule_and_locality.md` |
| Tile size, dispatch rounds, split-K, boundary bodies; what an idea can return at most | `flydsl_gemm_grid_pricing.md` |
| C store width, cross-lane reorder, cache policy, beta=1, fused epilogues | `flydsl_gemm_epilogue.md` |
| Per-shape race, cache keys, launch overhead, CUDA graphs | `flydsl_gemm_autotune.md` |
| Correctness gates, races, silent-wrong traps | `flydsl_gemm_correctness.md` |
| M ≤ 256: decode-shape rulers, grid, split-K reduction, aiter tuned rows | `flydsl_gemm_small_m.md` |
| Rulers, noise floors, A/B design, what to report | `flydsl_gemm_measurement.md` |
| The kernel is wrong, not slow | `../bottleneck/debug-flydsl-kernel.md` |
| Writing your first FlyDSL kernel | `../../API_docs/flydsl-tile-programming.md` |
| Not a GEMM, or the bottleneck is still unidentified | `../optimize/flydsl_levers/flydsl_authoring_method.md` |

**Two questions to settle before writing anything.** Does a shipped kernel already cover the shape
(`../optimize/flydsl_levers/flydsl_kernel_library.md`)? Is FlyDSL even the backend that will be dispatched
(`../../../../framework/aiter/skills/optimize/aiter_levers/aiter_flydsl_libtype.md`)? Both are cheaper to
answer than a rewrite.

## Step 0 — start from the reference kernel, not from a blank file

Read the Primus-Turbo FlyDSL GEMM for your dtype and layout before designing your own (kernel map in
`INDEX.md`). It is built for training shapes; for M ≤ 256 use `flydsl_gemm_small_m.md` instead. Its comments
record why each choice was made, and it runs on FlyDSL v0.3.2. Copy the structure and the reasoning;
re-derive the constants (tile tables, wait counts, margins, L2 thresholds) for your shape and machine. Those
constants were tuned on specific shapes.

## Step 1 — establish that it really is a GEMM

Read both the device kernel and the host launcher, and answer:
- What are the logical M, N and K, and which of them change at run time?
- How do workgroups and waves divide the output tile? Which waves share a SIMD?
- Which operands come from global memory every iteration, and how (LDS DMA, sync copy, direct to VGPR)?
- Does the epilogue store straight out, reorder first, or fuse other work?

A kernel with incidental MFMA is not a GEMM, and GEMM advice will not help it.

## Step 2 — let the evidence pick the problem

First do the arithmetic that needs no profile: tiles, dispatch rounds and last-round fill
(`flydsl_gemm_grid_pricing.md` §1). Then gather the evidence in this order: wall time across shapes, an ATT trace
(`vmcnt`, `lgkmcnt`, `s_barrier`, `ds_*`, `buffer_load_*`, `v_mfma_*`), and the ISA hot loop.

| What the evidence shows | What it usually means |
|---|---|
| `s_waitcnt vmcnt(0)` in front of the MFMAs | global latency is exposed, or the compiler inserted a conservative drain (`flydsl_gemm_datapath.md` §4) |
| `s_waitcnt lgkmcnt(0)` or `ds_*` stalls | LDS latency, bank conflicts, or a scalar load in the same window |
| Time in `s_barrier` | wave drift or too many rendezvous; check the wave → SIMD mapping first |
| Few MFMAs per memory instruction | the loop body or its schedule |
| Dense MFMAs but poor wall time | tile shape and rounds, occupancy, the store path, or the power cap |

**Two common misreadings.**
- TF/s rising with K is not proof that a kernel is memory-bound. A fixed per-tile cost (prologue,
  epilogue) produces the same curve (`flydsl_gemm_grid_pricing.md` §6).
- A busy matrix core is necessary, not sufficient. At the power cap a change that cuts register-file or
  LDS traffic can win as clock while every utilization counter stays the same
  (`flydsl_gemm_mainloop.md` §5).

## Step 3 — fix in this order

1. Grid and tile (rounds, last-round fill, split-K)
2. Loop design and data path (wave count, staging depth, how operands reach LDS and registers)
3. Waits, fences and schedule
4. Epilogue and store
5. Per-shape configuration (the race)

Each level changes the pressure the next one works under. Constants tuned against a bad tile have to be
re-tuned after the tile is fixed. Do not touch step 5 while steps 1–4 still have known problems.

## The round discipline

These rules come from Primus-Turbo's optimization loop (`agent/skills/kernel-optimize/SKILL.md`,
`agent/rules/iteration_rules.mdc`). Each exists because breaking it cost rounds.

- **One change per measured round.** Sweeping one configuration axis with the kernel code unchanged
  counts as one change. A new MFMA atom plus a new LDS stage is two, and their effects cannot be
  separated afterwards.
- **State the hypothesis before building:** the bottleneck, the change, the expected gain with its
  ceiling (`flydsl_gemm_grid_pricing.md` §7), the risk, and the signal that will confirm it (which stall,
  which counter, which instruction count).
- **Two levels of validation.**
  - Quick: 3–5 representative shapes that pass correctness, spread over small, medium and large, with
    the high-variance ones included.
  - Full: every scored shape. Run it when a quick gain is under about 5%, when a direction ends, and after
    any control-flow or layout change.
  - Never drop a shape inside a level.
- **Accept mechanically.** The primary metric must improve beyond the noise floor
  (`flydsl_gemm_measurement.md`), and no scored shape may regress beyond its own floor, even when the
  geomean improves.
- **Escalate on consecutive rollbacks.**
  1. One rollback: write down why it failed.
  2. Two: re-profile from scratch.
  3. Three: read the ISA of the hot loop.
  4. Four or more: compare against the best reference kernel for this dtype and layout, instruction by
     instruction.
- **After two rollbacks in a row, review the plan.** List at least three fundamentally different
  directions, drawn from different families: grid and tile, data movement, pipelining, occupancy,
  comparison with a reference kernel, fusion or reordering. Do not re-sweep the family that just failed.
- **No benchmark-only gains.**
  - A cache keyed on an input tensor's identity works only because the benchmark reuses tensor objects.
  - A shortcut that assumes uniform shapes or distributions will not survive real inputs.
  - Preprocess static weights once, outside the timed region.
  - If a gain is larger than the change could structurally produce, look for one of these before
    accepting it.

## Recurring mistakes
- Reaching for scheduler constants before the bottleneck is proven.
- Copying tile sizes, wait counts or XCD settings from another kernel without checking that the work
  decomposes the same way (`flydsl_gemm_schedule_and_locality.md` §2).
- Adding LDS stages or prefetch until occupancy or registers collapse.
- Treating every LDS symptom as a swizzle problem without checking the write-to-read distance.
- Benchmarking one shape and tuning until it wins.
- Replacing a measured choice with a model because the model agreed on the scored shapes
  (`flydsl_gemm_autotune.md` §1).

## Verifying a change
1. Correctness first, with a gate that exercises the changed path (`flydsl_gemm_correctness.md` §1).
2. Re-measure the same shapes as the baseline, in the same session, each arm at its own best
   configuration.
3. If you had trace or ISA evidence, confirm that the specific stall moved. A wall-time drop for some
   other reason teaches nothing, and the next change will be a guess.
4. Check that no neighbouring shape regressed.

## Related
`INDEX.md` (this package's map) · `../optimize/flydsl_levers/flydsl_authoring_method.md` ·
`../optimize/flydsl_levers/flydsl_knob_space.md` · `../optimize/flydsl_levers/flydsl_kernel_library.md` ·
`../../../../hardware/mi350_lds.md` · `../../../../hardware/mi350_matrix_core.md`
