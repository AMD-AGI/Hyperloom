---
title: FlyDSL grouped GEMM — inference MoE expert GEMMs: sorted-token mapping, persistence, cache policy, stage 2
kind: language
lever: flydsl_grouped_moe_inference
gens: [gfx950]
dtypes: [bf16, fp8, int8, int4, mxfp4]
regimes: [moe-inference, decode, prefill, skew]
updated: 2026-10-27
sources:
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_sorting_kernel.py:3-21
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_sorting_kernel.py:43
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_gemm_2stage/gemm1.py:62-72
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_gemm_2stage/gemm1.py:394-423
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_gemm_2stage/gemm1.py:518-535
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_gemm_2stage/gemm2.py:384
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_gemm_2stage/gemm2.py:434-443
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_gemm_2stage/gemm2.py:595-619
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_gemm_2stage/moe_reduce.py:164-203
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/__init__.py:4-17
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/gemm1.py:59-67
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/gemm1.py:120-122
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/gemm1.py:748-774
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/gemm2.py:105
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/gemm2.py:230-289
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/gemm2.py:315-319
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/host.py:21-48
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/host.py:110
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/host.py:125-138
  - ROCm/FlyDSL@dd8373349:kernels/moe/mxfp_moe/host.py:202-209
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_2stage_a16wmix/gemm2.py:34-41
  - ROCm/FlyDSL@dd8373349:python/flydsl/expr/numeric.py:502-505
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_2stage_a16wmix/gemm2.py:495-505
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_2stage_a16wmix/gemm2.py:517
  - ROCm/FlyDSL@dd8373349:kernels/moe/moe_2stage_a16wmix/gemm2.py:630-641
---

# FlyDSL MoE expert GEMMs for inference

> The one card on inference MoE expert GEMMs: tokens are sorted by expert into blocks of tile-M rows, and each
> workgroup runs one (M-block, N-block) of one expert — stage 1 is gate + up with the activation, stage 2 the down
> projection with the routing weights. FlyDSL facts are for v0.3.2 (`kernels/moe/`). Numbers marked *(field)* are
> measured values from a4w4 MoE kernels of this family at decode batch sizes; use them for direction and order
> of magnitude. Which kernel aiter dispatches and its tuned database are in
> `../../../../framework/aiter/skills/optimize/aiter_levers/aiter_moe_pipeline.md` (A4W4 on FlyDSL:
> `aiter_flydsl_libtype.md` beside it). Training grouped GEMMs (variable-K wgrad, group offsets) are the rest of
> this package (`INDEX.md`).

## Route here when
- You are writing, calling or tuning a FlyDSL MoE stage 1 or stage 2, or the token sort in front of them.
- You are choosing tile-M, tile-N, grid order, the weights' cache policy, atomics or a reduce, or a persistent
  stage 2.
- A MoE GEMM is fast at one batch size and slow at another.

## 1. What FlyDSL v0.3.2 ships

| Path | Operands | Stage 2 | Grid |
|---|---|---|---|
| `moe_gemm_2stage/` | fp8 E4M3, int8, int8smooth, int4 W (W4A8) | atomics, or partials + reduce | 2D: x = N, y = M-block |
| `mxfp_moe/` | a4w4, a8w4 (fp8 A × MXFP4 W), gfx950 | atomic, non-atomic, CShuffle, MXFP4-/fp8-out | 1D, N fastest |
| `moe_2stage_a16wmix/` | bf16 A × MXFP4, int4 or bf16 W | atomic scatter, optional persistent | 1D, N fastest |
| `moe_sorting_kernel.py` | token sort by expert | — | — |

`moe_gemm_2stage` runs on gfx94* and gfx95* with f16/bf16/f32 outputs. `mxfp_moe` stage 1 fuses gate+up, SiLU
and an fp4 re-quantization of the intermediate; its MXFP4-out stage 2 is persistent. `moe_2stage_a16wmix` uses a
K=16 MFMA path on gfx942 and K=32 on gfx950.

`mxfp_moe` uses its own sorting contract (`cumsum` plus `m_indices`, `__init__.py:4-17`). FlyDSL also ships a
fused dispatch + GEMMs + combine kernel (`kernels/mega_moe/`) and a gfx1250 MoE kernel; neither is covered here.

## 2. How the work is mapped

- **The sort is the contract.** FlyDSL's sorting kernel is a counting sort in LDS (histogram, prefix sum,
  scatter) with three paths by token count: one kernel for small batches, two kernels (two phases fused) up to
  2048 tokens, four beyond. Token ids are packed as `(slot << 24) | token`; padding is `(topk << 24) | M`. The
  unit is the GEMM's tile-M (`UNIT_SIZE = 32`, "block_size in CK") (`moe_sorting_kernel.py:3-21`, `:43`). It
  produces sorted token ids, one expert id per M-block, and the number of valid ids.
- **One expert per M-block, read once and wave-uniform.** `moe_gemm_2stage` reads `expert_ids[e_idx]` for the
  block's BM sorted rows (`gemm1.py:399-403`), so the sort unit must equal its tile-M. `mxfp_moe` broadcasts
  the expert id with `readfirstlane` (`gemm1.py:120-122`).
- **Over-launch, exit on the device.** Grids come from an upper bound on M-blocks: in `mxfp_moe`,
  `(tokens·topk + active·(BM − 1) + BM − 1) // BM` with `active = min(tokens·topk, experts)` (all experts at
  BM 128), since each active expert pads at most BM − 1 rows (`gemm1.py:59-67`). Surplus workgroups exit when
  `e_idx · BM` reaches the valid count (`moe_gemm_2stage/gemm2.py:384`, `gemm1.py:394`), or when `bx` reaches
  `bound = cumsum[0] / BM · n_blocks` computed on the device (`mxfp_moe/gemm1.py:748-750`). No host read of the
  counts.
- **Sorted ids staged in LDS, used twice.** `moe_gemm_2stage` loads the block's sorted ids into LDS once and
  reads them for the A gather and again for the output scatter. The gather and scatter layouts read up to
  `256 / (tile_k / 16)` rows, more than BM at decode, so every readable slot is first seeded with the
  out-of-range id M — the buffer bound drops those accesses — before the real ids are written; an unseeded slot
  decodes to token 0 and piles garbage onto it (`gemm1.py:412-423`).
- **Exit on the raw id, then remap.** `mxfp_moe` tests `bx < bound` first; only then does it map `bx` through a
  bijection over `[0, bound)` that gives each of the 8 XCDs a contiguous slice of the tile order, followed by an
  M-group swizzle of `xcd_swizzle` blocks. Stage 1 applies this only when `xcd_swizzle > 0` (default 0,
  `gemm1.py:752-774`, `host.py:110`); stage 2 always applies the XCD part (`gemm2.py:230-254`, `:273-289`).
  Remapping before the exit would put all dead ids on one XCD (`flydsl_grouped_group_mapping.md` §4).
- **Persistent stage 2, capped at the CU count.** `mxfp_moe` (MXFP4-out epilogue only, `gemm2.py:105`) and
  `moe_2stage_a16wmix` (optional) launch `min(work, NUM_CU)` workgroups when the padded upper-bound work exceeds
  `4 · NUM_CU`, and stride through `[0, bound)` by the grid (`mxfp_moe/gemm2.py:315-319`,
  `a16wmix/gemm2.py:495-505`); with many experts the padded grid would otherwise over-launch mostly empty
  workgroups (`a16wmix/gemm2.py:34-36`). A barrier at the top of the loop keeps the previous tile's epilogue LDS
  apart from the next tile's A DMA (`a16wmix/gemm2.py:630-641`).
- **The non-persistent atomic stage 2 is HBM-bound**: each tile reads a full fp4 weight column block with about 4%
  L2 reuse; `mxfp_moe` applies the XCD remap above to it to balance HBM channels (`gemm2.py:262-289`).
- Per-expert 64-bit bases: `flydsl_grouped_group_mapping.md` §7.
- **A guard inside a helper needs `@flyc.jit`.** `a16wmix`'s atomic epilogue helper is a `@flyc.jit` function so
  that `if token_id < M` is rewritten into a device branch. As a plain Python function the `if` runs at trace
  time: on a `Numeric` value it raises, but on a raw `ArithValue` (e.g. an `llvm.load` result) it is always true,
  so the guard disappears silently, the atomic scatter fires on padded rows, and stage 2 ran ~13× slower at a
  very large expert count (`gemm2.py:39-41`; `Numeric.__bool__`, `python/flydsl/expr/numeric.py:502-505`).

## 3. Grid shape at decode sizes

- **At decode M the N blocks are the parallelism.** With tile-M at the sort unit, a decode batch fills only a few
  M-blocks. *(field)* On an a4w4 atomic stage 2 with tile-M 32, narrowing tile-N from 256 to 128 to 64 helped at
  each step (64 against 128: +0.84%, +0.88%, +0.96% in three pairs); 32, buildable only with an in-block K
  split, was slower than 64. Race tile-N downward until the per-block fixed cost wins.
- **Wider channel blocks pay until the grid runs out.** FlyDSL's fp8/int8 stage 1 uses a 128-channel block (half
  the blocks and waves, less barrier and LDS-sync overhead) but caps it at 128 so tile-N 256 stays two blocks per
  row, because one 256-channel block made a 256-wide intermediate ~40% slower (grid too small, register pressure)
  (`moe_gemm_2stage/gemm1.py:62-72`).
- **Split K inside a block only when K is long.** *(field)* Splitting K across the waves of a stage-2 block,
  where K is the intermediate size, was neutral to negative: the split landed on launch and epilogue fixed costs.
- **Keep the grid N-fastest.** *(field)* Swapping to M-fastest cost 2.4% on stage 1 and 1.5% on stage 2 (whose
  relative L2 error also rose from 0.0003 to 0.0017). N-fastest keeps the N-blocks that share one A tile running
  together, so the A tile stays in L2 and MALL. Both FlyDSL grids are N-fastest
  (`moe_gemm_2stage/gemm1.py:518-535`, `mxfp_moe/gemm1.py:120-121`).
- *(field)* Halving stage 2's A-tile LDS (packing fp4 at its real width) was neutral: that stage 2 was not limited
  by LDS occupancy.

## 4. Cache policy on the weights

- **FlyDSL's switch.** `mxfp_moe` stage 1 (its BM-32 variant) streams B non-temporally at small M, where no tile
  reuses another's weights, and switches to cached loads once total M-blocks ≥ experts (on average at least one
  padded block per expert), because consecutive M-blocks of one expert then reuse the same weight columns. Its
  note: L2 hit ~34% streaming against ~59% cached, ~22% faster at an effective M of 8192 (`host.py:125-138`). The
  `a16wmix` stage 2 defaults its B loads to the same non-temporal cache modifier (`b_cache_mod=2`,
  `gemm2.py:517`).
- **Non-temporal is a property of each load, not of the operand.** *(field)* Choosing per M-block — cached when a
  neighbour in sorted order has the same expert (`sorted_expert_ids[blk ± 1] == expert`), non-temporal otherwise —
  gained 1.82% over all-non-temporal on an a4w4 decode score. Conditions:
  - the reuse exists only if same-expert neighbours share an XCD: with the hardware XCD order (no remap),
    adjacent M-blocks of an N-fastest grid are `grid.x` workgroups apart, so `grid.x` must be a multiple of 8;
    a contiguous XCD remap such as `mxfp_moe` stage 2's keeps them on one XCD except where a slice ends;
  - one neighbour on each side is enough: in a run of two or more same-expert blocks, every block has one;
  - compile the branch only when the host knows a repeat is possible (tokens > block M): the if/else duplicates
    the GEMM body, and *(field)* small batches that could not repeat a block paid up to 2.1% for it.
- **A skip-the-block bound underestimates reuse.** *(field)* A probe that skipped the repeated blocks bounded the
  gain at 2.11% of the score and 1.0 µs at one batch size; at that batch size the cached version saved 1.83 µs,
  beyond the bound, because it also relieved L2 contention with the other streams.
- **Keep scale loads cached.** *(field)* Marking the B-scale loads non-temporal on single-read blocks lost 1.7%:
  4-byte loads gain nothing from streaming and lose line merging. A and its scales stay cached as well — all the
  N-blocks of a row share the A tile.
- **Dead alternatives.** *(field)* One workgroup walking two M-blocks of an expert (turning cross-workgroup reuse
  into same-workgroup reuse) halved the workgroup count and scored 8.5% lower than the all-non-temporal
  baseline; an XCD swizzle that grouped same-expert blocks scored 16.6% lower. Bypassing L1 (`sc0`/GLC) on
  the cached blocks scored the same as plain cached: the reuse is across CUs.

## 5. Tile-M equals the sort unit

- Each sorted block holds one expert's rows. A tile-M smaller than the block splits it into several M-tiles, and
  every M-tile streams the expert's weight slab again: weight traffic ∝ M-tiles × N-tiles × (tile-N · K bytes).
  *(field)* A fused MoE GEMM running tile-M 16 against a 32-row sort unit read every weight byte twice; tile-M 32
  cut its time by 16–34% at the larger batches.
- **Tile-M, the B cache policy and the K tile move together.** *(field)* Non-temporal B and a 256-deep K tile
  each lost on tile-M 16 (non-temporal B made three of four batch sizes much slower) and paid on tile-M 32.
  Sweep them as one axis.
- A tile-M larger than the sort unit is not available when the unit must be a multiple of tile-M; to try one,
  change the sort unit with it.

## 6. Stage 2: atomics or split + reduce

- **`moe_gemm_2stage`** (`gemm2.py:595-619`): `accumulate=True` adds into the output with atomics — f16 half2
  atomics are fast but can overflow to ±inf, bf16 uses buffer atomics on gfx95+, f32 scalar atomics are slower and
  avoid the overflow. `accumulate=False` writes per-(token, slot) partials (f16/bf16 only) for
  `compile_moe_reduction`, launched over `(tokens, model_dim blocks)` (`moe_reduce.py:164-203`).
- **Keep packed atomics coalesced.** FlyDSL's stage 2 keeps the CShuffle layout channel-major (32 lanes walk
  contiguous channels of one row) and `e_vec = 2` on the atomic path, so the 32 lanes cover 64 contiguous
  channels; its note puts the wider form at 4.25× the atomic traffic and 4.8× slower (`gemm2.py:434-443`).
- **Atomics are not bit-reproducible**; gate on a tolerance against an independent reference
  (`../gemm/flydsl_gemm_correctness.md` §8).
- **`mxfp_moe` stage 2 variants** (`host.py:21-48`): atomic at BM 16/32/64 with or without non-temporal B,
  non-atomic, CShuffle, MXFP4-out and fp8-out at BM 128 (CShuffle also at 32/64). The host defaults
  `xcd_swizzle = 3` for the non-atomic and fp8-out epilogues and routes BM-128 non-atomic to CShuffle
  (`host.py:202-209`).

## 7. The small kernels around the GEMMs

- **Price sorting and quantization kernels by launching them again** (`flydsl_grouped_measurement.md` §6).
  *(field)* One sorting phase cost 2.3–2.4 µs per extra launch under graph replay; shortening the loops inside it
  moved the score by 0.03%, inside the noise; merging two sorting phases into one kernel gained 1.08%. Only
  removing a whole kernel pays. FlyDSL v0.3.2's mid-size sorting path runs two kernels
  (`moe_sorting_kernel.py:12-14`).
- **Merging kernels turns an inter-kernel order into an intra-kernel race.** One phase seeded a table with −1
  and the next overwrote the local experts' entries; the kernel boundary ordered the two. Once merged,
  `gpu.barrier()` is not guaranteed to order the global stores (it synchronizes the waves of one workgroup).
  Make the two writes disjoint (seed only the entries the scatter never writes).

## 8. Measuring at decode sizes

- Decode reads cold weights, runs under graph replay and pays a per-node floor: rotate weight copies past the
  MALL, time a replayed graph, and price through the deployed entry (`../gemm/flydsl_gemm_small_m.md` §1).
- *(field)* A MoE decode scorer's in-batch noise was ~0.2%, but identical code drifted 0.3–0.4% between batches.
  Decide on interleaved pairs within one batch; use cross-batch values for direction only.
- Cover several batch sizes, from a few tokens (single M-block per expert, non-temporal B) to above
  `experts · tile-M / top-k` tokens (total M-blocks ≥ experts: repeated blocks, cached B); the policy flips
  between them.

## Verify
- [ ] The sort unit equals tile-M (or is a multiple of it), and the tile-M sweep includes the unit itself.
- [ ] The grid is N-fastest, exits on the raw id before any XCD rotation, and bounds its work on the device.
- [ ] The weights' cache policy was raced per batch size, with the repeated-block condition and its host gate.
- [ ] Atomic stage 2 is gated on a tolerance, and the atomic layout keeps 32 lanes on contiguous channels.
- [ ] Small kernels were priced by extra launches, and merged kernels write disjoint regions.

## Related
`../../../../framework/aiter/skills/optimize/aiter_levers/aiter_moe_pipeline.md` ·
`../../../../framework/aiter/skills/optimize/aiter_levers/aiter_flydsl_libtype.md` ·
`../gemm/flydsl_gemm_small_m.md` · `../optimize/flydsl_levers/flydsl_kernel_library.md` ·
`flydsl_grouped_scheduling.md` · `flydsl_grouped_measurement.md` · `INDEX.md`
