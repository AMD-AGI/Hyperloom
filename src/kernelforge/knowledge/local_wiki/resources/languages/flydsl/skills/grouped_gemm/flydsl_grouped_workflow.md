---
title: FlyDSL grouped GEMM — working order: pass and skew, mapping and order, fixed cost, then the body
kind: language
lever: flydsl_grouped_workflow
gens: [gfx950]
dtypes: [bf16, fp8, mxfp8, mxfp4]
regimes: [forward, dgrad, wgrad, variable-k, skew]
updated: 2026-10-27
sources:
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel.py:507-510
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:386-396
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:906-911
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/pytorch/ops/grouped_gemm_fp8.py:806-815
---

# Working on a FlyDSL grouped GEMM

> The order in which to look at a grouped GEMM — forward (NT), dgrad (NN, or NT on a transposed weight) and
> variable-K wgrad (TN) — in bf16, fp8 with per-tensor scales, MXFP8 and MXFP4. It does not say which
> configuration wins. The round discipline (one change per round, hypothesis before build, escalation after
> rollbacks, no benchmark-only gains) is in `../gemm/flydsl_gemm_workflow.md` and applies unchanged. Numbers
> marked *(field)* are measured values; use them for direction and order of magnitude, not as targets. Inference
> MoE expert GEMMs (sorted token ids) are in `flydsl_grouped_moe_inference.md`.

## Route here when
You are starting on a grouped GEMM, or a run of rounds has stopped paying and you need to know which layer you
have not audited yet.

| The open question | Card |
|---|---|
| Upper-bound grid, tile → group lookup, empty groups, 64-bit offsets | `flydsl_grouped_group_mapping.md` |
| XCD partition, band order, longest-first, persistent or one tile per workgroup | `flydsl_grouped_scheduling.md` |
| Per-tile fixed cost, tile and occupancy per pass, boundary and tail bodies | `flydsl_grouped_tile_pricing.md` |
| Contraction length per group, hot-group split-K, beta=1 accumulation | `flydsl_grouped_wgrad.md` |
| fp8 and MX scales, preshuffle, quantization, store forms, GLU/dGLU | `flydsl_grouped_lowprec.md` |
| Launch path, CUDA graphs, cache keys, the per-shape race | `flydsl_grouped_host_autotune.md` |
| Shapes, distributions, deployment conditions, per-cell noise, trace FLOPs | `flydsl_grouped_measurement.md` |
| Gates for skew, empty groups, large G, bodies that must be proven to run | `flydsl_grouped_correctness.md` |
| Main loop, LDS, DMA, MFMA atom, epilogue basics — the same as dense | `../gemm/INDEX.md` |

## 1. What the group dimension changes

- **The forward/dgrad grid is an upper bound.** M is cut at group offsets that exist only on the device.
  Cutting at group boundaries costs at most one short tile per group, so `(ceil(M_total / BM) + G) · n_blocks`
  workgroups cover any distribution; the surplus exits on the device (Turbo: `grouped_gemm_bf16_kernel.py:507-510`,
  `grouped_gemm_fp8_kernel.py:386-396`, `:906-911`). A variable-K wgrad's plain grid is exact
  (`G × tiles_per_group`) and the offsets set each tile's trip count; a deterministic split-K adds a fixed head
  of split ids that the offsets enable or retire (`grouped_gemm_fp8_kernel.py:3959-3965`,
  `flydsl_grouped_wgrad.md` §9). No host read of the offsets is needed (`flydsl_grouped_group_mapping.md` §2–§3).
- **Fixed costs per tile and per group.** Every tile finds its group; a variable-K wgrad restarts its prologue and
  pipeline at every group; a group with a handful of tokens still pays a whole tile. Dense GEMMs have none of
  this, and it is often the whole gap: where it was measured, the grouped main loop ran at dense speed and the
  loss was MFMA idle time around tiles and group boundaries (`flydsl_grouped_wgrad.md` §1,
  `flydsl_grouped_tile_pricing.md` §6).
- **Skew.** Real routing is uneven, changes per batch and leaves some groups empty. A kernel that is fast on a
  balanced distribution is a hypothesis until the skewed cells agree.
- **Three passes, three bottlenecks.** Settle which pass you are on before reusing a lesson.

| Pass | Layout | Where token skew lands | What usually binds |
|---|---|---|---|
| Forward | NT, `A[M_total,K] × B[G,N,K]` | M tiles per group; tiles cost the same | per-tile fixed cost; L2 reuse of a group's slab |
| Dgrad | NN, or NT on a transposed copy | as forward | as forward, plus transposed weight reads (NN) |
| Wgrad | TN, out `[G,N,K]`, contraction = the group's `M_g` tokens | each tile's contraction length | unequal tiles; short groups' full grids |

Turbo's MXFP8 dgrad is the NT form on a transposed weight copy (`grouped_gemm_fp8.py:806-815`).

*(field)* With an fp8 per-tensor kernel set, forward and dgrad stayed close to their balanced throughput under
skew; only wgrad lost a large share of it (`flydsl_grouped_wgrad.md` §1). The long tails of forward and dgrad
calls in a training trace came from overlap with collectives, not from skew (`flydsl_grouped_measurement.md` §3).

## 2. Step 0 — reference kernels and the body the benchmark reaches

- **FlyDSL v0.3.2 ships no training grouped GEMM.** Its grouped kernels are inference MoE GEMMs over sorted
  token ids (`kernels/moe/`, `flydsl_grouped_moe_inference.md`). The training references are Primus-Turbo's
  FlyDSL grouped kernels (`INDEX.md` § Kernel map). Turbo pins a newer FlyDSL: copy the technique, and check
  every API you use against v0.3.2.
- **Know which body each scored cell runs** before the first timing: persistent or one tile per workgroup,
  masked or whole-loop wgrad, split-K or not. A benchmark that passes one optional argument the deployment
  does not pass can optimize a different kernel (`flydsl_grouped_measurement.md` §2).
- **Write down the deployment's operating point**: groups per call, per-call tokens per group, the histogram,
  K padding, beta, eager or graph (`flydsl_grouped_measurement.md` §2).

## 3. Step 1 — name the pass and where its skew lives

1. **Is the skew in the tile count or in the cost per tile?** In the tile count (NT/NN), a group-aware XCD order
   absorbs it: a group-aligned band deal removed most of an MXFP4 NT kernel's heavy-skew penalty
   (`flydsl_grouped_scheduling.md` §3). In the cost per tile (variable-K wgrad), any XCD partition coarser than
   one tile adds imbalance in proportion to the skew: a wgrad given its NT sibling's XCD setting, and an XCD
   remap that won on balanced inputs, both collapsed under skew (`flydsl_grouped_scheduling.md` §1–§2).
2. **Are the tiles equal in cost?** Equal (NT/NN): once every dispatch round is full, dispatch order does not
   matter. Unequal (variable-K wgrad): longest-first is a live lever, but choose its direction on the device
   from the group table; reversing unconditionally loses wherever the table already descends
   (`flydsl_grouped_scheduling.md` §5).
3. **Which distributions decide?** Balanced, moderate and heavy skew, a near one-hot case and empty groups
   (`flydsl_grouped_measurement.md` §1). Take the worst of them as the score of an ordering change.

## 4. Step 2 — audit group mapping and tile order before the body

In the measured kernels, index and order changes outside the main loop were among the largest single gains.
Ask, in order:

1. **What does a tile pay to find its group?** Count the prologue instructions; a per-tile O(G) search is the
   usual culprit, and a lane-resident prefix table with a ballot count replaces it
   (`flydsl_grouped_group_mapping.md` §5). **When MFMA counts match the reference's and the kernel is still
   slower, suspect index and decode overhead first.**
2. **On an over-launched grid, does the exit test the raw workgroup id before any XCD remap?** With the remap
   first, the dead ids all land on one XCD (`flydsl_grouped_group_mapping.md` §4).
3. **Does the XCD partition match where the skew lives** (Step 1)? On a variable-K wgrad, group-major order with
   the hardware round-robin, in place of a group-interleaved order under a contiguous XCD remap, removed a skew
   collapse (`flydsl_grouped_scheduling.md` §3).
4. **Do the passes share a configuration they should not?** NT and wgrad want opposite orders; give each its
   own default and its own race.

**Judge this layer by balance counters, not by L2 hit rate.** A static partition shows as identical MFMA and L2
counters across distributions with `GRBM_GUI_ACTIVE` apart: idle CUs, not memory and not latency
(`flydsl_grouped_group_mapping.md` §4).

## 5. Step 3 — price the per-tile fixed cost against the steady state

Fit `t_tile = F + n_kblocks · P` over a K sweep (`../gemm/flydsl_gemm_grid_pricing.md` §6; its systematic errors
are in `../gemm/flydsl_gemm_attribution.md` §7). Then compare `P` with the dense kernel of the same dtype.
- In the measured grouped kernels `P` matched the dense rate and the loss sat in `F`; a kernel with the faster
  steady state still lost to a reference whose tiles ran several times as many K iterations
  (`flydsl_grouped_tile_pricing.md` §6).
- **State whether an MFMA utilization is wall or steady state.** *(field)* The same body read 60% over the wall
  and 87% in the steady-state window; the first misreads "tiles too short" as "the loop is 40% idle".
- At the power cap, cycle savings convert to wall time at a shape-dependent rate
  (`../gemm/flydsl_gemm_attribution.md` §4); on the grouped table where it was measured, the lower rate was the
  skewed worst cell (`flydsl_grouped_measurement.md` §7).

Where `F` comes from and what removes it: `flydsl_grouped_tile_pricing.md`.

## 6. Step 4 — pass-specific levers

- **Forward and dgrad:** group lookup, exit and remap order, group-aligned XCD band, boundary and tail bodies,
  per-tile `F` (`flydsl_grouped_group_mapping.md`, `flydsl_grouped_scheduling.md`,
  `flydsl_grouped_tile_pricing.md`).
- **Wgrad:** group-major order, longest-first direction, splitting the hot group's contraction, scale timing, the
  store cost of short groups, beta=1 (`flydsl_grouped_wgrad.md`, `flydsl_grouped_scheduling.md`).
- **fp8 and MX:** scale delivery and packing, preshuffle, quantization share, K padding, fused GLU/dGLU
  (`flydsl_grouped_lowprec.md`).
- **Any pass:** host launch path, graph or eager, the race's scoring points (`flydsl_grouped_host_autotune.md`).

## 7. Usually dead

Each entry held where it was measured. Reopen one only with a probe that shows its precondition changed.
- **One launch per group instead of one grouped launch.** Per-group launches were slower than one grouped
  launch; the gap to dense came from groups competing for L2 while they run concurrently, which group-major
  order addresses inside one launch (`flydsl_grouped_scheduling.md` §9).
- **A persistent wgrad that walks one tile position across all groups.** With unequal tiles and a grid divided
  by G it lost far more than it saved (`flydsl_grouped_scheduling.md` §6).
- **Pairing remainder tiles into equal-cost work.** Equalize workgroup lifetime, not MFMA count
  (`flydsl_grouped_scheduling.md` §7).
- **Interleaving groups tile by tile** (`group = tile % G`; band-level interleave is a different lever,
  `flydsl_grouped_scheduling.md` §3). Locality inside a group outweighs spreading experts over neighbouring
  workgroups.
- **Wider group spans in the XCD or L2 order.** On variable-K wgrad, every hand-set super-block span and an
  in-group contiguous remap lost to the automatic span with the hardware round-robin
  (`flydsl_grouped_scheduling.md` §2); one global `GROUP_M` across K lost where K changed (§4 there).
- **`BLOCK_N = 128` on an MXFP4 whole-loop NT body** to cut N padding timed slower, though only on a build
  whose output was not yet correct; halving the N tile halves the reuse of each A tile
  (`flydsl_grouped_tile_pricing.md` §1).
- **Occupancy-2 rebuilds of an MXFP4 grouped NT body** (8 waves splitting the same 256×256 tile): splitting the
  tile halves each wave's MFMA stream. This does not make the opposite (1 wave per SIMD with AGPR
  accumulators) a rule; that move measured both ways (`flydsl_grouped_tile_pricing.md` §2).
- **A hand-written whole-loop wgrad.** *(field)* On an MXFP8 variable-K wgrad it lost to the production chunked
  body once scales were applied (`flydsl_grouped_wgrad.md` §4); without scales it won only 5–7% on three shapes
  and lost 51–59% on the others (on one, a 16×28 tile grid turned its per-phase barrier and drain into stalls).
- **Moving the beta=1 read-back of C into the main loop** at one wave per SIMD spills
  (`flydsl_grouped_wgrad.md` §10).
- **LLVM scheduler options on a hand-scheduled grouped body** (`max-memory-clause`, `post-misched`,
  `lsr-drop-solution` on a body pinned by `s_setprio`, `sched_barrier` and explicit waits). *(field)* +0.06%
  over 24 samples; the ISA differed by 1 instruction of 4975. Diff the ISA before benchmarking any
  "zero-risk" compiler option (`../gemm/flydsl_gemm_schedule_and_locality.md` §3).

## 8. Rules that save rounds

- **A one-geometry benchmark overfits NT changes.** *(field)* One change measured on three model geometries: its
  wgrad part gained 0.7–10.9% in all six cells; its NT part gained 3.8–4.3% in three of four cells of the
  geometry it was tuned on and between −1.3% and +1.2% on the other two geometries, because its gate was an
  absolute tile count. Score across several geometries (`flydsl_grouped_measurement.md` §1) and write gates as
  ratios.
- **A gain larger than the structure allows** usually means the benchmark took a uniform-distribution shortcut
  or reused tensor identities (`../gemm/flydsl_gemm_workflow.md` § The round discipline). A gain that vanishes
  under skew is benchmark overfit; revert it.
- **When the goal is to beat a specific kernel, dump its ISA and dispatch metadata first.** Compare waves per
  SIMD, accumulator class, K iterations per tile and synchronization instructions per MFMA (waits, barriers,
  priority changes, `s_nop`) rather than MFMA utilization. *(field)* In the measured case the candidate's steady
  state was already faster and the whole gap was tile length (`flydsl_grouped_tile_pricing.md` §2, §6).
- **Carry the parent kernel's measured dead ends into the plan.** A direction that the kernel it was derived from
  measured negative is reopened only with a probe that shows a changed precondition (§7).
- **Re-measure on the current tree.** A probe tree copied before a refactor stops reaching the code it tests
  (`flydsl_grouped_measurement.md` §4).

## Verify
- [ ] The pass, its layout and where its skew lands are written down for every scored cell.
- [ ] The body each cell runs (persistent or not, masked or whole-loop, split or not) is logged.
- [ ] `F` and `P` are fitted before any main-loop work.
- [ ] Every kept change was measured on balanced and heavy-skew inputs and on more than one model geometry.
- [ ] § Usually dead was read before the plan was written; a reopened entry names the precondition that changed.

## Related
`INDEX.md` · `../gemm/flydsl_gemm_workflow.md` (round discipline) · `../gemm/flydsl_gemm_grid_pricing.md` ·
`../gemm/flydsl_gemm_attribution.md` · `../gemm/flydsl_gemm_schedule_and_locality.md` ·
`../../../../common_methodology/optimization/lever_grid_sizing.md` ·
`../../../../common_methodology/optimization/lever_xcd_locality.md`
