---
title: FlyDSL grouped GEMM — ordering and placing tiles across groups
kind: language
lever: flydsl_grouped_scheduling
gens: [gfx950]
dtypes: [bf16, fp8, mxfp8, mxfp4]
regimes: [forward, dgrad, wgrad, variable-k, skew, persistent]
updated: 2026-10-27
sources:
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/utils/gemm_helper.py:688
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/utils/gemm_helper.py:1639
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/utils/gemm_helper.py:2694
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:102
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:2024
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:3898
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:4211
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:4652
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel.py:263
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel.py:427
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_bf16_kernel.py:622
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:104
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:593
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:1086
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:1164
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:1725
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:1866
---

# Ordering and placing tiles across groups in a FlyDSL grouped GEMM

> Which XCD runs a tile, in which order the groups' tiles are dispatched, and how many tiles a workgroup walks
> decide how a grouped GEMM behaves when the token counts per group are skewed. The levers here are pure
> permutations of the tile index or of the grid; they leave every MFMA and every output byte unchanged, so
> they are cheap to try and easy to get backwards. XCD basics, 2D bands and bijection proofs are in
> `../gemm/flydsl_gemm_schedule_and_locality.md` §2; dispatch order as a list schedule is in
> `../attention/flydsl_attn_grid_dispatch.md` §4. "FP8", "BF16" and "MX4" are Primus-Turbo's
> `primus_turbo/flydsl/grouped_gemm/grouped_gemm_{fp8,bf16,mxfp4}_kernel.py` and "GH" is
> `primus_turbo/flydsl/utils/gemm_helper.py`, at main `ca1fdd67`.
>
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as targets.

## Route here when
- A grouped kernel is fast on balanced token counts and slow under skew, or the other way round.
- You are choosing an XCD remap, a band width or a dispatch order, or porting one from dense or another pass.
- You are deciding between one tile per workgroup, a persistent grid and several tiles per workgroup.
- A grouped GEMM trails a dense GEMM of the same total size.

**Somewhere else if:** the device-side exit and lookup → `flydsl_grouped_group_mapping.md` (exit on the raw
id and remap over the live count is §4 there); tile size and boundary bodies →
`flydsl_grouped_tile_pricing.md`; split-K for deep groups → `flydsl_grouped_wgrad.md`; ranking candidates on
balanced and skewed data → `flydsl_grouped_host_autotune.md`.

## 1. First ask where the skew lives

| Pass | Skew changes | Order lever | XCD lever |
|---|---|---|---|
| forward, dgrad | tiles per group | none while rounds are full | group-aligned runs (§3) |
| variable-K wgrad | contraction per tile | longest-first (§5), 1 tile per WG (§6) | round-robin (§2) |

Forward and dgrad tiles all cost the same except ragged boundary tiles (`flydsl_grouped_tile_pricing.md` §4); wgrad
groups all have the same tile count.

- *(field)* Copying the forward settings (GROUP_M = 2, eight contiguous XCD slices) to the variable-K wgrad of
  the same MXFP4 kernel ran wgrad at 0.38–0.55× speed: under skew the slices gave one XCD the four largest
  groups, 86% of the work.
- On wgrad, any partition coarser than one tile leaves up to one run of the heaviest group as imbalance. With
  276 tiles per group and runs of 8 tiles, one run per XCD is 23% of a group's tiles; times a hot group's 62%
  share of the work that is 14% of the launch. Only the hardware's own `bid % 8` (run length 1) is exact.
- Do not share one default configuration, or one race, between the forward/dgrad path and wgrad.

## 2. XCD placement: which way the remap should run
The rule from dense GEMM: a tile stream that reads **a slab of its own** wants its group's workgroups on one
XCD; streams that read **a shared slice** want the hardware round-robin
(`../gemm/flydsl_gemm_schedule_and_locality.md` §2). For grouped passes:
- **Forward and dgrad read a private slab.** Every M tile of group g streams `B[g]`. When `B[g]` exceeds an
  XCD's 4 MB L2, co-resident workgroups that share a B column block should sit on the same XCD: contiguous
  slices with a narrow band. *(field)* fp8 NN dgrad, 8.7 MB slab: the shared-slice recipe (identity XCD order)
  lost 2.3–5.7% depending on the band. The slab size alone did not decide the sign: on the same pass, 8
  contiguous slices with GROUP_M = 4 gained 0.79% at K = 2944 and lost 1.91% at K = 5760 (17 MB slab). Treat K
  as part of the key.
- **Contiguous slices trade balance for locality.** *(field)* MXFP8 NT: 8 slices made the heavy-skew down
  configs 2.8–3.0% slower, balanced unchanged. MXFP4 NT went the other way: 8 slices beat 1 by 7–8% on the
  two worst (skewed) configs and lost 2.3–3.2% on the balanced down configs, and fewer slices dropped one
  wide-N short-K forward config from 1.373 to 1.223. Sweep per pass, and put both a narrow-N and a wide-N
  short-K shape in the screen: a screen with only a long-K shape missed that collapse.
- **Variable-K wgrad reads a shared slice.** Both operands are token-major slices of the group, shared by all
  of its output tiles, so XCD clustering fragments the sharing. *(field)* fp8 wgrad: identity order with
  GROUP_M = 2 beat XCD-affine rectangles and wide bands by 0.8–1.0%; GROUP_M = 1 lost 5.1% (co-resident
  workgroups no longer share rows). A contiguous remap applied inside each group, keeping the group order,
  still lost 0.6% overall and 2–3 points on two configs: round-robin keeps all eight XCDs on the same band.
  That measurement (G = 32, 144 tiles per group, ~18 rounds) is at the depth where Turbo's tuner leads with
  the affine rectangle (§3, last bullet); race both on the deployed shape.
- **A contiguous remap also undoes a longest-first order** on variable-K wgrad (measured: `flydsl_grouped_wgrad.md`
  §5). A race scored on balanced data alone picks the losing arm; score on the worse of balanced and skewed.
- **Affinity that stays balanced.**
  - Decide per band, on the device, from the offsets: XCD affinity only for bands whose groups are even.
    *(field)* fp8 wgrad: +0.18% over no affinity, where affinity for every band lost 0.6%.
  - Rotate which class serves the heavy side. When super-blocks of `gp` groups are dealt to the XCD classes and
    the heavy group of each super-block always lands on the same classes, advance the position by one per
    super-block (class stride `k+1`). The stride must be odd: ragged boundary tiles are cheaper, and an even
    stride freezes them on fixed classes. *(field)* fp8 wgrad: +5.3% on a strongly skewed config (an offline
    greedy model predicted +5.0%), the mildly skewed configs unchanged, balanced slightly faster, one extra
    SGPR. Turbo rotates each group's tiles by an odd step coprime with the 16-tile CU-residue class
    (`_WG_TILE_ROT = 5`, FP8:2053, 4652) and gives its XCD-affine super-block exactly one group "so every XCD
    holds the same fraction of every group's K" (FP8:3898-3901).
  - Do not hand-tune the super-block span. *(field)* fp8 wgrad, one projection: `gp` 4 / 8 / 1 against the
    automatic 2 were 0.5%, 9.2% and 5.8% slower.
- **Thin groups are a different regime.** Turbo's MXFP4 NT checks whether a group is at least
  `num_xcds × span` M blocks wide. Below that every XCD sees a different expert, and it switches to a blocking
  with non-temporal C stores, because "the write-only C stream evicts re-read weights" (MX4:1100, 1164-1170).

## 3. Two schemes both called "band-cyclic"
They permute different things. Name them apart.

| | XCD band deal | group interleave |
|---|---|---|
| Permutes | which XCD runs a tile | when a tile is dispatched |
| Rule | XCD x gets runs x, x+8, … of `band` ids (GH:2694) | band 0 of all groups, then band 1, … (FP8:2024) |
| In Turbo | bf16 forward/dgrad/wgrad, MXFP4 NT (forward and dgrad) | fp8 masked wgrad fallback (`_WG_INTERLEAVE`, FP8:102) |

**XCD band deal.** Contiguous slices and per-tile round-robin are the two ends of one axis, the run length.
Contiguous parks a run of small groups (no B reuse) on one XCD; per-tile scatters a band's reuse over eight.
- *(field)* MXFP4 NT, 32 groups of 16 M blocks under uniform tokens. Run of 2 M blocks (one GROUP_M band):
  ~10% slower balanced. Run of 16 (one group): balanced 0–0.8% slower, heavy skew 4.5–6.1% faster. Run of 32:
  heavy skew only 3–4% faster. Net +1.6% geomean, +4.8% on the worst config; the heavy-vs-balanced penalty
  fell from 6.7–7.4% to 0–1.8%, with `SQ_INSTS_MFMA` and the L2 footprint bit-identical.
- Rule: a run is the M blocks of one group under uniform tokens, with at least 4 runs per XCD. A run that
  crosses a group boundary pulls two slabs into one XCD. Turbo's MXFP4 NT uses the group-aligned run when it
  divides every group's M-block count and falls back to a narrow band otherwise (MX4:593-629).
- The band size matters more than the XCD count. *(field)* bf16 wgrad: a band of one N row inside a group
  beat the identity order (worst config +1.05%); a band of a whole group's tiles left the worst config 15%
  below the narrow band under skew. Turbo's bf16 wgrad keeps "runs short enough to stay inside one expert, so
  skew spreads" (BF16:431). On bf16 forward/dgrad a band of M blocks gave +2.6% *(field)*.

**Group interleave.** It keeps every group in flight at once.
- *(field)* fp8 per-tensor wgrad at identity XCD order: group-contiguous dispatch was 20%, 34% and 59%
  slower at 2:1, 4:1 and 32:1 skew than balanced. Interleaving cost 0.9% balanced and cut the 30:1 penalty
  from 46% to 26%.
- It also dispatches a hot group's later bands late: with three bands per group, the third starts 67% of the
  way through the stream. Combined with contiguous XCD slices it stacks the hottest groups on XCD 0.
  *(field)* MXFP8 wgrad: interleave plus 8 contiguous slices was the skew collapse; group-major order with the
  identity XCD order took the heavy down config from 0.474 to 1.005 against its reference (+112%), the worst
  config of the table from 0.474 to 0.928, and balanced from 0.955 to 1.001. Group-major is longest-first
  whenever group sizes fall with group index.
- Never interleave per tile. *(field)* bf16 wgrad with `group = tile % G`: 1081 → 833 TF/s balanced,
  1048 → 482 at 8:1 skew. In-group L2 locality outweighs balance.
- Turbo keeps interleave only in its masked fallback; its 4-wave wgrad keeps group-major order under the
  hardware XCD split (`num_xcd = 1`, FP8:4654-4656) and, on launches of at least 8 dispatch rounds, leads its
  race with an XCD-affine rectangle inside one group (`_WGRAD_AFF_ROUNDS`, FP8:4657-4663, 4693-4720).

## 4. Band width (GROUP_M) against K
- Two terms pull against each other: B traffic falls as 1/GROUP_M (each band re-streams B), and the band's A
  footprint `GROUP_M × BLOCK_M × K × bytes` must fit an XCD's 4 MiB L2. Only K moves the second term.
- *(field)* MXFP8 NT: at K = 2944 (A band 2.88 MiB at GROUP_M = 4) the peak was GROUP_M = 4; at K = 5760
  (5.62 MiB) it was 8 — GROUP_M 2/4/8/16 scored 1.010/1.036/1.045/1.031. "A band ≤ 4 MiB" alone would pick 2,
  the worst point. One global GROUP_M = 8 cost 0.21% on the geomean: 1.1% faster on one config and 3.0% slower
  on another K.
- Key the choice on K: switch to the wider band when `GROUP_M × BLOCK_M × K` crosses 4 MiB, with K in the
  autotune key (`flydsl_grouped_host_autotune.md`).
- It does not carry across layouts. *(field)* The same K rule on fp8 NN dgrad cost 1.2 points: NN reuses an
  N stripe of `B[K,N]`, not an A slab.
- On MXFP4, keeping B resident with a 2D band lost to a 1D band that sweeps all of N for each row pair. Turbo
  records a 6-wide `group_n` band losing on all four NT cells; on wgrad the 1D band was 3.7% faster, with an
  L2 hit rate of 62.7% instead of 57.9% and 2.49 GB of HBM reads instead of 2.86, at identical request and MFMA
  counts (MX4:1102-1110, 1125-1138).
- *(field)* bf16 forward/dgrad: GROUP_M 1/2/4/8 within 1% once the XCD band deal was in.

## 5. Longest-first inside a wave
- Forward and dgrad tiles cost the same: *(field)* order changes read exactly 0 where every dispatch round
  was full. Variable-K wgrad tiles cost what their group's tokens cost, so the dispatch order is a list
  schedule (`../attention/flydsl_attn_grid_dispatch.md` §4).
- Three ways to put the deepest group first, all computed on the device from the offsets:
  - Rank all groups in one wave: lane g holds group g's length, the stable descending rank takes G `readlane`
    compares, and slot → group is a one-hot ballot plus a popcount (`wave_rank_desc_stable`,
    `wave_lane_with_rank`, GH:688-709). bf16 wgrad maps `tile // TILES_PER_GROUP` through it (BF16:424-433);
    it needs G ≤ 64.
  - Rank groups by token span the same way in a split launch (FP8:4211-4230).
  - Pick one of the two walk directions from two half-sums of the offsets: reverse the group-major stream when
    the second half of the groups holds more tokens (MX4:1866-1876).
- *(field)* MXFP4 wgrad: the direction choice gained 1.65% and 2.3% on the two cells of a 10× skew, others
  bit-identical. An unconditional reverse gained 3.4–6.0% there but was 7% and 19.8% slower on the configs
  whose histogram already descended, where group-major was already longest-first.

## 6. Persistent grid or one tile per workgroup
- With one workgroup per CU (the usual grouped tile), one tile per workgroup makes the hardware dispatcher
  the list scheduler: the next tile goes to whichever CU frees first (MX4:1112-1115). A persistent worker
  fixes the assignment at launch.
- *(field)* fp8 per-tensor wgrad under heavy skew: a grid capped at 2× the CU count left a workgroup that drew
  a hot group's long tile with ~14 more tiles queued behind it. One tile per workgroup was ~26% faster on the
  heaviest config and 8% on the moderate one, unchanged elsewhere, byte-identical. On its scored shapes a grid
  of exactly the CU count was flat, and every grid between the CU count and the tile count was 1.3–4.9% slower.
- *(field)* Truncating to a persistent grid lost wherever it added a tile loop to a kernel at its register
  wall: MXFP8 NT 0.70–0.81× speed (twice; the second time with an O(1) lookup, so decode cost was not the
  reason), variable-K wgrad 11–18% slower. The NT attempts showed 23–25 spilled VGPRs in the ISA each time:
  the loop keeps the group table, block id, grid size and live count alive across the main loop at 246/256
  arch VGPRs.
- *(field)* bf16 wgrad: a persistent grid-stride loop was 8–18% slower and no flatter under skew.
- *(field)* Persistent across groups (one workgroup walks one tile position through all G groups; grid
  divided by G), MXFP8 variable-K wgrad: its time against the reference kernel went from ~1.02× to 1.23–2.00×.
  276–448 tiles per group is 1.1–1.75 rounds on 256 CUs, and the 8× unrolled body thrashed the instruction cache.
- *(field)* fp8 forward/dgrad: non-persistent was ~11% faster than a persistent `scf.for` tile loop, but only
  with the L2 tile swizzle kept (without it one small-K shape was 13% slower). fp8 4-wave wgrad at one wave per
  SIMD, balanced loads: within ±2% either way.
- When it can pay: free registers and a loop that prefetches the next tile across the boundary. *(field)* 1
  wave per SIMD with AGPR accumulators left 72 arch VGPRs free on one MXFP8 kernel but only 32 on plain MXFP4 NT,
  where 2 and 4 tiles per workgroup compiled without spills and still lost (§7). *(field)* On a dense
  GEMM at one workgroup per CU, pairing two tiles alone was 0.78% slower, and pairing with the peeled tail's
  prefetch aimed at the next tile was 0.69% faster than one tile per workgroup: the value of a persistent
  worker is the next tile's fill overlapping this tile's epilogue (`../gemm/flydsl_gemm_mainloop.md` §3,
  persistent carry), not the launch it saves. Gate on a compile-only spill check
  (`flydsl_grouped_tile_pricing.md` §1).
- Repeating a ping-pong tile body breaks barrier parity. A body in which one wave group issues an extra
  `s_barrier` (`if wave_m == 1`) is consistent only once per workgroup; from the second tile the groups pair
  with the wrong barrier and the output is noise. Balance the count at the tile boundary; Turbo's bf16 wgrad
  takes a full barrier there when persistent (BF16:263-268).
- Turbo's capped persistent grids (`cap_cu`, `num_cu > 0`) exist to leave CUs to a co-running kernel, not for
  speed (BF16:622-625; FP8:2669-2670).
- Which variable-K loop body to run (chunked masked or exact-trip) is keyed on the per-group contraction
  `m_total / G`, not on `m_total` (`flydsl_grouped_wgrad.md`).

## 7. Two tiles per workgroup
- What it buys is the per-workgroup fixed cost (launch, decode, table load, pipeline fill) shared by WGT
  tiles; what it gives up is the dispatcher's dynamic balance. Turbo ships `wg_tiles = 1` in all three of its
  MXFP4 wgrad tuples (MX4:1112-1122) and one tile per workgroup on MXFP4 NT (MX4:1086).
- If you pair, gate on the paired grid width, not on how many rounds a group spans: `TOTAL / WGT ≥ N_CU` with
  TOTAL divisible by WGT (MX4:1725), and take the two tiles from opposite ends of the stream, so a heavy one
  pairs with a light one.
- On MXFP4 variable-K wgrad the sign has followed the rest of the kernel; re-measure whenever the boundary
  bodies or the C store change.
  - *(field)* With half-price M-tail bodies and whole-line `nt` folded stores in place, WGT = 2 gained 1.8–2.3
    points on the 10× skew and 0.7–1.0 on moderate skew; without them it lost 0.7–1.2% on heavy skew. WGT 3
    dropped the heavy config to 0.872 and 4 was worse; run-to-run spread on heavy tripled.
  - On another build, one tile per workgroup beat WGT = 2 on three of four cells (up to 2.7% under skew) and
    lost 1.05% on one balanced cell; WGT 4 and 8 lost up to 7.5% under skew and 11.9% balanced (Turbo's
    comment, MX4:1112-1121).
- *(field)* Plain MXFP4 NT, two tiles per workgroup with one emitted body: 1.1–3.7% slower on all four cells
  (0.9% on one cell in another build, 1.8% there with four tiles); +30 instructions, identical
  MFMA/load/store counts, VGPRs 480 → 512 (MX4:1086-1099). Static pairing locks in the imbalance of the cheap
  boundary tiles that the dispatcher would otherwise absorb.
- Do not pair remainder tiles into "equal-cost" tiles. *(field)* fp8 wgrad, one workgroup running two
  half-height tiles back to back: MFMAs exactly 3.89% fewer, output bit-exact, wall 20% slower, L2 hit rate
  68.0% → 44.5%, DRAM reads +70%. A half-height body is 0.5× the MFMAs and 0.75× the bytes but 1.0× the K walk,
  so the pair lives ~1.5× as long; with 32 co-resident workgroups per XCD it straddles two runs, and the live
  slabs went from 12 to ~24. Co-residency is aligned in workgroup lifetime, not in MFMA count.

## 8. Launch skew
Turbo's MXFP4 NT makes workgroup `bid < 128` run `bid` iterations of `s_sleep 2` (65–128 clocks each in the
CDNA4 ISA) before starting and lets the rest start at once (MX4:104-130). Against a ramp over all 256 ranks,
its paired rounds record the NT total 0.88% less time (better in 6 of 6 rounds), but no delay at all took
0.84% less time against the same ramp: over no skew, the 128-rank ramp is level on the total and differs only
on single dgrad cells (MX4:102-115). Measure a launch skew against no delay, not against another ramp.

## 9. Why a grouped GEMM trails one large GEMM
*(field)* fp8 per-tensor forward on one shape: the grouped kernel at G = 1 reached 2966 TF/s against 3044 for
a vendor dense GEMM. At G = 8 it reached 2867 with its best order and 2740 with contiguous XCD slices; eight
per-group launches of it gave 2836, and eight per-group launches of the vendor GEMM 2983.
- The loss is L2 temporal locality, not bytes. In one launch, CUs run different groups at the same time and
  their weight slabs compete for L2; per-group launches keep one slab hot but pay launch and ramp per group.
- The group-major order with the identity XCD order approximates "one group at a time" inside one launch,
  which is why it is the base the other levers in this card are measured against.

## Verify
- The pass is classified (skew in tile count, or in contraction per tile) before any order change.
- Every remap is a bijection over the live tile count and is scored on balanced and skewed data separately.
- Run lengths of an XCD band deal are group-aligned with at least 4 runs per XCD.
- Persistent or multi-tile variants passed a compile-only spill check before timing.
- Order changes are checked bit-identical: they must not change a single output byte.

## Related
`flydsl_grouped_group_mapping.md` · `flydsl_grouped_tile_pricing.md` · `flydsl_grouped_wgrad.md` ·
`flydsl_grouped_host_autotune.md` · `flydsl_grouped_measurement.md` ·
`../gemm/flydsl_gemm_schedule_and_locality.md` §2 · `../gemm/flydsl_gemm_grid_pricing.md` §1, §10 ·
`../attention/flydsl_attn_grid_dispatch.md` §4 · `../../../../hardware/mi350_chiplet.md` ·
`../../../../common_methodology/optimization/lever_xcd_locality.md` ·
`../../../../common_methodology/optimization/lever_grid_sizing.md`
