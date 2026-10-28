---
title: FlyDSL grouped GEMM — tile, occupancy and boundary bodies per pass, priced before building
kind: language
lever: flydsl_grouped_tile_pricing
gens: [gfx950]
dtypes: [bf16, fp8, mxfp8, mxfp4]
regimes: [forward, dgrad, wgrad, variable-k, small-m, ragged-tail, register-saturated]
updated: 2026-10-27
sources:
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:2487
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:2553
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py:3921
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:1689
  - AMD-AGI/Primus-Turbo@ca1fdd67:primus_turbo/flydsl/grouped_gemm/grouped_gemm_mxfp4_kernel.py:1716
---

# Pricing the tile, the occupancy and the boundary bodies of a FlyDSL grouped GEMM

> Grouped GEMMs on gfx950 reuse the dense 256×256 tile, and where it was measured their main loop ran at dense
> speed. What the group dimension changes is the price of everything around the loop: more tiles per FLOP,
> more ragged tiles, a per-tile prologue that looks up the group. This card prices the tile and the waves per
> SIMD per pass, the small-M dgrad tile, boundary half tiles and tail bodies, and the per-tile fixed cost.
> Round counting, split-K, the generic fixed-cost fit and the carry cost of a body variant are in
> `../gemm/flydsl_gemm_grid_pricing.md`; register classes and the K loop are in
> `../gemm/flydsl_gemm_mainloop.md`. "FP8" and "MX4" are Primus-Turbo's
> `primus_turbo/flydsl/grouped_gemm/grouped_gemm_{fp8,mxfp4}_kernel.py` at main `ca1fdd67`.
>
> Numbers marked *(field)* are measured values. Use them for direction and order of magnitude, not as targets.

## Route here when
- You want a bigger tile, more waves per SIMD, or a smaller tile for a short pass.
- A dgrad with few tokens per group is far from roofline.
- Output dimensions are not multiples of 256 and the last tile of a group or of N is mostly padding.
- A grouped kernel runs its main loop at dense speed and still trails dense by 10% or more.

**Somewhere else if:** the lookup and the dead workgroups that add to the per-tile cost →
`flydsl_grouped_group_mapping.md`; tile order and tiles per workgroup → `flydsl_grouped_scheduling.md`;
short-contraction wgrad → `flydsl_grouped_wgrad.md`; MX scale layout → `flydsl_grouped_lowprec.md`.

## 1. The register wall comes before LDS
256×256 is the largest tile that fits, for both large-tile loop designs (`../gemm/flydsl_gemm_mainloop.md` §1):
- **8 waves, 2 waves per SIMD** (one 512-thread workgroup per CU, at most 256 registers per wave). *(field)* bf16
  with a 2×4 wave grid: 128 accumulator VGPRs plus ~96 operand VGPRs is 224 (228 measured); doubling either
  tile side needs ~352. MXFP8 NT sat at 246–254 of 256 VGPRs depending on the build, with no spill: a 16-wave
  version of the same tile would need four such waves per SIMD, about 1000 registers against 512.
- **4 waves, 1 wave per SIMD** (one 256-thread workgroup per CU). The fp32 accumulators of a tile are
  `BM × BN ≤ 4 waves × 64 lanes × 256 AGPRs = 65536`: 256×256 is exactly the AGPR file. *(field)* An MXFP4
  BM = 384 would need ~620 registers per lane against 512. In the measured MXFP4 grouped kernel BM must also
  be a multiple of 128 because of its scale-slab layout (`flydsl_grouped_lowprec.md` §4).
- **So the operand fill cannot shrink by tiling.** A `buffer_load … lds` costs the 128-byte lines it covers
  (`../gemm/flydsl_gemm_datapath.md` §6); lines per unit of MFMA work go as `(BM + BN) / (BM · BN)` and do not
  depend on BLOCK_K. Only a larger tile lowers them, and the tile is at the wall. *(field)* `BLOCK_N = 128` on
  an MXFP4 whole-loop NT body, tried to cut N padding, timed 4.9–10.2% slower on a build whose output was not
  yet correct: halving the N tile halves the reuse of each A tile, which cost more than the padding saved.
- **At the wall, anything that lengthens a live range spills.** *(field)* MXFP8 NT at 246/256 arch VGPRs:
  folding the store earlier spilled 24 VGPRs, a tile loop 25, holding an A fragment across iterations 44,
  each 19–26% slower. A second heavy body in the same kernel compiled to 256 VGPRs with 23 spills in the MFMA
  stream's accumulators; two copies of the same body did not spill, one full plus one paired body did.
- **Gate on a compile-only ISA dump** (20–60 s): `.vgpr_count`, `.agpr_count`, `vgpr_spill_count`,
  `group_segment_fixed_size`, and the `s_barrier` count. Do not time a candidate that spills
  (`../gemm/flydsl_gemm_mainloop.md` §10).

## 2. Waves per SIMD: decide per pass
- On fp8 per-tensor, the 4-wave design (1 wave per SIMD) won every variable-K wgrad point against 8 waves (2
  per SIMD), a minority of the NN dgrad points and almost no forward points (`flydsl_grouped_wgrad.md` §8). It
  pays where both operands are transpose-read (`ds_read_b64_tr_b8`; mechanism in `flydsl_grouped_wgrad.md` §8);
  the forward pass reads both operands K-contiguous.
- The fastest grouped kernel measured is that wgrad: 4 waves, 1 wave per SIMD, 256 AGPR accumulators, a full
  160 KB of LDS, 0.13 synchronization instructions per MFMA against 0.93 for an 8-wave MXFP8 NT *(field)*.
- **1 wave per SIMD with AGPR accumulators is a direction to test, not a rule.**
  - *(field)* Moving a kernel to 4 waves while keeping the 8-wave phase skeleton (6.4 `s_barrier` per K
    iteration, `s_setprio`) lost 31% and 41%. Compile-only gate: at most 3 `s_barrier` per K iteration and no
    `s_setprio`; if the gate fails, the attempt says nothing about 4 waves.
  - *(field)* An MXFP8 grouped kernel at 32 groups of 4096 rows, moved to AGPR-tied accumulators (arch VGPRs
    246 → 184, no spill), measured 0.03–0.17% slower: no gain in that regime.
  - Freed registers buy nothing by themselves (`../gemm/flydsl_gemm_mainloop.md` §4).
- **On MXFP4 NT, more waves per SIMD by splitting the tile lost.** *(field)* An 8-wave kernel at 2 waves per
  SIMD ran at 0.83–0.98× of the 4-wave whole loop when built as a tiled software pipeline and 0.78–0.92× when
  built as a whole loop; in the whole-loop case MFMA busy cycles were identical and MFMA utilization fell from
  57.3% to 48.2%. Each wave's MFMA stream halves, doubling the share of `ds_read` → MFMA bubbles, and the
  waves contend for the same LDS read port. Turbo keeps its MXFP4 wgrad at one wave per SIMD (`OCC = 1`, MX4:1689).
- **The 8-wave NT barriers that look removable are load-bearing for speed.** On MXFP8 NT the pre-MFMA rendezvous
  added no ordering the gates could see (the barrier after the MFMAs guards LDS reuse), but it kept the waves' MFMA
  bursts in phase; without it they drift and contend for issue and LDS. Whether an opening rendezvous orders
  anything differs per kernel (`flydsl_grouped_correctness.md` §8). *(field)* MXFP8 NT: removing the pre-MFMA
  rendezvous barriers was 3.3% slower overall (2.6–6.9% on all 8 NT configs), removing a steady-state barrier 6.1%
  (worst config 8.7%), merging MFMA phases 1.6%, all with SNR and determinism intact. Before deleting a barrier, ask
  whether it is an ordering constraint (`../gemm/flydsl_gemm_mainloop.md` §2) or a phase-alignment device. Deleting
  the rendezvous of the last K steps, which issue no new fill but still make the fills issued two steps earlier
  visible across waves, raced (`flydsl_grouped_correctness.md` §8).

## 3. Small-M dgrad: when BLOCK_M = 128 pays
- With few tokens per group a dgrad grid of 256-row tiles leaves CUs idle. Halving BLOCK_M doubles the M tiles,
  and it pays only while the whole 128-row grid still fits in one round:
  `G · ceil(pm / 128) · ceil(N / 256) ≤ N_CU` with `pm = M_total // G` (the host sees no per-group length).
  Turbo's fp8 dgrad takes a fixed BLOCK_M = 128 config inside that gate and races 256-row candidates outside
  it (`_np_regime`, FP8:2487-2494; FP8:2553-2555).
- *(field)* Past the gate it falls off a cliff: on fp8 NN dgrad, BLOCK_M = 128 lost once its grid needed 320
  tiles on 256 CUs, and at about three times that M the 256-row tile was 58% faster. At production sizes
  BLOCK_M = 128 was 21–25% slower on fp8 NN dgrad and 25–35% slower on small fp8 forward shapes.
- Recompute the grid bound from BLOCK_M. *(field)* An MXFP8 NT sweep whose bound was hard-coded for 256 rows
  launched half the tiles at BLOCK_M = 128 and read 9–35% "faster"; with `ceil(M_pad / BM) + G` it was 1.55×
  slower than the per-tensor reference kernel (1666 vs 2576 TF/s). Before believing a tile change, show the
  full grid ran (`flydsl_grouped_correctness.md`).

## 4. Boundary half tiles: what to delete, what to keep
When `OUT % BLOCK ≠ 0` and `OUT % BLOCK ≤ BLOCK / 2`, the back half of the last block is padding: its
quadrants' MFMAs and LDS reads are dead. Pick the variant with a scalar branch on `readfirstlane(block == last)`.
- **Start by deleting only `ds_read` + MFMA** and keep every g2s, `s_barrier` and `vmcnt` identical, so there
  is no new race surface. Stores of the skipped quadrants can stay unconditional; the descriptor drops them.
  *(field)* MXFP8 wgrad: 6.25–8.16% of the MFMAs on paper, +1.3–4.1% measured (about half: barriers, fills and
  LDS round trips stay). NT, N side only: the worst config +1.9%.
- **Count on 20–50% of the MFMAs saved.** *(field)* fp8 wgrad: a short-N body that saved 6.25% of the MFMAs
  returned 0.6–1.2%, because the L2 hit rate fell 6.4 points and MFMA-busy 5.4 points.
- **In a whole-loop asm kernel, keep each variant's register footprint.** Variants that pin register windows
  share one allocation. *(field)* fp8 wgrad: one variant that dropped a dead operand pool (and its `ds_read`s)
  on 12 of 276 tiles made the whole kernel spill 16 VGPRs and the pass 14.6% slower. On MXFP4, deleting
  MFMAs plus one g2s group kept the spill count at 0. Check the spill count after adding any variant.
- **Deleting the dead half's g2s: lower `vmcnt` by the number deleted, when those loads are the youngest in
  flight.** The in-flight allowance is "loads issued per iteration minus the earliest group".
  - *(field)* MXFP4 whole loop: `vmcnt(WLV − n_deleted)`, `WLV` being the full body's allowance, on the half-N body,
    +3.8% geomean, +4.7% worst config. MXFP8 NT (all pools prefetched two iterations ahead, the deleted pool issued
    last): drain 6 → 4, the down configs 0.985 → 1.005 against the reference.
  - *(field)* The same surgery on MXFP8 wgrad, where the deleted pool is one iteration ahead and issued first:
    0.09% slower.
  - *(field)* fp8 NN dgrad (8 waves): the reduced counts raced at K = 1024. Keeping the loads and deleting only
    MFMAs and stores reused the full body's graded drain and beat the fully drained no-load body by 0.5 points.
    Prove any changed count by repetition at small K and wide N (`../gemm/flydsl_gemm_correctness.md` §3).
- **The barrier pair around an empty phase can go with its quadrant.** *(field)* NT half-N body: 8 → 5
  barriers per K iteration, +1.0–1.5% on the down configs; on wgrad, with 4 groups and a saturated grid, the
  same change measured 0.00%.
- **The two M halves of a tile sit on different waves.** Idling `wave_m == 1` does not shorten the tile;
  re-assign the waves to live rows instead (`../gemm/flydsl_gemm_grid_pricing.md` §8). *(field)* MXFP4 wgrad:
  +0.4–3.1% on the deployment-like configs (median +1.7%). Turbo's MXFP4 wgrad serves a block ragged in both
  dimensions with the N re-tile (MX4:1716-1722).
- *(field)* fp8 per-tensor wgrad, last M block with its upper `a` half all padding, `a_halves = 1`: +4.5–5.8%.
- **Is the boundary tile on the makespan?** A cheap boundary tile on wgrad finishes early, which is why the
  barrier trim above did nothing there. The M-side bodies above were measured at 18–35 dispatch rounds, while
  Turbo's fp8 wgrad keeps its M-side boundary body only below 8 rounds: "a cheap tile frees its CU early and
  desyncs the L2-slab phase cohorts (no resync at occ=1) -- pays off shallow, not deep" (FP8:3921-3925;
  `_WGRAD_AFF_ROUNDS`, FP8:4660). Measure at the deployed depth.
- **Leave boundary tiles where they are.** *(field)* Moving the half-price tiles (an N-first band walk, or the
  ends of the groups) cost 0.8–1.0 points of band locality. Peeling the boundary strip into a second dispatch
  was 2.9% slower overall and 4.6% on the worst config (MXFP8 NT): the strip loses the A slab it shared in L2.

## 5. Tail bodies: compute the hit rate first
The carry cost of a body variant (code size paid by every shape in the module) is in
`../gemm/flydsl_gemm_grid_pricing.md` §8. What the group dimension adds is that the hit rate depends on the
pass and on the token histogram:
- A ragged wgrad output (`OUT_M`, `OUT_N`) is a compile-time property: every call hits the tail body.
- A ragged forward/dgrad M tail exists per group and depends on the group lengths. *(field)* Under the scored
  skew 19 of 529 M blocks (3.59%) had a remainder ≤ 128 rows; under balanced tokens, none. With the tail tile
  ~17% faster the bound was 0.6–0.7 points on each of two cells. Measured, the two skewed cells gained 0.44
  and 0.30 points, the two balanced cells of the same module lost 0.31 and 0.34 to the carry, and the
  geomean moved −0.13%.
- *(field)* Two tail bodies with 32 and 64 MFMAs per phase both made their tail tile ~17% faster: the saving is
  the tail tile's fixed cost, not its MFMAs.
- *(field)* Disabling the half-N body of an MXFP4 kernel cost 1.79 points of the 8-cell geomean (NT and wgrad
  cells, 7 of 8 in the same direction); re-tiling it to quarter-N gained another 0.67–1.09 points per NT cell.

## 6. Short tiles: where grouped loses to dense
Fit `T_tile = F + n · P` from a K sweep, n K iterations at P each (`../gemm/flydsl_gemm_grid_pricing.md` §6, which
writes it `a · n_kblocks + F`; its pitfalls: `../gemm/flydsl_gemm_attribution.md` §7). For grouped kernels the
answer has been the same every time:
- *(field)* MXFP4 grouped: P = 1.565 µs per K phase against 1.59 µs derived from the dense MXFP4 kernel's
  5405 TF/s. The steady state ran at dense speed; the whole loss against dense was in F = 7.81 µs per tile.
- *(field)* MXFP8 NT against the fastest fp8 wgrad: 6192 tiles of 23 K iterations against 1008 of 134; F was
  15.7% of the wall against ≤3.7%, while MXFP8's P (1.164 µs) beat the reference's (≤1.35). Pushing F's
  share to the reference's level predicts the wall 779 → 673 µs, a ratio of 1.06 — enough on its own. The
  same table carried the control: two of the three cells with 45 K iterations per tile passed, none of the
  nine with 23.
- *(field)* That F split into launch 1.39, group decode 0.85, C store 2.37 and cold fill 0.80 µs.
- What the group dimension puts into F: the group lookup (`flydsl_grouped_group_mapping.md` §5), dead
  workgroups (§3 there), ragged tiles per group (§4 here), and a cold L2 for each group's slab
  (`flydsl_grouped_scheduling.md` §9).
- So the levers are F (lookup, decode, store overlap, boundary bodies) or longer tiles per workgroup
  (`flydsl_grouped_scheduling.md` §6–§7), not the main loop.
- For wgrad the grid is `G × tiles per group`, so G sets the last-round fill: 144 tiles per group fill 18.0
  rounds of 256 CUs at G = 32 and 13.5 at G = 24. Count rounds for every G you deploy
  (`../gemm/flydsl_gemm_grid_pricing.md` §1).

## 7. Store-bound short contractions
A variable-K wgrad writes G full output slices whatever the token split, so groups with few tokens make it
store-bound. The short-contraction blocking with non-temporal C stores, and its crossover, which scales with
tiles per group rather than with a token count, are in `flydsl_grouped_wgrad.md` §6.

## Verify
- The compile-only dump shows no spill and the expected barrier count before any timing.
- The waves-per-SIMD choice is per pass, with the 4-wave sync gate passed before a 4-wave result is read.
- A small-M tile is used only inside the one-round gate, and the grid bound follows BLOCK_M.
- Every boundary variant keeps the race surface (or proves its changed counts at small K and wide N) and the
  spill count, and its hit rate is computed on the scored distribution.
- A K sweep separates F from P before work on the main loop.

## Related
`flydsl_grouped_group_mapping.md` · `flydsl_grouped_scheduling.md` · `flydsl_grouped_wgrad.md` ·
`flydsl_grouped_lowprec.md` · `flydsl_grouped_correctness.md` · `flydsl_grouped_measurement.md` ·
`../gemm/flydsl_gemm_grid_pricing.md` · `../gemm/flydsl_gemm_mainloop.md` · `../gemm/flydsl_gemm_datapath.md` §6 ·
`../gemm/flydsl_gemm_attribution.md` §7 · `../../../../hardware/mi350_execution.md`
