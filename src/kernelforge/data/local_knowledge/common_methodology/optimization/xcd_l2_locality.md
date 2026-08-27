---
title: XCD / L2 locality (chiplet-aware grid mapping)
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, training, both]
updated: 2026-06-05
---

# XCD / L2 locality

## TL;DR
MI300X is a chiplet GPU: **8 XCDs (Accelerator Complex Dies)**, each with its **own L2 cache** and a
slice of CUs. Workgroups that share input data should land on the **same XCD** so they hit a shared L2
instead of crossing the die boundary to another XCD's L2 / the Infinity Fabric. The practical recipe:
launch **≥1024 workgroups** (feed all XCDs/CUs), use **8-multiple tile counts** (so the round-robin grid
splits evenly across 8 XCDs), and apply an **XCD-aware / swizzled CTA order** so consecutive program-ids
that reuse the same B-tile co-locate on one XCD. See `[[hardware/cdna3_mi300/xcd_chiplet.md]]`,
`[[hardware/shared/l2_xcd_swizzle.md]]`, and `[[operators/dense_gemm/tuning.md]]`.

## Concepts (the hardware)
- **8 XCDs / MI300X**: each XCD has ~38 CUs (304 total) and a **partitioned L2** local to that die.
  L2 is *not* unified across XCDs — a miss serviced from another XCD's L2 or HBM costs more.
  CDNA4/MI350X keeps the multi-XCD design with a different CU/XCD count
  (`[[hardware/cdna4_mi350/arch.md]]`).
- **Round-robin block dispatch**: the hardware assigns workgroup ids to XCDs in round-robin; the
  *default* linear program-id → XCD mapping scatters data-sharing blocks across all 8 dies, defeating L2
  reuse.
- **L2 partition**: keeping the working set of a tile within one XCD's L2 turns repeated B/A re-reads
  into L2 hits.

## The levers
- **≥1024 workgroups**: ensures every XCD and CU gets work; sub-1024 grids leave dies idle on prefill
  (`[[operators/dense_gemm/tuning.md]]`, `[[optimization/wave_and_grid_sizing.md]]`).
- **8-multiple tile counts**: choose grid dims so the number of tiles is a multiple of 8; round-robin
  then balances exactly across the 8 XCDs, no straggler die.
- **Swizzled / XCD-aware CTA order**: remap `pid → (xcd, local_id)` so blocks reusing the same B column
  panel (same N-tile, varying M) are issued to the **same XCD** and reuse its L2. This is the
  GEMM "grouped"/super-grouping launch (triton `GROUP_SIZE_M` is the row-grouping form of the same idea;
  XCD swizzle is the die-grouping form). See `[[hardware/shared/l2_xcd_swizzle.md]]`.
- **8-multiple tile sizes** (BLOCK_M/N): a separate but compounding rule — keep BLOCK_M/N multiples of 8
  for MFMA/alignment friendliness (`[[optimization/vectorization_and_coalescing.md]]`).
- **Persistent kernels**: a persistent grid (one workgroup/CU looping over tiles) lets you assign the
  tile→XCD mapping explicitly instead of trusting the dispatcher
  (`[[optimization/wave_and_grid_sizing.md]]`).

## Example mapping (conceptual)
For a tiled GEMM with `T` tiles on 8 XCDs, instead of `xcd = pid % 8` (scatters reuse), compute
`group = pid / tiles_per_xcd; xcd = group; local = pid % tiles_per_xcd` so a contiguous run of
data-sharing tiles stays on one XCD's L2. Tune `tiles_per_xcd` to the L2 working-set size.

## Pitfalls
- Default linear pid mapping ⇒ B/A panels re-fetched from HBM/foreign L2 instead of local L2 hits.
- Tile count not a multiple of 8 ⇒ one or more XCDs finish early (load imbalance, tail latency).
- <1024 workgroups on prefill ⇒ idle CUs/XCDs (`[[operators/dense_gemm/tuning.md]]`).
- Over-grouping (too many tiles pinned to one XCD) ⇒ L2 thrash; size groups to the L2 capacity.
- Assuming a unified L2 (it is **partitioned per XCD**).

## Verify
- Omniperf: L2 hit rate per channel / cross-XCD traffic, HBM read BW; XCD-aware order should *raise* L2
  hits and *lower* HBM reads at equal FLOPs (`[[profiling/]]`).
- A/B: linear vs swizzled pid mapping; compare HBM read volume and latency.
- Sanity: workgroup count ≥1024 and tile count % 8 == 0.
