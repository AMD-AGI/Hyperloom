---
title: weight streaming — keeping a decode launch's weight loads in flight
kind: lever
lever: monokernel_weight_streaming
gens: [gfx950]
bottleneck: bandwidth-bound GEMV stages
updated: 2026-11-09
sources:
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/ops.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/minimax_m31/ffn.py
  - https://github.com/sgl-project/sgl-amd-mono/pull/59
---

# Weight streaming

## Route here when
A decode launch's GEMV units (router, expert up / down, projections) take longer than their weight
bytes over HBM bandwidth, and the per-CTA timeline shows a unit waiting on its own loads.

## What to change
1. **Count the traffic before the weights.** In MiniMax-M3.1's launch each lane loaded its own column's
   B operand from global memory; at 8 rows a slot has one or two columns, so the repeats were most of a
   CTA's vector memory traffic — 24 KB of activations a wave a unit against 12 KB of weights. Weights
   served from the MALL timed the same as from HBM: the weights were not the limit (#59).
2. **Shared operands from LDS.** Load the activations into LDS once a CTA and read B there; copy only a
   slot's own columns (16 bytes a lane) for the down stage (#59). Mind the pool: MiniMax-M3.1 fits it up
   to 8 rows; 16 rows do not fit with the rest.
3. **Issue the next unit's weights during the current one.** A routed unit issues its next unit's
   weights after its activation reads; past the CTA's last unit they read through a zero-sized buffer,
   so the loop keeps one shape (#59). Before this, an up unit waited on its own weights for about 3 µs
   of its 5.6.
4. **Nontemporal loads for weights a layer reads once**: `ops.CM_NT = 2`, so they do not displace the
   hand-off lines and activations the launch reuses.

## Measure it honestly
Rotate the weights through several copies so they stream from HBM as a server's layers do; #59 used
four, with one copy still resident in the 256 MB MALL. Timing one copy in a loop measures the MALL, and
weight streaming then looks free.

## Verify
- Bitwise equal to `main` (#59: every output identical at 1, 3, 8, 9 and 16 rows).
- VGPR and LDS budgets at every width (`skills/profile/compile_check.md`): prefetching the next unit
  holds its registers across the current one.

## Expected magnitude
#59 (with o_proj in the launch and routing on the published `rstd`): 54.7 → 44.8 µs (−18%) at 8 distinct
experts, −13% at 14, −12% at 18; c1 decode +7.4%.

## Failure modes
| Symptom | Cause |
|---|---|
| prefetch adds nothing | each slot's acquire waits on every load in flight, so a prefetch issued before it delays it (#59 tried and dropped prefetching the next down slot) |
| faster in the microbenchmark, flat in serving | the benchmark read weights from the MALL |
| spills after adding a prefetch | the prefetched unit's registers live across the current unit |
