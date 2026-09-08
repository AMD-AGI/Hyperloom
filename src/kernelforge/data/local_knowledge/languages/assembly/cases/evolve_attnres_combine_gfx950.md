---
title: Evolve AttnRes combine on gfx950 - scheduling and live registers
kind: case
gens: [gfx950]
status: source-inspected; performance author-reported
updated: 2026-09-08
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Evolve AttnRes combine: schedule without clobbering live values

Use this case for a small softmax followed by a weighted sum, or when moving
loads across a barrier appears promising. Its most useful failed hypothesis
is that mathematically independent work can still conflict through registers.

## Provenance and scope

Neha Prakriya's [SGLang PR #33746](https://github.com/sgl-project/sglang/pull/33746)
publishes the
[combine assembly at `1a505cb`](https://github.com/sgl-project/sglang/blob/1a505cb04072f2de48349a86e9603860eedb6510/python/sglang/srt/layers/kimik3_attnres_combine.s).
AITER's
[launcher at `1ddd136`](https://github.com/ROCm/aiter/blob/1ddd136b0cf8ee9cd67afbb9b1e69202dc96ff68/aiter/ops/flydsl/kernels/kimi_k3_attnres.py)
attributes the kernels to Evolve-Kernel and reports **104.02 us -> 3.32 us**
for combine on MI355X. These are author-reported whole-kernel timings against
Triton. They do not isolate the effect of the final instruction schedule;
the assembly header also contains historical variants and different baselines.

This is gfx950, wave64, `T=64`, `NVB=8`, `H=7168`, `BLOCK_H=1024`,
`MAX_ROWS=16`. It normalizes nine FP32 scores and combines eight BF16 bank
vectors plus the BF16 prefix into BF16 output `[T,H]`. The launcher uses one
1024-thread workgroup per token and H tile, with seven H tiles.

## Final instructions versus the proposed overlap

The opening header proposes issuing BF16 loads before softmax and its
barrier, then explains why that proposal conflicts with register allocation.
Read the final instructions before adopting the header's initial plan:

1. Work item 0 loads all nine FP32 scores, subtracts their maximum, and scales
   by `log2(e)` for `v_exp_f32`.
2. The code interleaves independent exponentials with additions that build
   their sum, retaining explicit NOPs, then normalizes the weights.
3. It writes the weights to LDS, restores `exec`, executes
   `s_waitcnt lgkmcnt(0)`, and reaches `s_barrier`.
4. **After the barrier**, all work items issue the nine BF16 loads. They read
   the LDS weights and consume the loads with `vmcnt(8)` down to `vmcnt(0)`
   waits and FMAs.

Thus the final source demonstrates interleaved softmax arithmetic and
post-barrier load batching. It does not demonstrate successful BF16
load-before-barrier overlap. The descriptor declares 14 VGPRs, 256 bytes of
static LDS, and a 56-byte kernarg segment.

## Failed hypothesis: early loads with the same VGPRs

The proposed early BF16 loads would keep values in `v2..v10` while softmax
needs those registers for FP32 scores. Restricting `exec` to work item 0 does
not create a separate register bank: that work item's BF16 values and FP32
scores would occupy the same registers. The header abandons this overlap
unless allocation or kernel structure changes.

Before a similar transformation, write down each live value's producing
instruction, last consumer, register, and active lanes. Include pending load
destinations. A valid redesign must separate overlapping lifetimes, then
measure the cost of additional VGPRs, spills, or changed occupancy. A barrier
does not repair a clobbered register or replace the required memory waits.

## Experiment in Forge

1. Start from the current kernel's verified assembly baseline using
   [the workflow](../INDEX.md) and, for a standalone module,
   [the integration guide](../guides/hip_module_validation.md).
2. If profiling points to dependent special-function arithmetic, try
   interleaving independent exponentials while preserving numerical semantics.
   Check the ISA hazards for every operand; the source's comments about
   instruction latency are not a proof that a new schedule is valid.
3. Evaluate earlier memory issue as a separate experiment only after the
   register-lifetime analysis. Recompute wait counters from the actual pending
   operations and target completion rules whenever loads are added, removed,
   or reordered; `vmcnt(8)` is not a portable constant for this algorithm.
4. Test the score-to-output pipeline and combine in isolation against an
   independent reference. Preserve the driver's tolerances and include
   supported inputs with near-uniform and strongly peaked finite logits.
   Upstream's combine SNR threshold is a case observation, not a new gate.
5. Reject incorrect or timing-regressing variants, even if they have fewer
   NOPs or earlier loads. Return structural changes such as tiling or a new
   softmax decomposition to the high-level implementation and reestablish
   assembly parity before further instruction edits.

Forge has not independently reproduced this case's reported speedup. Do not
attribute model throughput changes or MoE improvements to this schedule.
