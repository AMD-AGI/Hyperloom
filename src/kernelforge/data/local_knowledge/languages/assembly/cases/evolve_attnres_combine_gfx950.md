---
title: Evolve AttnRes combine on gfx950 - scheduling and live registers
kind: case
gens: [gfx950]
status: GPU-reproduced; upstream address defect; corrected candidate measured
updated: 2026-09-09
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

## Independent reproduction: address carries and BF16 rounding

Testing the pinned AITER source on MI355X, ROCm 7.2.3, PyTorch
2.11.0+gitd0c8b1f, and Triton 3.6.0 found two details that matter before
integration. The tests used Forge's assembler and a separate HIP harness
forwarding the actual stream; this is not a general production HIP adapter.

First, nine Phase B address updates modify only the low half of a 64-bit
pointer: the initial bank H offset, seven bank-row increments, and the prefix
H offset. They use `v_add_u32 v12, ...` without updating `v13`. Crossing a
4-GiB address boundary loses the carry and accesses an address 4 GiB too low.
Single-allocation warm tests can miss this. The original rotating-buffer
test faulted; do not treat its warm timing success as general address safety.

A local experimental correction replaced each affected add with a matching
`v_add_co_u32` / `v_addc_co_u32` pair, without increasing VGPR allocation.
Valid prefix and bank tensor views deliberately spanning a 4-GiB boundary
passed the oracle and nondefault-stream graph replay after the correction.
The original incorrect address arithmetic was checked on the CPU rather than
deliberately launching a known out-of-range access. A 64-buffer GPU test then
completed with the corrected candidate. The pinned upstream snapshot was
retained unchanged; this is an experimental correction, not an upstream fix.

Second, the final `v_lshrrev_b32 v2, 16, v1` truncates FP32 to BF16. It differs
from the Triton reference's nearest rounding. A separate control used
`v_cvt_pk_bf16_f32 v2, v1, v1`, storing the low BF16 result. On seed 42, combine
SNR against FP64 improved from about 47.83 dB to 57.59 dB, close to Triton;
the fraction differing from FP64 rounded to BF16 fell from about 57% to
0.0011%. This is not a claim of bitwise equality for all inputs.

The address-corrected, nearest-rounding candidate passed eight input cases
(multiple seeds, scales, zero vectors/weights, and padded strides) plus four
independent logit cases (uniform, peaked, negative, and large-offset finite
logits). Checks included input preservation and graph/pointer rebinding.
Fixed combine/pipeline bounds were `rtol=0.01, atol=1e-4` against FP64; these
experimental bounds do not replace a production driver's required semantics.

Independent graph timings exclude compilation, loading, allocation, clearing,
and the oracle. Single-buffer warm measurements used 11 interleaved rounds,
nine samples per round, and 100 calls per graph. Rotating measurements used
nine rounds, nine samples, and 64 distinct buffers totaling 589,299,712 bytes.
Standalone combine used fixed logits, separate from pipeline intermediates.

| Operation | Warm Triton / corrected ASM | Rotating Triton / corrected ASM |
| --- | --- | --- |
| Combine only | 3.624 / 3.306 us | 7.538 / 4.546 us |
| Triton score + combine, replacing only combine | 6.612 / 6.227 us | 8.266 / 8.107 us |

Combine improves by about 1.096x warm and 1.658x rotating, but the two-kernel
hybrid improves only 1.062x and 1.020x. Replacing score as well regresses the
pipeline; see [the score measurements](evolve_attnres_score_gfx950.md).
The pipeline reuses data from score, so its latency is not the sum of two
independently measured rotating kernels.

The author's original profiler benchmark faulted in this environment. With
only the address-corrected combine object substituted, that protocol completed
and reported 11.617 us Triton versus 5.513 us corrected combine. Its rotating
inputs, profiler, and unequal wrapper initialization differ from the matched
graph protocol above; do not mix their baselines or attribute the discrepancy
solely to synchronization. Neither protocol reproduced the reported 104.02-us
baseline. No model throughput or MoE speedup was measured, and no new Forge
agent search or statistical KEEP was claimed for these independent tests.
