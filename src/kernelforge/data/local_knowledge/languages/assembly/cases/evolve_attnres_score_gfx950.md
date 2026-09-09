---
title: Evolve AttnRes score on gfx950 - interleaved reductions
kind: case
gens: [gfx950]
status: GPU-reproduced; slower than the tested Triton baseline
updated: 2026-09-09
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Evolve AttnRes score: interleave independent reductions

Use this case when a kernel computes two reductions over the same input and
the generated ISA serializes their dependency chains. It provides a concrete
gfx950 schedule to investigate, not a general replacement for reductions.

## Provenance and scope

Neha Prakriya's [SGLang PR #33735](https://github.com/sgl-project/sglang/pull/33735)
publishes the handwritten kernel. AITER's
[launcher at `1ddd136`](https://github.com/ROCm/aiter/blob/1ddd136b0cf8ee9cd67afbb9b1e69202dc96ff68/aiter/ops/flydsl/kernels/kimi_k3_attnres.py)
attributes the score and combine kernels to Evolve-Kernel island search and
reports score latency of **34.78 us -> 5.53 us** on MI355X. This is the author's
whole-kernel comparison with Triton, not a Forge reproduction or an ablation
isolating the DPP schedule's contribution.

## Independent reproduction: do not assume the reported baseline

On MI355X/gfx950 with ROCm 7.2.3, PyTorch 2.11.0+gitd0c8b1f, and Triton
3.6.0, the unmodified score source rebuilt through Forge's assembler passed
an independent FP64 oracle. The reference was the unmodified Triton kernel
in AITER's pinned test, with its original launch configuration. Both sides
used the same stream and GPU graph timing, excluding compilation, output
initialization, Python launch overhead, and the oracle.

Warm medians were **2.892 us Triton versus 5.302 us assembly**; a fresh process
with an empty compiler cache reproduced **2.890 versus 5.292 us**. With 64
distinct input sets totaling about 589 MB, medians were **4.539 versus 10.234
us**. These protocols do not reproduce the author's 34.78-us Triton baseline.
The assembly replacement is slower in this environment; do not keep it merely
because the source reports a large speedup. This does not establish which
compiler, launch, or measurement difference explains the author's result.

Checks covered multiple seeds, unit-scale and near-zero inputs, zero vectors,
zero weights, padded outer strides with contiguous H, input preservation,
changed pointers, determinism, and nondefault-stream graph replay. Score was
compared against FP64 with fixed `rtol=3e-4, atol=3e-4`; zero-reference cases
were checked elementwise. A separate assembly edit replacing the final score
with zero was rejected, and restoring the original module restored correctness.
These are fixed-shape kernel checks, not a model-quality certification.

The source specializes Kimi-K3 decode: `T=64`, `NVB=8`, `H=7168`,
`BLOCK_H=1024`, `MAX_ROWS=16`, wave64, gfx950. Inputs are BF16 prefix
`[T,H]` and bank `[T,NVB,H]`, plus FP32 weights `cw[H]`; scores are FP32
`[T,MAX_ROWS]`. For each token and each of eight bank rows plus the prefix,
it computes `dot(v, cw) * rsqrt(sum(v*v) / H + 1e-6)`.

## What the source actually does

Inspect the
[complete score assembly at `fbbf9e6`](https://github.com/sgl-project/sglang/blob/fbbf9e6e16d7f5295512ae3159555a3467afc348/python/sglang/srt/layers/kimik3_attnres_score.s).

| Stage | Source observation | Optimization hypothesis |
| --- | --- | --- |
| H loop | Seven chunks of 1024 elements; each work item loads BF16 `v` and FP32 `cw`, then updates sumsq and dot accumulators. | Share input loads between the two reductions. |
| Wave reduction | Alternate the sumsq and dot DPP chains using `row_shr:1/2/4/8` and `row_bcast:15/31`. | Place independent arithmetic between dependent reduction instructions. |
| Workgroup reduction | Lane 63 in each of 16 waves writes two partials into separate LDS regions; restore `exec`, wait for LDS, then barrier. | Communicate wave partials rather than every lane's values. |
| Finalization | Work item 0 reads four `ds_read_b128` groups per accumulator and computes the normalization and score. | Use contiguous LDS reads for the final small reduction. |

The descriptor declares 16 VGPRs and 128 bytes of static LDS. These are source
resource declarations, not measured occupancy. `v0` supplies the work-item ID;
the code derives wave and lane IDs separately. Replacing it with a lane-only
ID would alias different waves' LDS slots.

The header includes claimed `s_nop` requirements around DPP and reciprocal /
reciprocal-square-root instructions. Treat these as source commentary, not
universal latency rules. Check the target ISA's dependency hazards and both
operands of each consumer before removing waits or changing scheduling.

## Experiment in Forge

1. Establish the original kernel's correctness and timing with the protected
   driver. For an existing supported FlyDSL callable, first pass the unedited
   assembly roundtrip in [the workflow](../INDEX.md). This standalone Triton
   replacement instead needs the [HIP module integration checks](../guides/hip_module_validation.md).
2. Inspect whether the current ISA has independent reductions and serializes
   them. Try interleaving those chains as one candidate. Evaluate changes to
   the LDS reduction separately so the winning mechanism is attributable.
3. Preserve DPP masks, lane ownership, LDS offsets, active-lane restoration,
   and workgroup synchronization. Recalculate descriptors when registers or
   LDS change; check spills and occupancy with the actual toolchain.
4. Check reduction-order error against the unchanged oracle across the
   driver's inputs. The upstream test's score SNR threshold is case-specific;
   it must not replace Forge's tolerances. Score writes only the nine active
   columns, so preserve the wrapper's contract for the remaining columns.
5. Keep only a candidate that passes correctness and the campaign's measured
   improvement gate. Revert if additional live registers or finalization
   work removes the benefit. If the reduction topology or tile shape must
   change, do that in the high-level source and regenerate assembly.

There is no demonstrated MoE GEMM or cross-generation speedup in this case.
Porting it requires a new architecture/shape contract and fresh measurements.
