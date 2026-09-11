---
title: Evolve AttnRes score on gfx950 - interleaved reductions
kind: case
gens: [gfx950]
status: GPU-reproduced; Forge improves the ASM seed but not the source Triton
updated: 2026-09-11
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

## Production baseline and runtime epsilon: September 11 remeasurement

On MI355X with PyTorch 2.11 / ROCm 7.2 / Triton 3.7, a matched experiment
compared the pinned ASM with both the AITER test reference and SGLang's
production `_score_kernel`. These Triton implementations have different
reduction structures: the production kernel reduces each H chunk before
accumulating scalars (`num_warps=8`); the test reference accumulates vectors
across chunks and reduces once (default four warps). Do not treat their
latencies as interchangeable.

At `T=64, NVB=8, H=7168, eps=1e-6`, seven interleaved rounds, five samples per
round, and 100 calls per graph measured warm medians of **5.880 us production
Triton, 2.828 us test-reference Triton, and 5.272 us Neha ASM**. The ASM reduces
latency by about 10% relative to this production score, but remains slower
than the test reference. This still does not reproduce the author's 34.78 us.

Check scalar arguments against instructions, not only metadata. The published
score declares `eps` at kernarg offset 36 and loads it into `s13`, but uses
`v_mov_b32 v4, 0x358637BD` (hardcoded `1e-6`) in the final normalization.
Passing `1e-5`, the local Kimi-K3 model's RMSNorm epsilon, failed the fixed
FP64 score tolerance. A separate correctness repair replaces that move with
`v_mov_b32 v4, s13`; tests at `eps=1e-5` passed for T=1,4,16,64,256. The
minimal example intentionally keeps its documented `1e-6` specialization.
The repair is not a scheduling speedup and does not authorize silently
changing a verified launcher's input domain.

SGLang 0.5.19's normal ROCm path for H=7168/NVB<=8 uses a single fused Triton
kernel, bypassing the two-kernel `_mix_fused` branch where the published score
replacement is installed. At T=64, mixture-only timing was **7.290 us fused
Triton versus 8.578 us for Neha score plus repaired ASM combine**. The real
fused path can also perform residual add, bank snapshot and output RMSNorm.
Prove dispatch and compare the complete caller before making an E2E claim.

## Verified Forge instruction-only campaign

The minimal `examples/triton2asm-attnres/` was run with Forge revision
`48bd5d730` (the `191d4c7f0` snapshot plus its GPU-verified example-driver fix),
`--kernel-backend assembly`, one lane, and Codex `gpt-5.6-sol`. PORT copied
the attributed seed and standalone launcher, passed correctness and the
deliberate assembler-failure probe, then froze the launcher and driver.
Two instruction-only iterations passed the unchanged four-case FP64 oracle,
stream/graph checks, canonical suite and three independent measurements:

1. After restricting `exec` to work item 0 for finalization, branch around
   the scalar/LDS tail when `exec` is empty. Other waves still restore their
   saved execution masks and retain the preceding workgroup barrier.
2. Replace repeated full per-lane 64-bit address reconstruction in the
   seven-chunk loop with `global_load_ushort v3, v5, s[20:21]` and
   `global_load_dword v4, v6, s[8:9]`. Compute lane offsets once and advance
   them by 2048/4096 bytes per chunk. The scalar base retains the full
   address; this is not the combine kernel's incorrect low-half pointer add.

The campaign measured approximately **3.1 us source Triton, 5.6 us initial
ASM, and 4.3 us optimized ASM**. Its per-case score gives **1.314x incremental
speedup over the ASM seed**, but **0.724x relative to the original source**.
Both iterations were KEEP relative to the incumbent; the final result
correctly records `incremental_improved=true` and `improved=false`.
Do not relabel these KEEP decisions as wins over Triton or the model.

Campaign `07ad340a` retained best workspace commit
`68fad1d45b97893ca354f9ec7792c40fa872b1ab`; the launcher hash remained unchanged.
Artifacts, the exact ASM-only patch, and raw measurements were preserved under
`/shared_nfs/chenyi/forge-neha-repro-20260911/forge-score-campaign-ca/` and
`recovery-134141/`. A clean candidate replay on a second MI355X node passed
at SNR 137.377 dB and measured 4.041 us. Its source and seed were not remeasured
there; do not combine those cross-node numbers into another speedup ratio.
No model E2E improvement has been established by this campaign.

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
