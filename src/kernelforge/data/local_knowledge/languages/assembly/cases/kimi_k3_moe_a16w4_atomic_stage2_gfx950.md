---
title: Kimi-K3 A16W4 stage2 - validate a BF16 atomic reduction
kind: case
gens: [gfx950]
status: bounded-precision gate passed; combined E2E and limited quality replicas completed
updated: 2026-09-15
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Separate reduction order from an incorrect expert contribution

AITER's `compile_mixed_moe_gemm2_a16w4` down-projects Kimi-K3's expert outputs
and accumulates sixteen contributions into a BF16 output using
`global_atomic_pk_add_bf16`. The tested specialization has activations
`(T, 16, 512)`, 384 valid intermediate columns, 896 experts and output
`(T, 3584)`. Measurements used FlyDSL 0.2.4, ROCm 7.2 and MI355X.
Its Forge source-IR fingerprint is
`2ab4f9121708ac07120754ca9dc2bda7926d00b18105c5a41d71dd4ba266fb78`.

The source already included FlyDSL optimization. A twelve-hour Forge assembly
campaign retained candidate commit `a46cfe9d74651a28529d7db87eb7743a8abbe8d5`;
its assembly SHA-256 is
`102314728bfbe53c09b407de9a6a0430ee47122747c34147e0a606edc743dafb`.
The candidate retains the BF16 atomic reduction while changing address
arithmetic, load scheduling and the epilogue. It does not provide a deterministic
or FP32 reduction. This is different from Stage1's
[LDS reuse race and repair](kimi_k3_moe_a16w4_lds_reuse_gfx950.md).

## Preserve the rejected contract and diagnose its controls

The initial expanded gate required candidate oracle and repeat errors to be no
larger than the sampled source-before/source-after maxima, with multiplier 1.0
and floor `1e-6`. The candidate failed, despite mathematical SNR above 44.95 dB.
Untouched compiler assembly also failed the same comparison. Restoring atomic
address order and serializing atomic issues did not pass that contract either.
Those verdicts remain failures; later validation does not rewrite them.

Finite sample maxima are not a sound equality test for a nondeterministic
control. Conversely, a failing control does not establish that a candidate has
no regression. Faster issue order can change the interleaving and rounding of
expert contributions even when each individual contribution is identical.
Always zero the output on each launch, including direct calls and inside a
captured graph, before investigating such differences.

## A separately frozen, bounded-precision experiment

This experiment used the existing task-owned numerical contract; it did not
change Forge's gate implementation or defaults. It explicitly replaced the
earlier tiny absolute floor with a source-calibrated precision envelope. It
establishes bounded numerical error, not unchanged repeat-output variance.

Source-only calibration used sixteen routing seeds at each of 24 and 64 tokens,
three execution modes (eager, direct and graph replay), and two blocks of 32
outputs per case/mode. No ASM candidate was loaded during calibration. An
initial envelope equal to the calibration maximum still rejected untouched
compiler assembly on an unseen input by approximately `9.8e-7` RMS. This failed
control and its policy were retained, before candidate holdout evaluation.

A new policy was then frozen: for each shape/mode, use the maximum calibration
error plus three times the across-seed standard deviation of per-seed maxima.
Each maximum includes source oracle and repeat errors. The resulting absolute
floors were approximately `0.005724` to `0.005920` in reference-normalized L2
RMS. The mathematical floor remained 30 dB and the relative multiplier remained
1.0. This allowance is empirical and specific to the tested atomic reduction;
it is neither a distribution-free confidence bound nor a recommended universal
Forge tolerance. Do not copy it to a deterministic kernel or derive it from a
candidate's failures.

Both unchanged compiler assembly and the original ASM candidate passed all
96 declared shape/seed/mode combinations on sixteen new routing seeds per
shape, with 32 outputs per source-before/candidate/source-after role. Protected
inputs, weights and source hashes were checked. The contract and candidate
hashes were frozen before these holdouts ran.

| Metric in the candidate holdout run | Native source before | ASM | Native source after |
| --- | ---: | ---: | ---: |
| Worst oracle SNR, dB | 44.9193 | 44.9282 | 44.9163 |
| Mean normalized oracle RMS error | 0.00552238 | 0.00552283 | 0.00552237 |
| Mean normalized repeat RMS error | 0.00351410 | 0.00385549 | 0.00351539 |

The approximately 9.7% increase in mean repeat RMS is real in this sample and
must not be described as unchanged repeatability. Individual case maxima
reached 1.409 times the sampled source maximum; the untouched compiler control
also reached 1.261 times its own sampled source maximum. The candidate's oracle
error stayed close to source. Model-level quality must still be tested.

## Isolate the contributions without changing the launcher

A separate diagnostic retained the production specialization and routing,
zeroed activation slots for fifteen experts and enabled one expert slot at a
time. It exercised all sixteen slots, two new seeds at each token count, and
all three execution modes. Four independently owned output snapshots were
taken for source-before, candidate and source-after.

All **192 combinations / 2304 measured launches** were finite, deterministic
and bitwise equal across implementations. This supports unchanged individual
contributions for those inputs; the complete atomic reduction still rounds in
an order that can vary. It is not proof covering every routing or tensor value.

## Provenance and model-validation boundary

Evidence is under
`/shared_nfs/chenyi/forge-kimi-k3-asm-12h-20260914/`:

- `combined-repair-e2e/`: original failed gate, compiler control and ablations.
- `stage2-calibrated-20260915/`: source-only measurements and the rejected
  maximum-only calibration control.
- `stage2-calibrated-v2-20260915/`: frozen protocol, source measurements,
  protected drivers/contracts, control/candidate verdicts and expert isolation.
- `combined-validated-e2e/`: fresh source/combined/source model experiment.

## Completed combined E2E experiment

The combined model experiment used the repaired Stage1 and this original Stage2
winner on eight MI355X GPUs with SGLang 0.5.17. Each fresh leg ran two serving
measurements, each with 192 requests, 196608 output tokens, ISL 8192 / OSL 1024,
concurrency 64, warmup 8 and seed 42. Each then answered the same 256 GSM8K and
256 MMLU questions. All requests completed; all answers were valid and
untruncated. Binding audits confirmed both selected specializations on all eight
workers. Runtime assembly/gate implementation matched the PR checkout after
normalizing Windows/Linux line endings.

| Version | Mean output tok/s | GSM8K / 256 | MMLU / 256 |
| --- | ---: | ---: | ---: |
| Original MoE, reused from preceding experiment | 452.8584 | 248 | 233 |
| Optimized FlyDSL, before | 478.0836 | 247 | 234 |
| Repaired Stage1 plus Stage2 ASM | 488.9041 | 248 | 232 |
| Optimized FlyDSL, after | 476.8234 | 248 | 233 |

The pooled source mean was 477.4535 tok/s. ASM improved throughput by **2.3983%**
against that mean, **2.2633%** against the faster source-before mean, and
**2.5336%** against source-after. Source mean drift was -0.2636%; retain the
individual source measurements (478.0889, 478.0784, 477.8715, 475.7753) rather
than discarding the slower final sample. Original-MoE to combined-ASM gain was
7.9596%, with the original baseline explicitly reused. Do not carry forward the
earlier invalid combined Stage1/Stage2 gain or subtract the separate Stage1-only
experiment to claim an isolated Stage2 contribution.

**This throughput result is not a model-quality acceptance.** The combined leg
lost one MMLU answer versus both original MoE and source-after, and two versus
source-before. `mmlu-11715` already varied in source controls; `mmlu-9183` had
been correct in the available earlier controls and Stage1-only test but was
wrong in the combined leg. Paired 95% bootstrap intervals do not establish
non-inferiority: the MMLU difference versus source-after was -0.3906 percentage
points with interval [-1.1719, 0]. Failure to demonstrate a significant decline
would not prove equal model quality.

A fixed four-leg full-512-question replication (source, combined, combined,
source, independently launched) was prepared before its outputs, under
`combined-quality-replications/`. These diagnostic replicas omit the throughput
prelude and make no additional performance claim. Retain every result; do not
rerun a failing question or select one favorable model run as acceptance.

All four replicas completed, with valid, untruncated answers and eight-worker
binding audits:

| Replica order | GSM8K / 256 | MMLU / 256 |
| --- | ---: | ---: |
| Source | 246 | 233 |
| Combined ASM | 248 | 234 |
| Combined ASM | 248 | 234 |
| Source | 246 | 234 |

`mmlu-9183` was correct in all four replicas. No question was correct in both
source replicas and wrong in both combined replicas. Resampling questions after
averaging each variant's two repetitions gave combined-minus-source differences
of +0.7813 percentage points for GSM8K (95% interval [0, 1.9531]) and +0.1953
for MMLU ([0, 0.5859]). Repeated answers to the same question are not independent
new test questions.

These fixed-count replicas did not reproduce a persistent quality loss. They
do not erase the original E2E leg's MMLU miss, establish equality of output
repeatability, or certify all workloads. The complete serving experiment and
these quality-only replicas have different preludes and are reported separately.
Both kernels have numerical acceptance within the declared scopes, and the
combined speedup is measured; broader quality claims need broader evaluation.
