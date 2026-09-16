---
title: "Kimi-K3 A16W4 MoE: ASM after FlyDSL optimization"
kind: case
scope: languages/assembly
updated: 2026-09-16
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Kimi-K3 A16W4 MoE: ASM after FlyDSL optimization

This case retains one measured E2E result and the numerical work needed to
interpret it. Both source kernels already included FlyDSL optimization.
Stage1's final repair was written during manual diagnosis and validated through
Forge; Stage2 came from the twelve-hour Forge search. This is not proof that a
fresh autonomous campaign will rediscover both final implementations.

## Reproduction boundary

The experiment used eight MI355X GPUs, ROCm 7.2, FlyDSL 0.2.4 and SGLang 0.5.17.
AITER's `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py` supplied
`compile_mixed_moe_gemm1_a16w4` and `compile_mixed_moe_gemm2_a16w4`.
The workload had 896 experts, topk 16, model dimension 3584 and 384 intermediate
columns padded to 512. Stage1 used a 32x64x256 tile, BF16 activations, MXFP4
weights and BF16 SiTUv2 output without routing-weight multiplication.
Stage2 down-projected `(T, 16, 512)` and accumulated into BF16 `(T, 3584)` with
`global_atomic_pk_add_bf16`. Frontend, launcher and independent oracle stayed fixed.

## Stage1: repair LDS reuse before claiming a gain

The search winner passed the original small 30 dB suite, but expanded inputs
found 29.57 dB against the oracle and about 27 dB between repeated outputs.
Its earlier combined +3.32% throughput result remains rejected.

Before `buffer_load_dwordx4 ... lds` overwrites a reused tile, every consuming
wave must finish its old LDS reads. `s_waitcnt lgkmcnt(0)` drains the issuing
wave; it does not synchronize other waves. A barrier after the new writes begin
cannot repair that race. The repair inserted the wait followed by `s_barrier`
before each of twelve reuse groups, leaving the two initial fills unchanged.
It also restored combined VMEM drains and the correct 41984-byte LDS declarations;
the winner's 32768-byte descriptor did not match its retained address layout.
Other cache-hint and scheduling edits remained. Analyze each specialization's
buffer lifetime rather than copying fence locations or resource sizes.

The repaired Stage1 passed 32 case/mode combinations with 32 repetitions per
source-before/candidate/source-after role: eight inputs at 24/64 tokens across
eager, dirty-output, direct-call and graph modes. The relative error multiplier
was 1.0, the absolute RMS floor `1e-6`, and the mathematical floor remained 30 dB.
Worst oracle SNR was 88.78 dB; measured repeat error was zero. Inputs, weights and
output padding were checked. This establishes only the tested scope.

## Stage2: retain the distinction between bounded precision and repeatability

The original Stage2 winner changed addressing, load scheduling and the epilogue
while retaining BF16 atomic accumulation. Its initial relative-error gate failed;
unchanged compiler assembly also failed. Serializing atomic issues or restoring
address order did not pass that contract. Those rejections remain recorded.

A separate source-only calibration froze a new task precision envelope before
candidate holdouts. It used the calibration maximum plus three across-seed
standard deviations of per-seed maxima, with absolute floors around
0.005724-0.005920 in reference-normalized L2 error. The relative multiplier stayed
1.0 and the mathematical floor stayed 30 dB. This empirical allowance is not a
confidence bound or a recommended default; no Forge default was relaxed.

Source control and candidate passed 96 fresh shape/seed/mode combinations with
32 repetitions per role. Candidate worst oracle SNR was 44.93 dB, but mean repeat
RMS was approximately 9.7% greater than source. Isolating one expert slot at a time
gave bitwise agreement in 192 tested combinations; it did not make the full atomic
reduction deterministic. The extra variability was not repaired.

## E2E and quality evidence

Each leg used two throughput measurements of 192 requests, ISL 8192 / OSL 1024,
concurrency 64, warmup 8 and seed 42. Binding audits passed on all eight workers.
Source-before and source-after bracketed each candidate. These are separate
experiments; do not subtract their gains to claim an isolated Stage2 contribution.

| Experiment | Pooled optimized FlyDSL, tok/s | FlyDSL + ASM, tok/s | Gain |
| --- | ---: | ---: | ---: |
| Repaired Stage1 only | 477.98 | 486.88 | +1.86% |
| Repaired Stage1 plus original Stage2 winner | 477.45 | 488.90 | +2.40% |

Source mean drift was 0.018% in the Stage1 experiment and -0.264% in the combined
experiment. Against the faster combined source control, the gain was +2.26%.
The original-MoE baseline of 452.86 tok/s came from the preceding experiment;
it is not a newly measured baseline for the combined trial.

Each leg also answered 256 GSM8K and 256 MMLU questions. Stage1 scores stayed
within observed source-repeat ranges. The combined E2E leg scored 248/232 versus
247/234 before and 248/233 after: retain that initial MMLU drop. Four fixed-count
quality-only replicas (source/combined/combined/source) scored 246/233, 248/234,
248/234 and 246/234. They did not reproduce a persistent loss, but omitted the
throughput prelude and reused the same 512 questions. They do not prove model
quality equivalence or erase Stage2's greater repeat variability.

## Exact artifacts

Evidence root: `/shared_nfs/chenyi/forge-kimi-k3-asm-12h-20260914/`.
Use `stage1-diagnosis/repaired-canonical/` and `stage1-repair-e2e/` for Stage1;
`combined-repair-e2e/` retains rejected Stage2 controls;
`stage2-calibrated-v2-20260915/` contains the frozen contract and holdouts;
`combined-validated-e2e/` and `combined-quality-replications/` hold model evidence.

| Artifact | SHA-256 |
| --- | --- |
| Protected FlyDSL source | `3e58a5035ae050cd8d361aaa514053b34060490c231e1b932644dfd5be0d6ae7` |
| Repaired Stage1 assembly | `7cee6ec838bd3707f901172ec73050c63deed4493983cf35138ae5aef6225820` |
| Stage2 assembly | `102314728bfbe53c09b407de9a6a0430ee47122747c34147e0a606edc743dafb` |

Verify these artifacts and their frozen drivers/contracts before replay.
Use the current [campaign workflow](assembly_workflow.md) for a new search;
the historical experiment does not waive its preparation or acceptance checks.
