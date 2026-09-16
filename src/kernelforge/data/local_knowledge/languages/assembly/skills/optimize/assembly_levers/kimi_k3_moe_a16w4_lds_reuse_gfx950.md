---
title: Kimi-K3 A16W4 stage1 - LDS reuse race and numerical repair
kind: case
gens: [gfx950]
status: repaired Stage1 passed canonical acceptance; 1.86% E2E gain on one workload
updated: 2026-09-15
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Audit both sides of an LDS buffer's lifetime

Use this case when repeated outputs change after editing waits, prefetches or
cache hints in a double-buffered GEMM. It concerns BF16 activations, MXFP4 weights
and BF16 SiTUv2 output on FlyDSL 0.2.4 / ROCm 7.2 / MI355X. The separate
[A8W4 case](kimi_k3_moe_a8w4_gfx950.md) has different activation and output formats.

## Identify the exact specialization

AITER's `compile_mixed_moe_gemm1_a16w4` in
`aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py` produced `moe_gemm1_0`.
The captured workload has 896 experts, topk 16, model dimension 3584,
384 valid intermediate columns padded to 512, and tile 32x64x256. Stage1 uses
separate gate/up weights and does not multiply routing weights into its output.
The protected source file SHA-256 was
`3e58a5035ae050cd8d361aaa514053b34060490c231e1b932644dfd5be0d6ae7`;
the Forge source-IR fingerprint was
`418fd256eb0442b546772f7791ada33f5ca674d4e8bdf1de2fd6ab572998ff00`.

The source already included FlyDSL optimization. The 12-hour ASM campaign kept
source, launcher and driver fixed while editing its selected `.s`. Its final
candidate passed the original smaller 30 dB suite, but expanded tests found
29.57 dB against the mathematical reference and approximately 27 dB between
repeated outputs. Its measured combined Stage1/Stage2 E2E gain was rejected.
A successful short test is not evidence that a later failing input is acceptable.

## Separate a wave's completion from a workgroup's completion

The compiled kernel reads one LDS tile while prefetching another with
`buffer_load_dwordx4 ... lds`. Before reusing a tile's storage, every consuming
wave must have finished its previous LDS reads. `s_waitcnt lgkmcnt(0)` drains
the issuing wave's tracked operations; it does not make other waves finish.
A barrier after the overwrite cannot repair data already read from a new tile.

In the compiler seed, the first reuse group starts around line 410, shortly
after reads such as `ds_read_b128 ... offset:25600`. That group's `m0` selects
the same LDS region. The existing barrier occurs after the new DMA writes have
already been issued. Waves progressing at different rates can therefore race
between reading the old tile and overwriting it with the next one.

The repair adds `s_waitcnt lgkmcnt(0)` followed by `s_barrier` immediately before
each of the twelve DMA groups that reuse an LDS buffer. The two initial fills
do not reuse a live tile and did not need these additional fences. This placement
was checked against the actual fully unrolled specialization, not inferred from
instruction counts alone. A different schedule needs its own lifetime analysis;
do not insert barriers inside a divergent region.

The repaired winner also restores the compiler's combined VMEM drains and both
41984-byte LDS declarations. Its previous 32768-byte declaration did not match
the retained address layout. Decreasing an allocation descriptor does not move
the memory accesses. Other winner instructions, including weight-load cache
hints and prefetch/register scheduling, remain in the repaired `.s`.

## Use controls to distinguish a repair from a timing accident

Native source and untouched rebuilt assembly had identical `.text`, `.rodata`
and `.note` sections, yet both showed smaller output variability. In the new
diagnostics, all weights, scales, activations and routing/sorting buffers stayed
unchanged. The issue was observable through the direct compiled callable as well
as graph replay; it was not resolved by avoiding the Python wrapper.

| Diagnostic change to compiler seed | Observed result |
| --- | --- |
| Drain VMEM after each load | Error worsened; the wave timing changed without fixing cross-wave buffer reuse. |
| Delay every MFMA | Variability decreased but remained. Timing changes can hide a race without removing it. |
| Add DMA completion barriers after each group | Variability remained. The overwrite had already started. |
| Drain LDS reads and synchronize before each reused group | Repeated outputs became identical in the tested inputs and execution modes. |

The repaired winner used Forge's canonical numerical gate at revision
`8844cb83045c9e3eeed12c1fd81fd3887d1fa3ff`. Eight input cases covered six routing
seeds at 24 tokens and two at 64 tokens, each with eager, dirty-output,
direct-call and graph-replay execution. Each source-before/candidate/source-after
measurement repeated 32 times. The 32 declared case/mode combinations all passed
with source-relative error multiplier **1.0**, absolute RMS floor `1e-6`, and
the unchanged 30 dB mathematical floor. Candidate worst mathematical SNR was
**88.78 dB** and measured repeat error was zero. Input and weight preservation
and zero output padding were checked independently.

Three preliminary paired microbenchmarks measured approximately 1.101x, 1.101x
and 1.106x against optimized FlyDSL. The old invalid combined result must not be
attributed to this repair or carried forward as its E2E result.

## Fresh Stage1-only model measurement

An eight-MI355X SGLang 0.5.17 run compared original MoE, optimized FlyDSL,
optimized FlyDSL with repaired Stage1 ASM, and optimized FlyDSL again. Stage2
remained FlyDSL in the ASM leg. Only the two MoE source patches were reverted
in the original-MoE leg; shared attention/KDA changes stayed fixed.

Each leg ran two throughput measurements, each with 192 requests, 196608 output
tokens, ISL 8192 / OSL 1024, concurrency 64, warmup 8 and seed 42. Per-rank
dispatch audits passed for all eight workers. Measurements were:

| Version | Mean output tok/s | GSM8K / 256 | MMLU / 256 |
| --- | ---: | ---: | ---: |
| Original MoE | 452.8584 | 248 | 233 |
| Optimized FlyDSL, before | 477.9388 | 247 | 234 |
| Repaired Stage1 ASM | 486.8766 | 247 | 233 |
| Optimized FlyDSL, after | 478.0268 | 248 | 233 |

The ASM gain was **1.8607%** against the pooled FlyDSL source measurements and
**1.8513%** against the final source measurement. Source drift was **0.0184%**.
FlyDSL plus the Stage1 repair was **7.5119%** above original MoE in this setup.
All fixed-input model answers were valid and untruncated. Source itself moved
by one answer on each task; candidate totals fell within those observed ranges.
These 512 questions do not establish unchanged model quality on all workloads,
and one bracketed workload does not establish gains for other serving settings.

A separate attempt to add Stage2 stopped at its canonical gate. Stage2's
mathematical SNR remained at least 44.96 dB, but per-case oracle/repeat errors
exceeded the declared 1.0 source-relative bound. The unchanged compiler-assembly
control also failed that comparison. Stage2 uses BF16 atomic accumulation;
finite-sample maxima from nondeterministic controls need calibration before
they can distinguish regressions reliably. Restoring atomic address order and
serializing the atomic issues did not pass that contract. Those failures remain
in the record. The [Stage2 validation case](kimi_k3_moe_a16w4_atomic_stage2_gfx950.md)
describes a subsequent source-calibrated bounded-precision contract and exact
isolated expert comparisons. It explicitly distinguishes that new acceptance
from unchanged repeatability. Do not add separate Stage1/Stage2 ratios or treat
the initial rejection as a mathematical 30 dB failure.

## Provenance and transfer limits

The repair was written during manual diagnosis, then assembled, bound through
Forge's FlyDSL adapter and accepted by its updated canonical gate. It was not
discovered by a new autonomous Forge search. It changes assembly only; it does
not change FlyDSL tiles, the production launcher or the mathematical reference.

Evidence is retained under
`/shared_nfs/chenyi/forge-kimi-k3-asm-12h-20260914/stage1-diagnosis/`:
`dependency-probe/`, `group-probe/` and `repaired-canonical/` contain the
ablations, assembly, patch, frozen contract, protected driver and fresh verdict.
The repaired `.s` SHA-256 is
`7cee6ec838bd3707f901172ec73050c63deed4493983cf35138ae5aef6225820`.
The completed Stage1-only model evidence is in `stage1-repair-e2e/` next to
`stage1-diagnosis/`; the rejected Stage2 gate and controls are in
`combined-repair-e2e/`.

Do not generalize these fences or timings to another specialization, assume
every native compiler output is race-free, or relax a numerical contract to
accommodate a faster candidate. If the compiler roundtrip control varies, retain
the failure and diagnose its buffer lifetime and measurement path first.
