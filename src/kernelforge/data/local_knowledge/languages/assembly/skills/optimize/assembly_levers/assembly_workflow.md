---
title: AMDGPU assembly campaign workflow
kind: guide
scope: languages/assembly
updated: 2026-09-16
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# AMDGPU assembly campaign workflow

`forge-loop --kernel-backend assembly` captures compiler output programmatically,
verifies rebuilding/loading through the existing launcher, and permits only the
selected `.s` to change. There is no correctness-only LLM rewrite or default
handwritten seed. Automatic capture currently supports one explicit
`flydsl.compiler.compile(...)` call and one specialization; unsupported frontends
must first supply a verified assembly binding. Do not improvise a source rewrite.
The minimal example is `examples/flydsl2asm-vector-add/`.

A source backend can run first, then its best implementation can enter a fresh
assembly campaign as an optional second stage. Keep its source result as the
baseline. Do not run this stage unconditionally for every language: Triton/Gluon,
HIP and operator libraries need distinct extraction/replacement adapters.
A deliberate build failure must propagate through the protected driver, and a
no-op assembly must report a measured SNR or allclose failure on fresh outputs
beyond compiler warmup. Timeouts, build/load errors and unclassified assertions
do not prove execution. Resume requires schema-3 preparation evidence and
unchanged frozen inputs, including reference helpers. Record
source, unchanged roundtrip and optimized ASM separately. Preparation is not a
KEEP; return the source when instruction optimization has no accepted benefit.

Before optimization, supply the protected `config.yaml` numerical contract and
fresh structured source/candidate measurements from the correctness driver.
Forge requires complete declared coverage and both absolute mathematical and
source-relative repeat-output error bounds before an ASM KEEP. A 30 dB
candidate-to-oracle pass does not imply that two candidate outputs agree at
30 dB. Use synchronized independent snapshots; timing repetitions are not
repeatability evidence. Do not loosen the contract after seeing a faster result.
Model quality and E2E throughput remain separate finalist checks.

When waits or prefetches change output repeatability, read the
[A16W4 LDS reuse repair](kimi_k3_moe_a16w4_lds_reuse_gfx950.md).
It distinguishes per-wave memory completion from cross-wave synchronization
before overwriting a shared tile. The repaired Stage1 passed the canonical gate
and measured a 1.86% E2E gain on one bracketed eight-GPU workload. The card also
records the limited model-quality evaluation. For BF16 atomic reductions, read
the [Stage2 validation case](kimi_k3_moe_a16w4_atomic_stage2_gfx950.md):
an unchanged compiler control failed the initial finite-max comparison, while
a separately frozen source-calibrated contract and isolated expert contributions
passed. Bounded precision and unchanged output repeatability are different claims.
The combined repaired-Stage1/Stage2 serving trial measured 2.26% against the
faster source control (2.40% against pooled source). Fixed-count full-question
replicas did not reproduce a persistent model-quality loss; the card retains
the initial MMLU miss, changed numerical contract and limits of that evidence.

## Evidence and artifacts

1. Measure the original high-level implementation with the protected driver.
2. Run the unmodified assembly through the same callable contract. Require the
   complete correctness suite before optimizing. Record roundtrip timing; investigate
   regressions in the binding instead of assuming compilation implies timing parity.
   The original implementation remains the scoring incumbent.
3. Make one instruction change and rerun correctness, graph-capture
   verification, per-case timing, and relevant counters. Keep the same inputs,
   dtype, tolerances, stream, launch dimensions, and measurement method.
4. Before trusting the route, use a disposable negative control that changes
   the output and confirm the unchanged oracle rejects it. Restore that edit.
5. Keep the launcher and `.s` in source control together. Use explicit
   source paths during preparation; the host tracks the binding, provenance manifest
   and compiler-emitted `.s`. During optimization, only the selected `.s` uses the existing KEEP/REVERT path. Do not commit temporary `.o`,
   `.hsaco`, IR dumps, compiler caches, or benchmark logs.

Assembly does not guarantee a speedup. Report parity, regression, and variance
as measured. When the limiting factor is an algorithm or layout, start a separate
high-level campaign, regenerate assembly, and repeat roundtrip validation.
