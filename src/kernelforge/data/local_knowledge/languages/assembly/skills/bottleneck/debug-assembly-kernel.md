---
title: "Debug an assembly kernel"
kind: guide
scope: languages/assembly
updated: 2026-09-16
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Debug an assembly kernel

Classify the failure before changing the instruction schedule. Keep the original
source and unchanged compiler-assembly control available, and use the same inputs,
oracle and execution mode for both. The [validation guide](../profile/hip_module_validation.md)
defines the measurement boundary.

| Symptom | Check and next action |
| --- | --- |
| Outputs vary after moving waits or prefetches | Map LDS readers and overwrites per wave. A wait drains that wave; synchronize all consumers before reusing shared storage. A barrier after overwrite is too late. Do not insert barriers in divergent control flow. |
| BF16 atomic reduction varies but individual experts agree | Reset outputs on every launch, including graph replay. Isolate one expert contribution at a time, then compare the full reduction. A changed issue order can change rounding despite identical contributions. |
| Both source control and candidate fail a repeat-error gate | Diagnose the control and measurement protocol first. Finite-sample maxima are not proof of equal distributions. Any new task contract needs source-only calibration and fresh holdouts; retain the original rejection. |
| Illegal accesses depend on allocation | Audit low-half additions and high-half carries, strides, complete buffer extents and LDS descriptors. Test valid views spanning address boundaries and the real caller's allocation sizes. |
| Stable but systematically different output | Trace runtime scalars such as epsilon, quantization scales, rounding mode and intermediate dtype. Fusion or staging can move a BF16 rounding boundary. Use the independent mathematical oracle and preserve public arguments. |
| An edit appears ineffective or unexpectedly fast | Rebuild a fresh callable; do not reuse a loaded code object just because its file changed. Check candidate identity, graph capture stream and changed-input replay. Empty graphs and skipped cases are unevaluated. |
| A prologue no-op still produces correct output | On gfx950, kernarg preloading can enter an aligned LLVM basic block after the prologue. Inspect the descriptor and all executed entry paths. The preparation probe stops compiler basic-block entries as well; a probe that still passes must reject preparation. |
| Kernel improves but serving does not | Verify model dispatch on every worker and include the complete caller. Compare against equivalent fusion and precision, account for hotspot weight, and inspect TTFT/tail latency as well as token throughput. |

The [A16W4 case](../optimize/assembly_levers/kimi_k3_moe_a16w4_gfx950.md)
demonstrates the difference between an LDS lifetime defect and nondeterministic
atomic accumulation. Its numerical policy is specific to that experiment;
do not copy the tolerance into another kernel.

An output compared to an oracle and two repeated outputs answer different
questions. Track both errors with independent snapshots and meaningful
zero-reference handling. Passing a mathematical SNR threshold cannot excuse a
separate failed repeatability contract. A passing kernel gate also does not
certify model quality.

For source attribution, Neha/Evolve's
[launcher and test sources](https://github.com/ROCm/aiter/pull/4863)
motivated checks for hardcoded epsilon, address carries, BF16 conversion and
dispatch. Qwen3 fusion experiments motivated the equivalent-fusion and tail
latency checks. Neither is retained as a validated general ASM E2E success.
