---
title: "Choose an instruction-level optimization"
kind: guide
scope: languages/assembly
updated: 2026-09-16
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Choose an instruction-level optimization

Start with the executed specialization and a profile-supported hypothesis.
Use the [campaign workflow](assembly_workflow.md) to retain the source baseline
and the [validation guide](../../profile/hip_module_validation.md) to verify the
result. Fewer instructions or a smaller descriptor alone do not establish a gain.

## Registers and load scheduling

For every value affected by a move, record its producer, last consumer, register
range and active lanes. Include destinations of outstanding loads. Delay loads
with long live ranges, reuse registers only after their last consumer, or
recompute inexpensive address temporaries when that reduces peak allocation.
Update all register references and matching allocation metadata together.

One Forge A8W4 experiment reduced 134 VGPRs to 128 by delaying an LDS read,
reusing dead weight-fragment registers and reconstructing address temporaries.
Theoretical residency rose from three to four 256-thread blocks per CU, but
some routing regimes regressed. Treat occupancy queries as upper bounds and
measure the actual workload. Its small, workload-specific serving result does
not justify a universal 128-register target.

Do not reduce declared VGPRs or LDS bytes without changing the actual live ranges
or addresses. Do not move a load over a barrier without checking both register
lifetimes and the memory dependency. An `exec` mask does not give the remaining
active lane a different register bank.

## Independent arithmetic chains

When one reduction chain waits on dependencies, interleave instructions from an
independent chain with the same data available. Neha/Evolve's
[AttnRes score source](https://github.com/sgl-project/sglang/blob/fbbf9e6e16d7f5295512ae3159555a3467afc348/python/sglang/srt/layers/kimik3_attnres_score.s)
provides a DPP-reduction schedule to inspect. Check lane masks, wave size,
partial-result storage and numerical ordering before transferring it.

The [combine source](https://github.com/sgl-project/sglang/blob/1a505cb04072f2de48349a86e9603860eedb6510/python/sglang/srt/layers/kimik3_attnres_combine.s)
interleaves softmax arithmetic but places BF16 loads after the barrier. Its
proposed early-load schedule would overwrite live scores in `v2..v10`.
Read the final instructions rather than adopting a historical header proposal.
These sources illustrate techniques; Forge did not reproduce their published
headline gains or establish a model E2E gain for that replacement.

## Waits and cache policy

Identify the exact operation tracked by each wait counter on the target ISA.
A per-wave wait and a workgroup barrier have different jobs; moving either can
change the lifetime of a shared tile. Audit both completion of writes and
completion of readers before reuse. See the [A16W4 repair](kimi_k3_moe_a16w4_gfx950.md).

Test cache hints and prefetch distance with both representative reuse and cold
or evicted inputs. Fix input/routing distributions before comparing candidates;
a warm-cache gain can reverse when expert weights leave cache. Preserve the
ABI's output initialization in eager and graph execution. Retain regressions
and report kernel measurements separately from serving throughput and latency.

The Evolve links are published kernels, not a recovered agent implementation.
