---
title: "AMDGPU assembly campaign workflow"
kind: guide
scope: languages/assembly
updated: 2026-09-16
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# AMDGPU assembly campaign workflow

`forge-loop --kernel-backend assembly` captures compiler output, verifies its
replacement through the existing launcher, and optimizes only the selected `.s`.
There is no LLM PORT phase or default handwritten seed. Automatic capture
supports one FlyDSL `compile(...)`, Triton/Gluon JIT `kernel[grid](...)`, or
standalone HIP `compile_hip(...)` boundary and one specialization. Resolve
autotuning and choose the actual kernel before capture. Linked libraries still
need an explicit extraction and launcher contract; the adapters do not infer
library dispatch. See the [runtime API](../../../API_docs/runtime_api.md).

A source backend's selected implementation can enter a fresh assembly campaign
with a separate budget. That source remains the performance baseline. The
minimal runnable examples are `examples/flydsl2asm-vector-add/` and
`examples/triton2asm-vector-add/`.

## Preparation and acceptance

1. Commit the source, launcher, independent reference, driver and numerical
   contract. Measure the original source with the protected driver.
2. Capture complete compiler output and validate the unchanged assembly through
   the same callable. Investigate roundtrip regressions before editing instructions.
3. Prove replacement identity: a deliberate assembler error must propagate;
   a no-op candidate must produce a measured SNR or allclose failure on fresh
   outputs beyond compiler warmup. Timeouts, crashes and missing metrics do not
   prove execution. Restore the source and validate it again.
4. Change one instruction-level hypothesis at a time. Keep frontend definitions,
   ABI, launch geometry, streams, oracle and measurement conditions fixed.
5. Run correctness, repeated timing and the canonical numerical suite before
   KEEP. The candidate must also beat the original aggregate time. Without an
   accepted winner, select the source and publish no assembly solution patch.

`config.yaml` must declare numerical coverage and tolerances before optimization.
Fresh source-before/candidate/source-after measurements must cover every declared
case, output and execution mode, with finite outputs, absolute mathematical
limits and source-relative oracle/repeat-error bounds. A 30 dB oracle pass says
nothing by itself about agreement between repeated candidate outputs. Timing
repetitions do not replace synchronized, independently owned output snapshots.
Do not loosen a contract to admit a faster candidate.

## Artifacts and resume

Keep the launcher, provenance manifest and selected `.s` together for clean
replay. Do not publish temporary code objects, compiler caches or benchmark logs
as implementation files. Resume requires schema-3 preparation evidence and
unchanged frozen inputs, including tracked reference helpers; only the selected
assembly may change.

Measure model throughput and quality separately for finalists. Use the
[validation guide](../../profile/hip_module_validation.md) for dispatch and
measurement checks, and the [A16W4 case](kimi_k3_moe_a16w4_gfx950.md) for a bounded
E2E result. If the useful change belongs to algorithm, fusion or layout, start a
source campaign and recapture its compiler output.
