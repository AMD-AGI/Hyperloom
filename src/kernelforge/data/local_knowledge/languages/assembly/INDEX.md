---
title: AMDGPU assembly knowledge map
kind: index
scope: languages/assembly
updated: 2026-09-16
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# AMDGPU assembly knowledge map

Use assembly to test a specific compiler limitation: instruction scheduling,
register pressure, spills, barriers, or waits. Structural changes belong first
in a separate FlyDSL, Triton/Gluon, or HIP campaign; recapture compiler output after
such a change. Inside an assembly optimization loop, only its selected `.s` is editable. Read the actual GPU's ISA and memory-ordering documentation from
the hardware knowledge map before changing synchronization or register usage.

## Reading order

As with the other language knowledge folders, load `INDEX.md` first and follow
its links to the relevant API reference or task playbook. Compiler and runtime
facts live in `API_docs/`; validation and optimization procedures live in `skills/`.
The measured cases document assembly-specific techniques and their limits,
not a duplicate catalog of backend-independent operator contracts.

| Task | Read |
| --- | --- |
| Run a compiler-output assembly campaign; understand KEEP and numerical gates | [Assembly workflow](skills/optimize/assembly_levers/assembly_workflow.md) |
| Capture complete `.s` and build a code object | [Compilation and build](API_docs/compilation_and_build.md) |
| Retain the FlyDSL ABI or load an explicit HIP kernel | [Runtime API](API_docs/runtime_api.md) |
| Verify candidate identity, correctness, streams, graphs and timing | [HIP module validation](skills/profile/hip_module_validation.md) |
| Diagnose LDS reuse and cross-wave synchronization errors | [A16W4 Stage1 repair](skills/optimize/assembly_levers/kimi_k3_moe_a16w4_lds_reuse_gfx950.md) |
| Separate BF16 atomic variability from a candidate regression | [A16W4 Stage2 controls](skills/optimize/assembly_levers/kimi_k3_moe_a16w4_atomic_stage2_gfx950.md) |
| Select an instruction change using measured evidence | The case routes below |

## Folder structure and file roles

```text
languages/assembly/
├── INDEX.md
├── API_docs/
│   ├── compilation_and_build.md
│   └── runtime_api.md
└── skills/
    ├── profile/
    │   └── hip_module_validation.md
    └── optimize/
        └── assembly_levers/
            ├── assembly_workflow.md
            ├── aiter_w4a16_roundtrip_gfx950.md
            ├── evolve_attnres_score_gfx950.md
            ├── evolve_attnres_combine_gfx950.md
            ├── kimi_k3_moe_a8w4_gfx950.md
            ├── kimi_k3_moe_a16w4_lds_reuse_gfx950.md
            ├── kimi_k3_moe_a16w4_atomic_stage2_gfx950.md
            └── qwen3_qk_rope_gfx950.md
```

## Case knowledge: Neha / Evolve

Read these cards when the symptom matches. Paths are relative to this folder.
They distill Neha Prakriya's published Evolve-produced kernels, launcher, and
tests, with commit-pinned sources. They are not a copy of Evolve's agent skill
or search controller, which are not available in these sources.

| Symptom or decision | Read | Transferable lesson |
| --- | --- | --- |
| Two reductions over the same input; dependency-bound wave reduction | [AttnRes score](skills/optimize/assembly_levers/evolve_attnres_score_gfx950.md) | Interleave independent DPP chains; finish wave partials through compact LDS. |
| Small softmax followed by a weighted sum; proposed load/barrier overlap | [AttnRes combine](skills/optimize/assembly_levers/evolve_attnres_combine_gfx950.md) | Schedule independent exponentials; audit live registers before moving loads. |
| A standalone `.s`/`.co` looks fast but its execution path is unverified | [HIP module integration](skills/profile/hip_module_validation.md) | Verify ABI, candidate identity, stream, oracle, and timing before accepting a result. |

These cases replace Triton kernels with handwritten gfx950 assembly. The
AITER launcher lives under `ops/flydsl/` but uses HIP module APIs directly;
it is not evidence of FlyDSL `CompiledFunction` compatibility. Their fixed
Kimi-K3 shape and author-reported timings do not establish performance on
other shapes, GPUs, MoE kernels, or Forge's FlyDSL adapter. The cards distinguish
source observations, reported results, and experiments still to run.

Independent MI355X reproduction is recorded in both cards. Distinguish the
production Triton score from AITER's faster test reference, and audit the score
ASM's hardcoded epsilon. A real Forge campaign improved the ASM seed by 1.314x
through empty-exec tail skipping and scalar-base addressing, while remaining
slower than the original test-reference Triton. Combine requires address-carry
corrections before wider allocation testing. Current SGLang also has a fused
path that bypasses the replaced score branch. The published headline ratios
and a model E2E gain were not reproduced; use the measured scope and full caller
contract when choosing a case.

## Case knowledge: verified FlyDSL roundtrip

For AITER's INT4/BF16 MoE, read the
[W4A16 stage1 case](skills/optimize/assembly_levers/aiter_w4a16_roundtrip_gfx950.md). It covers the actual
FlyDSL launcher adapter, weight/scale layouts, negative controls, and a packed
multiply experiment that passed correctness but produced no useful speedup.
This is Forge validation evidence, separate from the Evolve source cases.

For the FP8/MXFP4 SiTUv2 path, read the
[Kimi-K3 MoE stage1 case](skills/optimize/assembly_levers/kimi_k3_moe_a8w4_gfx950.md).
It validates one FlyDSL 0.3.2 roundtrip, explains the required sorted activation
scale layout, and retains five initial instruction candidates with no stable
winner. A later real Forge campaign reduced 134 VGPRs to 128 through liveness
and load-scheduling changes. Independent routing holdouts and reverse-order
Kimi-K3 serving trials measured a small 0.31%-0.43% latency reduction on one
fixed diverse workload; near-identical prompts did not establish a stable gain.
Matched original-FlyDSL controls also showed a smaller 0.17%-0.40% total
integration benefit on the same diverse requests.
Read its scope, output-repeatability limits, and separate source/PORT/ASM
attribution before transferring the result. Its earlier cache-policy holdouts
also show why a warm-cache gain can reverse after eviction.

## Case knowledge: Qwen3 model integration

The [Qwen3 Q/K normalization and RoPE case](skills/optimize/assembly_levers/qwen3_qk_rope_gfx950.md)
connects a standalone handwritten kernel to an existing vLLM model through
Forge's explicit-ABI HIP loader. It covers graph/worker dispatch verification,
BF16 intermediate rounding, same-fusion attribution, and the distinction
between decode throughput and first-token latency. It distinguishes the manual
fusion/numerical repair from a real Forge agent's staged-VMEM scheduling KEEP,
clean export replay, independent holdouts, and fault-injection REVERT. This is
a standalone HIP integration, not a FlyDSL source change. A kernel KEEP alone
does not establish an additional model-serving gain.
