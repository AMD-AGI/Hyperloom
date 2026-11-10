---
title: "AMDGPU assembly knowledge map"
kind: index
scope: languages/assembly
updated: 2026-09-16
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# AMDGPU assembly knowledge map

Use assembly to investigate a specific compiler limitation: instruction
scheduling, register pressure, spills, barriers or waits. Read the actual GPU's
ISA and memory-ordering documentation in the hardware knowledge map first.
Structural source changes belong in a source-backend campaign; recapture its
compiler output before starting an assembly campaign.

Load this index first, then only the reference or playbook needed for the task.
The knowledge base retains one measured E2E case; optimization methods and
failure diagnostics are separate from performance claims.

## Reading routes

| Task | Read |
| --- | --- |
| Prepare an assembly campaign; understand correctness and KEEP | [Campaign workflow](skills/optimize/assembly_levers/assembly_workflow.md) |
| Capture complete `.s` and assemble a code object | [Compilation and build](API_docs/compilation_and_build.md) |
| Preserve FlyDSL/Triton/Gluon launchers or load an explicit HIP kernel | [Runtime API](API_docs/runtime_api.md) |
| Select an instruction or register-lifetime experiment | [Instruction optimization](skills/optimize/assembly_levers/instruction_optimization.md) |
| Diagnose unstable outputs, addressing or misleading gains | [Debug an assembly kernel](skills/bottleneck/debug-assembly-kernel.md) |
| Verify execution, the oracle, graphs and measurement | [Candidate validation](skills/profile/hip_module_validation.md) |
| Study an E2E result with its numerical and attribution limits | [Kimi-K3 A16W4 MoE](skills/optimize/assembly_levers/kimi_k3_moe_a16w4_gfx950.md) |

## Folder structure

```text
languages/assembly/
├── INDEX.md
├── API_docs/
│   ├── compilation_and_build.md
│   └── runtime_api.md
└── skills/
    ├── profile/hip_module_validation.md
    ├── bottleneck/debug-assembly-kernel.md
    └── optimize/assembly_levers/
        ├── assembly_workflow.md
        ├── instruction_optimization.md
        └── kimi_k3_moe_a16w4_gfx950.md
```

The retained case starts from already optimized FlyDSL. Its final Stage1
included a manual numerical repair, and Stage2 passed a separately frozen,
source-calibrated precision contract while retaining greater repeat variability.
It does not establish that a new autonomous search will reproduce its gains.
Runnable workflow examples are `examples/flydsl2asm-vector-add/` and
`examples/triton2asm-vector-add/`.
