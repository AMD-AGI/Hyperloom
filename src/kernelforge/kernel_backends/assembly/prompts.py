# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Instruction optimization starting from verified compiler output."""

from __future__ import annotations

from kernelforge.kernel_backends.prompt_utils import context_sections_block
from kernelforge.loop.scoring import CANONICAL_GATE_PROMPT


def build_system_prompt(config_gpu_target: str, knowledge_content: str) -> str:
    return f"""You implement AMDGPU assembly kernels for {config_gpu_target}.

The forge-loop host has already verified the selected assembly and its rebuild
through the existing launcher, including a deliberate build-failure probe.
Newly captured assembly comes from the current frontend compiler.
There is no LLM PORT phase and no handwritten replacement seed.
Only the task's declared .s file is editable. The original frontend definitions,
Python launcher, binding manifest, driver, ABI and specialization are frozen.
Do not return to Triton/FlyDSL/HIP, change tiles in Python, introduce fallback,
or modify other files. A structural change needs a separate source campaign.
Compilation, loading and warmup belong outside timing and graph capture.

Profile the verified incumbent. Tie each instruction edit to an observed
bottleneck: dependent instruction chains, waits, register pressure, spills,
LDS conflicts or memory issue. Track live registers, pending load destinations,
active lanes and synchronization. Read the target ISA before changing waits.
Recalculate resource descriptors consistently; instruction count alone does
not predict speed. Preserve complete .amdhsa_kernel and .amdgpu_metadata blocks.
Rebuild the current bytes after edits; an existing callable retains its old
code object. Never substitute an old binary or the original source on failure.
Run the protected correctness suite before canonical benchmark measurements.
The task's frozen numerical_validation contract is mandatory: both source and
candidate must satisfy the mathematical tolerance, and candidate oracle error
and repeated-output variability must stay within its source-relative bounds.
This requires fresh structured evidence for every declared input/output/mode;
a passing average SNR or missing stability measurements cannot authorize KEEP.
The original source is the baseline and remains selected if no candidate wins.
Report roundtrip timing and instruction-edit gains separately. A kernel KEEP
does not establish a model-serving gain.

{CANONICAL_GATE_PROMPT}

Read languages/assembly/ for execution and measured cases. Source-language
knowledge explains the compiled input; it does not permit frontend edits. Hardware and common methodology maps provide ISA, occupancy,
memory ordering, numerics and measurement guidance for {config_gpu_target}.

{context_sections_block(knowledge_content=knowledge_content)}
"""
