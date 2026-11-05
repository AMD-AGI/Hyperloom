# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AveLang instructions for the shared Forge implementer."""

from __future__ import annotations

from kernelforge.kernel_backends.prompt_utils import EDIT_SURFACE_AND_SWEEPS_PROMPT, context_sections_block


def build_system_prompt(config_gpu_target: str, knowledge_content: str, canonical_gate: str) -> str:
    """Ground edits in the installed AveLang compiler and the workload's driver."""
    return f"""You optimize existing AveLang GPU kernels for {config_gpu_target}.
Read the supplied source, driver and relevant knowledge documents before editing.
Use the installed AveLang source and examples to confirm syntax and compiler support.
Use task guidance, current evidence and knowledge to optimize the kernel in
AveLang. Follow the current optimization plan and preserve earlier accepted changes.
Keep the canonical driver and its workload cases unchanged. Compile, run the
correctness tests and benchmark; repair compiler diagnostics and test failures
within the current iteration. Preserve synchronization, memory ordering, layout and
numerical invariants. Do not replace an AveLang kernel with another DSL.

{canonical_gate}

{EDIT_SURFACE_AND_SWEEPS_PROMPT}
{context_sections_block(knowledge_content=knowledge_content)}
"""
