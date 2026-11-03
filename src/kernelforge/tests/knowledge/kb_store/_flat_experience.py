# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Writing a flat local experience record, shared by the KB store writer and reader tests."""

from __future__ import annotations

import dataclasses

from kernelforge.knowledge.kb_store import writer as sink
from kernelforge.knowledge.kb_store.identity.run import LoopKernel
from kernelforge.knowledge.kb_store.writer import RunOutcome

KERNEL_SOURCE = "import triton\n\n\n@triton.jit\ndef my_kernel(x):\n    return x\n"
SUMMARY = {
    "category": "gemm",
    "strategy": "vectorize loads",
    "recipe": "Use vectorized loads.",
    "lessons": "Alignment matters.",
}


_LOOP_KERNEL_FIELDS = {field.name for field in dataclasses.fields(LoopKernel)}


def write_flat_experience(config, fields: dict) -> dict:
    """Write one run experience from the flat keyword fields the KB tests spell."""
    fields = dict(fields)
    summary_override = fields.pop("summary_override", SUMMARY)
    kernel = LoopKernel(**{name: fields.pop(name) for name in _LOOP_KERNEL_FIELDS if name in fields})
    return sink.write_run_experience(
        config=config, kernel=kernel, outcome=RunOutcome(**fields), summary_override=summary_override
    )
