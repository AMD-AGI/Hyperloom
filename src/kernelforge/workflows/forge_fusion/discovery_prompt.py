# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The text of the decode-path discovery prompt, with a named slot for every piece of evidence it cites."""

from __future__ import annotations

DISCOVERY_PROMPT_TEMPLATE = """You are analyzing the DECODE path of a {framework} model (`model_type={model_type}`)
on AMD MI325X (gfx942), bf16, to find SOURCE-LEVEL KERNEL FUSIONS. Analyze only; do
not edit anything. Return your answer as JSON (schema below).

## The lever (general, not model-specific)
The decode path is launch/dispatch bound: launch_bound_share={diagnosis.launch_bound_share:.2f}
of GPU-busy time is in many tiny fp32 ops ({lb}) rather than the heavy GEMM/attention.
Each tiny op is a separate kernel launch + HBM round-trip. The win is to FUSE a
CONTIGUOUS chain of these tiny ops (as they appear in the decode forward, between two
heavy ops) into ONE kernel — fewer launches, fewer round-trips.

## Measured launch-bound hot kernels (top, this trace)
{hot_lines}
Dominant launch-bound categories: {dom}
Representative decode shapes: {shapes}

## Ordered fusion boundaries (same stream, compute endpoints retained)
{boundary_lines}

## Existing ROCm operator evidence
Retrieved by whole-word overlap between the observed kernel/category names and the
documented operation chain; a higher score means stronger evidence. The score is a
retrieval hint, NOT a correctness claim: you must still confirm from the source and
the card that shape, dtype, and cache layout actually match.
{operator_lines}

## Your task
{read_verb} up to {max_fusions} CONTIGUOUS op chains in
the DECODE forward that are worth fusing. Include GEMM/attention prologue or epilogue
boundaries when the ordered trace proves adjacency. Treat an existing ROCm operator
that covers a larger boundary as an `integration` candidate and benchmark/wire it
before proposing a new kernel. Judge from the SOURCE and ordered trace what actually
runs back-to-back on the decode path.

{fusion_constraints}

{output_schema_block}

{source_block}
"""
