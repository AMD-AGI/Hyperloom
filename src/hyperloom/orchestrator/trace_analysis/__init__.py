# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Profile-trace analysis: the agent and non-llm TraceLens routes, roofline, and trace self-certification."""

from __future__ import annotations

from ._capture_shapes import discover_capture_folder, is_capture_fragment
from ._stream_overlap_repair import clamp_or_skip_reason
from ._trace_inputs import count_gpu_kernel_events, discover_trace_inputs
from ._trace_split import (
    DECODE_ONLY_OR_GENERIC_MODE,
    UNSPLIT_MODE,
    TraceSplitJob,
    copy_unsplit_trace,
    default_split_num_steps,
    run_trace_split,
)
from ._tracelens_checkout import run_command

__all__ = [
    "DECODE_ONLY_OR_GENERIC_MODE",
    "UNSPLIT_MODE",
    "TraceSplitJob",
    "clamp_or_skip_reason",
    "copy_unsplit_trace",
    "count_gpu_kernel_events",
    "default_split_num_steps",
    "discover_capture_folder",
    "discover_trace_inputs",
    "is_capture_fragment",
    "run_command",
    "run_trace_split",
]

# The TraceLens commit this repository is pinned to (what a toolbox's preflight compares the importable one with).
from ._tracelens_checkout import _TRACELENS_REF_DEFAULT as TRACELENS_REF

__all__ += ["TRACELENS_REF"]
