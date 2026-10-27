# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Profile-trace analysis: TraceLens and bypass routes, roofline, and trace self-certification."""

from __future__ import annotations

from ._trace_inputs import count_gpu_kernel_events, discover_trace_inputs
from ._capture_shapes import is_capture_fragment

__all__ = ["count_gpu_kernel_events", "discover_trace_inputs", "is_capture_fragment"]

# The TraceLens commit this repository is pinned to (what a toolbox's preflight compares the importable one with).
from ._tracelens_checkout import _TRACELENS_REF_DEFAULT as TRACELENS_REF

__all__ += ["TRACELENS_REF"]
