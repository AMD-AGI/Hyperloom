# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The in-session gate's outcome tag that opens an Implementer session's rationale.

This module owns the tag end to end: the Implementer agent puts it in front of the agent's own text, and
``kernelforge export-runs`` strips it to read that text back.
"""

from __future__ import annotations

import re

_OPENING = "[gate edits="
_TAG = re.compile(rf"^{re.escape(_OPENING)}[^\]]*\]\s*")


def gate_outcome_tag(gate, end_reason: str, num_turns: int | None) -> str:
    """The gate outcome, so the outer loop's commit message records whether the session self-converged."""
    tag = f"{_OPENING}{gate.edit_count} pass={gate.passed} end={end_reason}"
    if num_turns is not None:
        tag += f" turns={num_turns}"
    if gate.last_mean_case_speedup is not None:
        tag += f" mean_case_speedup={gate.last_mean_case_speedup:.6f}x"
    if gate.last_wall_ms is not None:
        tag += f" raw_mean={gate.last_wall_ms:.4f}ms"
    return tag + "]"


def strip_gate_outcome_tag(rationale: str) -> str:
    """``rationale`` without the tag ``gate_outcome_tag`` put in front of it."""
    return _TAG.sub("", rationale, count=1)
