# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The forge-loop's stdout lines that a run's log is read back for.

This module owns those lines end to end: the loop prints them through the renderers here, and ``kernelforge
export-runs`` reads a log with the patterns compiled from the same pieces. A log has no timestamps, so it is read only
for the experiment each session announces and for the record of Kernel Forge's own experience KB each iteration wrote.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

LOOP_BANNER = "Starting autonomous iteration loop"

_EXPERIMENT_LABEL = "  Experiment: "
_ITERATION_LABEL = "--- Iteration "
_KB_WRITTEN_LABEL = "[kb] experience written: "

EXPERIMENT_LINE = re.compile(rf"^{re.escape(_EXPERIMENT_LABEL)}(\S+)\s*$", re.M)
ITERATION_HEADER = re.compile(rf"^{re.escape(_ITERATION_LABEL)}(\d+)\b")
KB_WRITTEN = re.compile(rf"^\s*{re.escape(_KB_WRITTEN_LABEL)}(\S+)")


def experiment_line(experiment_id: str) -> str:
    """The line that names the experiment id of the session now running."""
    return f"{_EXPERIMENT_LABEL}{experiment_id}"


def iteration_header(iteration: int, best_mean_case_speedup: float | None, remaining_s: float) -> str:
    """The line that opens an iteration's output."""
    if best_mean_case_speedup is None:
        return f"{_ITERATION_LABEL}{iteration} ---"
    return (
        f"{_ITERATION_LABEL}{iteration} "
        f"(best mean case speedup: {best_mean_case_speedup:.6f}x, remaining: {remaining_s / 60:.0f} min) ---"
    )


def kb_write_line(status: Mapping[str, Any]) -> str:
    """What became of writing the run to Kernel Forge's own experience KB, from the ``status`` the writer returned."""
    if status.get("written"):
        return f"  {_KB_WRITTEN_LABEL}{status.get('solution')} (speedup {status.get('speedup'):.3f})"
    return f"  [kb] experience not written: {status.get('reason')}"
