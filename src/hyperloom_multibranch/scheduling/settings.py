# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What one multi-branch run is given: the InferenceX workload, the wall-clock and GPU-hour budgets and the hard
limits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom_multibranch.scheduling.roster import RosterCaps

#: The metric an InferenceX session's values are read in: the frontier policy, final selection, the dashboard and
#: the final recipe's metric.
PRIMARY_METRIC = GRADED_OUTPUT


class KernelOptimizer(str, Enum):
    """The kernel optimizer a session's recipes are scoped to in the Recipe KB."""

    FORGE = "forge"
    GEAK = "geak"


@dataclass(frozen=True)
class Workload:
    """The fixed InferenceX workload, spelled as the single-lineage command line spells it, and the kernel optimizer
    its recipes are scoped to.

    ``gpu_type`` is named, not detected: the toolbox writes it into every measurement spec, built before PRELUDE runs.
    """

    model: str
    framework: str
    tp: int
    isl: int
    osl: int
    conc: int
    precision: str
    gpu_type: str
    kernel_optimizer: KernelOptimizer

    @property
    def model_basename(self) -> str:
        return Path(self.model.rstrip("/")).name


@dataclass(frozen=True)
class RunSettings:
    """``max_hours`` is the whole session's wall-clock budget, PRELUDE included, and ``max_gpu_hours`` the GPU hours
    the session's tests may use; 0 means no limit for either."""

    workload: Workload
    max_hours: float
    max_gpu_hours: float
    caps: RosterCaps
    judge_interval_seconds: float


__all__ = ["PRIMARY_METRIC", "KernelOptimizer", "RunSettings", "Workload"]
