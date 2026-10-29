# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What every stage of one forge-loop run shares: the run it works on, and the deadline it works against."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kernelforge.config import Config
from kernelforge.knowledge.kb_store.identity.run import LoopKernel
from kernelforge.loop.bootstrap.errors import ForgeLoopError
from kernelforge.loop.bootstrap.request import ForgeLoopRequest, TimeBudget

if TYPE_CHECKING:
    from kernelforge.loop.campaign_config import CampaignConfig, CampaignConfigStore
    from kernelforge.loop.iteration.iteration_types import IterationConfig
    from kernelforge.tracker import ExperimentTracker, UsageAccumulator


@dataclass(frozen=True)
class _ForgeDeadline:
    """The run's absolute deadline, and the reserve finalization needs out of it.

    Every stage asks the same two questions -- how long is left, and is there
    enough left to start this -- so they are answered in one place rather than
    by two closures over the same pair of numbers.
    """

    deadline_unix: float
    finalize_reserve_sec: float

    @classmethod
    def for_budget(cls, budget: TimeBudget) -> _ForgeDeadline:
        """The caller's absolute deadline, else ``--max-hours`` from now, and the env's finalization reserve."""
        deadline_unix = budget.deadline_unix
        if deadline_unix <= 0:
            deadline_unix = time.time() + budget.max_hours * 3600.0
        finalize_reserve_sec = max(30.0, float(os.environ.get("FORGE_FINALIZE_RESERVE_SEC", "120") or 120))
        return cls(deadline_unix, finalize_reserve_sec)

    def remaining(self, *, reserve: float = 0.0) -> float:
        """Seconds left before the deadline, less whatever reserve the caller must leave."""
        return max(0.0, self.deadline_unix - time.time() - reserve)

    def require(self, phase: str, minimum: float = 1.0) -> None:
        """Refuse to start a phase the finalization reserve no longer leaves room for."""
        if self.remaining(reserve=self.finalize_reserve_sec) < minimum:
            raise ForgeLoopError(f"absolute Forge deadline exhausted before {phase}")


@dataclass(frozen=True)
class _ForgeRun:
    """The request, campaign and collaborators every stage of one forge-loop run works from."""

    request: ForgeLoopRequest
    campaign: CampaignConfig
    campaign_store: CampaignConfigStore
    config: Config
    iter_config: IterationConfig
    tracker: ExperimentTracker
    usage: UsageAccumulator
    deadline: _ForgeDeadline
    publish_result: Callable[[dict], None]

    @property
    def caller_experiment_id(self) -> str:
        """The caller-owned experiment ID: an external recovery channel."""
        return (self.request.artifacts.experiment_id or "").strip()

    @property
    def loop_kernel(self) -> LoopKernel:
        """The kernel this run optimizes, as the knowledge base identifies it."""
        return LoopKernel(
            kernel_path=self.iter_config.kernel_file,
            kernel_backend=self.campaign.kernel_backend,
            target_functions=self.iter_config.target_functions,
            source_files=self.iter_config.source_files,
            framework=self.campaign.framework,
            operator_name=self.campaign.operator_name,
            workspace=self.iter_config.workspace_dir,
        )
