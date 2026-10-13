# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The state one forge-loop iteration carries between choosing, measuring and recording its candidate."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kernelforge.loop.device_hazard import DeviceHazard
    from kernelforge.loop.merge_candidates import MergeCandidate
    from kernelforge.loop.runner import HeldRound, IterationResult


@dataclass(frozen=True)
class LoopServices:
    """The collaborators and callbacks one ``IterationLoop.run`` call was handed."""

    agent_fn: Any = None
    agent_factory: Any = None
    analysis_service: Any = None
    orchestration_service: Any = None
    supervisor_fn: Any = None
    on_iteration: Any = None
    on_best_committed: Any = None
    on_best_ready: Any = None


class IterationFlow(Enum):
    """Where an iteration goes once its candidate has been chosen."""

    MEASURE = "measure"
    NEXT = "next"
    STOP = "stop"


@dataclass
class IterationAttempt:
    """What one iteration carries from choosing its candidate to recording it."""

    iteration: int
    digest: str
    hazard: DeviceHazard | None
    session_sink: dict = field(default_factory=dict)
    flow: IterationFlow = IterationFlow.MEASURE
    optimization_plan_path: str = ""
    optimization_plan_executable: bool = False
    fan_out_plan: HeldRound | None = None
    merge_pair: tuple[MergeCandidate, MergeCandidate] | None = None
    merge_diff: str = ""
    unmeasured_result: IterationResult | None = None
    commit_hash: str = ""
    rationale: str = ""
    attempt_source: str = ""
    attempt_diff: str = ""
    reusable_benchmark: dict | None = None
    agent_error: Exception | None = None
    best_before: float | None = None
    best_mean_case_speedup_before: float | None = None
    pending_keep: dict | None = None
    keep_checkpoint_finalized: bool = False


def append_session_finding(session_sink: dict, summary: str) -> None:
    """Append ``summary`` to the findings a session reported."""
    session_sink["findings"] = "\n---\n".join(
        part
        for part in (
            str(session_sink.get("findings") or ""),
            summary,
        )
        if part
    )
