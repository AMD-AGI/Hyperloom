# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Top-level lifecycle for one kernel rewrite controller invocation."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from kernelforge.durable_io import atomic_write_text
from kernelforge.kernel_rewrite_controller.handoff import read_handoff
from kernelforge.kernel_rewrite_controller.opportunity_agent import (
    ANALYSIS_STATUS_COMPLETED,
    run_opportunity_analysis,
)
from kernelforge.kernel_rewrite_controller.paths import ControllerLayout
from kernelforge.kernel_rewrite_controller.publisher import published_operator_dirs
from kernelforge.kernel_rewrite_controller.recovery import recover_all_task_results
from kernelforge.kernel_rewrite_controller.scheduler import dispatch_prepared_tasks

CONTROLLER_STATE_SCHEMA_VERSION = 1

CONTROLLER_STATUS_RUNNING = "running"
CONTROLLER_STATUS_COMPLETED = "completed"
CONTROLLER_STATUS_NO_OPPORTUNITY = "no_opportunity"
CONTROLLER_STATUS_NO_RESULT = "no_result"
CONTROLLER_STATUS_PARTIAL = "partial"
CONTROLLER_STATUS_FAILED = "failed"


class ControllerRunError(RuntimeError):
    """The controller could not establish or complete its top-level lifecycle."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ControllerRunState:
    """Durable top-level state for one macro cycle invocation."""

    status: str
    handoff_dir: str
    output_dir: str
    budget_minutes: float
    deadline_unix: float
    started_at: str
    finished_at: str = ""
    reason: str = ""
    analysis_status: str = ""
    analysis_reason: str = ""
    analysis_published_task_count: int = 0
    analysis_rejected_task_count: int = 0
    task_count: int = 0
    patch_count: int = 0
    schema_version: int = CONTROLLER_STATE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_budget(budget_minutes: object) -> float:
    if isinstance(budget_minutes, bool) or not isinstance(budget_minutes, (int, float)):
        raise ControllerRunError("budget_minutes must be a positive number")
    budget = float(budget_minutes)
    if not math.isfinite(budget) or budget <= 0:
        raise ControllerRunError("budget_minutes must be a positive finite number")
    return budget


def _write_state(layout: ControllerLayout, state: ControllerRunState) -> None:
    atomic_write_text(
        layout.controller_state,
        json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n",
    )


def _write_summary(layout: ControllerLayout, state: ControllerRunState) -> None:
    lines = [
        "# Kernel Rewrite Controller Result",
        "",
        f"- **Status:** `{state.status}`",
        f"- **Reason:** {state.reason or 'none'}",
        f"- **Handoff directory:** `{state.handoff_dir}`",
        f"- **Output directory:** `{state.output_dir}`",
        f"- **Budget minutes:** `{state.budget_minutes:g}`",
        f"- **Deadline Unix:** `{state.deadline_unix:.6f}`",
        f"- **Started at:** `{state.started_at}`",
        f"- **Finished at:** `{state.finished_at or 'not finished'}`",
        f"- **Task count:** `{state.task_count}`",
        f"- **Patch count:** `{state.patch_count}`",
        f"- **Analysis status:** `{state.analysis_status or 'not started'}`",
        f"- **Analysis reason:** {state.analysis_reason or 'none'}",
        f"- **Analysis published tasks:** `{state.analysis_published_task_count}`",
        f"- **Analysis rejected tasks:** `{state.analysis_rejected_task_count}`",
        "",
    ]
    patches = published_operator_dirs(layout)
    if patches:
        lines.extend(["## Published Operator Patches", ""])
        lines.extend(f"- `{path.name}`" for path in patches)
        lines.append("")
    atomic_write_text(layout.summary_md, "\n".join(lines))


def _initialize_layout(layout: ControllerLayout) -> None:
    if layout.controller_root.exists() or layout.result_root.exists():
        raise ControllerRunError(f"output directory is already initialized and cannot be resumed: {layout.output_dir}")
    for path in (
        layout.tasks_root,
        layout.workspaces_root,
        layout.patches_root,
    ):
        path.mkdir(parents=True, exist_ok=False)


def run_controller(
    *,
    handoff_dir: str | Path,
    budget_minutes: float,
    output_dir: str | Path,
) -> ControllerRunState:
    """Initialize one fresh controller run and emit a no-opportunity result."""
    budget = _validate_budget(budget_minutes)
    handoff_path = Path(handoff_dir).expanduser().resolve()
    layout = ControllerLayout(Path(output_dir))
    _initialize_layout(layout)

    started_unix = time.time()
    started_at = _now_iso()
    running = ControllerRunState(
        status=CONTROLLER_STATUS_RUNNING,
        handoff_dir=str(handoff_path),
        output_dir=str(layout.output_dir),
        budget_minutes=budget,
        deadline_unix=started_unix + budget * 60.0,
        started_at=started_at,
    )
    _write_state(layout, running)
    _write_summary(layout, running)

    try:
        handoff = read_handoff(handoff_path)
        recover_all_task_results(layout)
        analysis = run_opportunity_analysis(
            handoff=handoff,
            layout=layout,
            controller_deadline_unix=running.deadline_unix,
        )
        schedule = dispatch_prepared_tasks(
            layout,
            controller_deadline_unix=running.deadline_unix,
        )
        recover_all_task_results(layout)
        patch_count = len(published_operator_dirs(layout))
    except Exception as error:
        failed = ControllerRunState(
            **{
                **running.to_dict(),
                "status": CONTROLLER_STATUS_FAILED,
                "finished_at": _now_iso(),
                "reason": f"handoff validation failed: {error}",
            }
        )
        _write_state(layout, failed)
        _write_summary(layout, failed)
        raise ControllerRunError(failed.reason) from error

    if analysis.status != ANALYSIS_STATUS_COMPLETED and schedule.task_count == 0 and patch_count == 0:
        status = CONTROLLER_STATUS_FAILED
        reason = analysis.reason or "opportunity analysis did not complete"
    elif analysis.status != ANALYSIS_STATUS_COMPLETED:
        status = CONTROLLER_STATUS_PARTIAL
        reason = analysis.reason or "opportunity analysis was incomplete"
    elif schedule.task_count == 0 and patch_count == 0 and analysis.rejected_task_count:
        status = CONTROLLER_STATUS_NO_RESULT
        reason = "opportunity analysis produced only invalid tasks"
    elif schedule.task_count == 0 and patch_count == 0:
        status = CONTROLLER_STATUS_NO_OPPORTUNITY
        reason = "no prepared operator tasks are available"
    elif schedule.stopped_for_budget:
        status = CONTROLLER_STATUS_PARTIAL
        reason = "controller stopped admitting tasks because less than 30 minutes remained"
    elif patch_count:
        status = CONTROLLER_STATUS_COMPLETED
        reason = "all prepared operator tasks were processed"
    else:
        status = CONTROLLER_STATUS_NO_RESULT
        reason = "prepared operator tasks produced no validated improvement"
    completed = ControllerRunState(
        **{
            **running.to_dict(),
            "status": status,
            "finished_at": _now_iso(),
            "reason": reason,
            "analysis_status": analysis.status,
            "analysis_reason": analysis.reason,
            "analysis_published_task_count": analysis.published_task_count,
            "analysis_rejected_task_count": analysis.rejected_task_count,
            "task_count": schedule.task_count,
            "patch_count": patch_count,
        }
    )
    _write_state(layout, completed)
    _write_summary(layout, completed)
    return completed


__all__ = [
    "CONTROLLER_STATE_SCHEMA_VERSION",
    "CONTROLLER_STATUS_COMPLETED",
    "CONTROLLER_STATUS_FAILED",
    "CONTROLLER_STATUS_NO_OPPORTUNITY",
    "CONTROLLER_STATUS_NO_RESULT",
    "CONTROLLER_STATUS_PARTIAL",
    "CONTROLLER_STATUS_RUNNING",
    "ControllerRunError",
    "ControllerRunState",
    "run_controller",
]
