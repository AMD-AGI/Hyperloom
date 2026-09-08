# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Top-level lifecycle for one kernel rewrite controller invocation."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kernelforge.durable_io import atomic_write_text
from kernelforge.kernel_rewrite_controller.handoff import read_handoff
from kernelforge.kernel_rewrite_controller.opportunity_agent import (
    ANALYSIS_STATUS_COMPLETED,
    run_opportunity_analysis,
)
from kernelforge.kernel_rewrite_controller.paths import ControllerLayout
from kernelforge.kernel_rewrite_controller.publisher import (
    PUBLICATION_FILENAME,
    published_operator_dirs,
)
from kernelforge.kernel_rewrite_controller.dispatcher import SingleTaskResult
from kernelforge.kernel_rewrite_controller.recovery import (
    RecoveryResult,
    recover_all_task_results,
)
from kernelforge.kernel_rewrite_controller.scheduler import dispatch_prepared_tasks

log = logging.getLogger(__name__)

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
    #: Why each draft the analysis agent wrote was refused. A malformed contract
    #: is this stage's usual failure and the count cannot say which rule it broke.
    analysis_rejected_tasks: tuple[dict[str, str], ...] = ()
    task_count: int = 0
    patch_count: int = 0
    skipped_task_count: int = 0
    #: Base commit each source repository was pinned to, keyed by absolute root.
    #: A task naming a repository already pinned to a different commit is skipped,
    #: so this records what the campaign actually built against.
    repository_pins: dict[str, str] = field(default_factory=dict)
    #: Validated forge-loop best results that could not be turned into a patch.
    #: Durable on purpose: Hyperloom discards this process's stdout and stderr
    #: when it hard-kills the controller on timeout, so a reason that lives only
    #: in the log is a reason nobody can read afterwards.
    recovery_failures: tuple[dict[str, str], ...] = ()
    #: What each operator's forge-loop spent on the model, one row per run that
    #: counted a call. The controller is the only place that sees both the spend
    #: and the operator it bought, and it runs out of process, so the totals are
    #: recorded here for Hyperloom to append to its LLM ledger afterwards.
    forge_llm_usage: tuple[dict[str, Any], ...] = ()
    schema_version: int = CONTROLLER_STATE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _recovery_failures(results: Iterable[RecoveryResult]) -> tuple[dict[str, str], ...]:
    """Keep the recoveries that found a validated best result and could not ship it.

    A task that produced nothing, and one whose patch was already published, are
    both ordinary outcomes. What deserves a record is a trusted best commit that
    never became a patch, because the export or the publication raised.
    """
    return tuple(
        {
            "operator_id": result.operator_id,
            "best_commit": result.best_commit,
            "reason": result.reason,
        }
        for result in results
        if not result.published and result.patch_dir is None and result.best_commit
    )


def _forge_llm_usage(results: Iterable[SingleTaskResult]) -> tuple[dict[str, Any], ...]:
    """Collect each forge-loop's token accounting, one row per operator.

    A campaign spends nearly all of its budget inside these subprocesses, so
    without this the per-attempt cost of a rewrite is invisible. A run that
    counted no call is dropped rather than reported as zero spend, which is the
    distinction ``UsageTotals.totals`` draws with its ``calls`` field.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        outcome = result.forge_outcome
        if result.task is None or outcome is None:
            continue
        usage = outcome.llm_usage
        if int(usage.get("calls") or 0) <= 0:
            continue
        rows.append(
            {
                "operator_id": result.task.operator_id,
                "model": outcome.agent_model,
                **usage,
            }
        )
    return tuple(rows)


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
        f"- **Skipped task count:** `{state.skipped_task_count}`",
        f"- **Analysis status:** `{state.analysis_status or 'not started'}`",
        f"- **Analysis reason:** {state.analysis_reason or 'none'}",
        f"- **Analysis published tasks:** `{state.analysis_published_task_count}`",
        f"- **Analysis rejected tasks:** `{state.analysis_rejected_task_count}`",
        "",
    ]
    if state.repository_pins:
        lines.extend(["## Pinned Source Repositories", ""])
        lines.extend(f"- `{root}` @ `{commit}`" for root, commit in sorted(state.repository_pins.items()))
        lines.append("")
    if state.analysis_rejected_tasks:
        lines.extend(["## Rejected Analysis Drafts", ""])
        lines.extend(
            f"- `{rejected.get('draft', '')}`: {rejected.get('reason', '')}"
            for rejected in state.analysis_rejected_tasks
        )
        lines.append("")
    if state.recovery_failures:
        lines.extend(["## Unpublishable Validated Results", ""])
        lines.extend(
            f"- `{failure.get('operator_id', '')}` @ `{failure.get('best_commit', '')}`: {failure.get('reason', '')}"
            for failure in state.recovery_failures
        )
        lines.append("")
    patches = published_operator_dirs(layout)
    if patches:
        lines.extend(["## Published Operator Patches", ""])
        for path in patches:
            try:
                metadata = json.loads((path / PUBLICATION_FILENAME).read_text(encoding="utf-8"))
                operator_id = str(metadata.get("operator_id") or path.name)
            except (OSError, json.JSONDecodeError, AttributeError):
                operator_id = path.name
            lines.append(f"- `{operator_id}`")
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


def _raise_controller_failure(
    layout: ControllerLayout,
    running: ControllerRunState,
    reason: str,
    error: Exception,
    progress: dict[str, Any] | None = None,
) -> None:
    failed = ControllerRunState(
        **{
            **running.to_dict(),
            **(progress or {}),
            "status": CONTROLLER_STATUS_FAILED,
            "finished_at": _now_iso(),
            "reason": reason,
        }
    )
    _write_state(layout, failed)
    _write_summary(layout, failed)
    raise ControllerRunError(reason) from error


def run_controller(
    *,
    handoff_dir: str | Path,
    budget_minutes: float,
    output_dir: str | Path,
) -> ControllerRunState:
    """Run one macro cycle's rewrite campaign against a fresh output directory.

    Reads the handoff, runs opportunity analysis, dispatches every task it
    published, reclaims validated forge-loop results into patches, and records a
    terminal status. An already-initialized output directory is refused rather
    than resumed, so each macro cycle brings its own.
    """
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

    progress: dict[str, Any] = {}

    def _publish_running(**updates: Any) -> None:
        """Persist what the campaign has accounted for so far, still as running.

        These fields used to land only in the terminal write, which is the one
        write a Hyperloom hard kill never reaches: it raises ``TimeoutExpired``
        without this process's streams, so anything recorded only at the end is
        unreadable afterwards -- and a timeout is the ordinary end of a long
        campaign, not an edge case. Publishing per milestone is what makes the
        spend, the pinned baselines and the skip reasons survive it.
        """
        progress.update(updates)
        snapshot = ControllerRunState(**{**running.to_dict(), **progress})
        _write_state(layout, snapshot)
        _write_summary(layout, snapshot)

    try:
        handoff = read_handoff(handoff_path)
    except Exception as error:
        _raise_controller_failure(
            layout,
            running,
            f"handoff validation failed: {error}",
            error,
        )

    try:
        # No recovery pass before analysis: the layout was just created, and an
        # already-initialized output directory is refused rather than resumed, so
        # there is never a prior task workspace here to reclaim.
        analysis = run_opportunity_analysis(
            handoff=handoff,
            layout=layout,
            controller_deadline_unix=running.deadline_unix,
        )
        _publish_running(
            analysis_status=analysis.status,
            analysis_reason=analysis.reason,
            analysis_published_task_count=analysis.published_task_count,
            analysis_rejected_task_count=analysis.rejected_task_count,
            analysis_rejected_tasks=analysis.rejected_tasks,
        )
        schedule = dispatch_prepared_tasks(
            layout,
            controller_deadline_unix=running.deadline_unix,
            on_progress=lambda partial: _publish_running(
                task_count=partial.task_count,
                patch_count=len(published_operator_dirs(layout)),
                skipped_task_count=partial.skipped_count,
                repository_pins=partial.repository_pins,
                forge_llm_usage=_forge_llm_usage(partial.results),
            ),
        )
        recovered = list(recover_all_task_results(layout))
        patch_count = len(published_operator_dirs(layout))
    except Exception as error:
        _raise_controller_failure(
            layout,
            running,
            f"controller execution failed: {error}",
            error,
            progress,
        )

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
            "analysis_rejected_tasks": analysis.rejected_tasks,
            "task_count": schedule.task_count,
            "patch_count": patch_count,
            "skipped_task_count": schedule.skipped_count,
            "repository_pins": schedule.repository_pins,
            "recovery_failures": _recovery_failures(recovered),
            "forge_llm_usage": _forge_llm_usage(schedule.results),
        }
    )
    _write_state(layout, completed)
    _write_summary(layout, completed)
    log.info(
        "kernel rewrite controller finished: status=%s tasks=%s patches=%s skipped=%s reason=%s",
        completed.status,
        completed.task_count,
        completed.patch_count,
        completed.skipped_task_count,
        completed.reason,
    )
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
