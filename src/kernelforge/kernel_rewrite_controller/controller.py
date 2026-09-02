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
from kernelforge.kernel_rewrite_controller.paths import ControllerLayout

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
        "",
    ]
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
        read_handoff(handoff_path)
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

    completed = ControllerRunState(
        **{
            **running.to_dict(),
            "status": CONTROLLER_STATUS_NO_OPPORTUNITY,
            "finished_at": _now_iso(),
            "reason": "no analysis tasks are available in the controller skeleton",
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
