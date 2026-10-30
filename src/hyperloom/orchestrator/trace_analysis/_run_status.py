# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What one trace-analysis run records: its status file and the ``run_meta`` step ladder it returns."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.common.io import JsonFormat, atomic_write_json
from hyperloom.common.timeutil import now_iso

from ._io_utils import read_last_lines


@dataclass(frozen=True)
class RunStatus:
    """The status file one tracelens_analysis run keeps current.

    Attributes:
        status_path: Destination status JSON file.
        log_path: Log file whose size/tail are recorded.
        artifact_paths: Map of artifact names to paths, read at each write.
        run_id: Unique identifier for this run.
        started_at: ISO-8601 start time used to compute duration.
    """

    status_path: Path
    log_path: Path
    artifact_paths: dict[str, str]
    run_id: str
    started_at: str

    def write(self, *, state: str, current_step: str, error: str | None = None) -> None:
        """Atomically write the run-status JSON file.

        Captures the current run state, recent log tail, and (on terminal
        states) the wall-clock duration so downstream collectors can build a
        timeline event.

        Args:
            state (str): Current run state (e.g. ``running``, ``succeeded``).
            current_step (str): Human-readable label of the active step.
            error (str | None): Error message recorded when the run failed.
        """
        updated_at = now_iso(timespec="auto")
        payload: dict[str, Any] = {
            "tool": "tracelens_analysis",
            "run_id": self.run_id,
            "state": state,
            "current_step": current_step,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "updated_at": updated_at,
            "log_path": str(self.log_path),
            "artifact_paths": self.artifact_paths,
            "offset_bytes": self.log_path.stat().st_size if self.log_path.exists() else 0,
            "last_lines": read_last_lines(self.log_path),
        }
        # Emit ended_at + duration_seconds on terminal states for the timeline.
        if state in ("succeeded", "failed", "aborted", "cancelled"):
            payload["ended_at"] = updated_at
            try:
                start_dt = datetime.fromisoformat(self.started_at)
                end_dt = datetime.fromisoformat(updated_at)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                payload["duration_seconds"] = max(
                    0.0,
                    (end_dt - start_dt).total_seconds(),
                )
            except (ValueError, TypeError):
                payload["duration_seconds"] = None
        if error:
            payload["error"] = error
        atomic_write_json(self.status_path, payload, fmt=JsonFormat(trailing_newline=True))


def new_run_meta() -> dict[str, Any]:
    """The ``run_meta`` block both trace tools return: what the run saw and decided, and its step ladder.

    The caller records it on the SBD V6 roofline event, so both routes share this one shape.
    """
    return {"preflight": {}, "split": {}, "selection": {}, "steps": [], "route_ext": {}}


def note_step(
    run_meta: dict[str, Any],
    step_id: str,
    *,
    category: str,
    status: str,
    **detail: Any,
) -> None:
    """Append one step to the run's ladder.

    Args:
        run_meta: The run's ``run_meta`` block.
        step_id: Stable step name.
        category: Ladder phase (``preflight`` / ``install`` / ``split`` /
            ``select`` / ``analyze`` / ``emit``).
        status: ``ok`` / ``failed`` / ``skipped``.
        **detail: Measured values for the step; ``skip_reason`` names why a skipped step did not run.
    """
    skip_reason = detail.pop("skip_reason", None)
    run_meta["steps"].append(
        {
            "step_id": step_id,
            "order": len(run_meta["steps"]) + 1,
            "category": category,
            "status": status,
            "skip_reason": skip_reason,
            "detail": detail,
        }
    )
