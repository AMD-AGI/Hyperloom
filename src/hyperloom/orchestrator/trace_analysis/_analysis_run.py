# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One tracelens_analysis run: its identity and output paths, the step ladder it records, and how it closes."""

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom_common.coerce import first_str
from hyperloom_common.io import JsonFormat, atomic_write_json
from hyperloom_common.timeutil import now_iso

from ._io_utils import append_log
from ._run_status import RunStatus
from .analysis_route import AnalysisRoute


class AnalysisRun(NamedTuple):
    """One analysis run: its identity, output paths, status file and what its steps record."""

    args: argparse.Namespace
    route: AnalysisRoute
    session_id: str
    run_id: str
    run_dir: Path
    log_path: Path
    status_path: Path
    artifacts: dict[str, str]
    run_status: RunStatus
    trace_health_warnings: list[dict[str, Any]]
    run_meta: dict[str, Any]

    def note_step(
        self,
        step_id: str,
        *,
        category: str,
        status: str,
        skip_reason: str | None = None,
        **detail: Any,
    ) -> None:
        """Append one step to the run's ladder.

        Args:
            step_id: Stable step name.
            category: Ladder phase (``preflight`` / ``install`` / ``split`` /
                ``select`` / ``analyze`` / ``emit``).
            status: ``ok`` / ``failed`` / ``skipped``.
            skip_reason: Why the step did not run, when skipped.
            **detail: Measured values for the step.
        """
        self.run_meta["steps"].append(
            {
                "step_id": step_id,
                "order": len(self.run_meta["steps"]) + 1,
                "category": category,
                "status": status,
                "skip_reason": skip_reason,
                "detail": detail,
            }
        )


def start_run(args: argparse.Namespace, route: AnalysisRoute) -> AnalysisRun:
    """Name this run and lay out its run directory, log and status file."""
    session_id = args.session_id or uuid.uuid4().hex[:12]
    run_id = f"tl-{uuid.uuid4().hex[:8]}"
    started_at = now_iso(timespec="auto")
    # Keep each TraceLens invocation's artifacts in its own run subdirectory.
    # now_iso renders UTC as "+00:00", never "Z", so the suffix is always added.
    ts_compact = started_at.replace("-", "").replace(":", "").split(".")[0] + "Z"
    sub_dir = f"{ts_compact}_{run_id}"
    root = Path(args.workspace_path) / "kernel-agent"
    run_dir = root / "runs" / session_id / sub_dir
    log_path = run_dir / "logs" / "tracelens_analysis" / f"{run_id}.log"
    status_path = run_dir / "status" / "tracelens_analysis" / f"{run_id}.json"
    artifacts: dict[str, str] = {}
    run_status = RunStatus(status_path, log_path, artifacts, run_id, started_at)
    # Structured trace-health findings surfaced to the Coordinator.
    trace_health_warnings: list[dict[str, Any]] = []
    # What this run saw and decided, returned to the caller instead of only
    # reaching the CLI log. The caller records it on the SBD V6 roofline event as
    # the run happens, so "which file was analysed, how was the steady-state
    # window picked, which gate refused" survives without re-reading the log.
    run_meta: dict[str, Any] = {
        "preflight": {},
        "split": {},
        "selection": {},
        "steps": [],
        "route_ext": {},
    }
    return AnalysisRun(
        args=args,
        route=route,
        session_id=session_id,
        run_id=run_id,
        run_dir=run_dir,
        log_path=log_path,
        status_path=status_path,
        artifacts=artifacts,
        run_status=run_status,
        trace_health_warnings=trace_health_warnings,
        run_meta=run_meta,
    )


def finish_run(
    run: AnalysisRun, trace_input_type: str, candidates: list[dict[str, Any]], orchestrator_mode: str
) -> None:
    """Record the emit step and session state, mark the run succeeded and print its result JSON."""
    artifacts = run.artifacts
    # Surface the contracted ``analysis.md`` exit path alongside hot_kernels.
    analysis_report_path = first_str(artifacts.get("tracelens_agent_report"), artifacts.get("trace_report_path"))

    result = {
        "tool": "tracelens_analysis",
        "session_id": run.session_id,
        "run_id": run.run_id,
        "trace_input_type": trace_input_type,
        "hot_kernels": candidates,
        "trace_report_path": artifacts["trace_report_path"],
        "analysis_report_path": analysis_report_path,
        "cli_log_path": str(run.log_path),
        "status_path": str(run.status_path),
        "artifact_paths": artifacts,
        "orchestrator_mode": orchestrator_mode,
        # Trace-quality findings surfaced to the Coordinator (empty = nothing wrong).
        "trace_health_warnings": run.trace_health_warnings,
        # What this run saw and decided, for the caller's timeline event.
        "run_meta": run.run_meta,
    }
    run.note_step(
        "extract_hot_kernels",
        category="emit",
        status="ok",
        hot_kernel_count=len(candidates),
        orchestrator_mode=orchestrator_mode,
    )
    atomic_write_json(
        run.run_dir / "session_state.json",
        {
            "session_id": run.session_id,
            "last_tool": "tracelens_analysis",
            "last_run_id": run.run_id,
            "updated_at": now_iso(timespec="auto"),
            "model_name": run.args.model_name,
            "framework": run.args.framework,
        },
        fmt=JsonFormat(trailing_newline=True),
    )
    run.run_status.write(
        state="succeeded",
        current_step="done",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def report_failure(run: AnalysisRun, exc: Exception) -> None:
    """Log the failure, mark the run failed and print the failure JSON."""
    append_log(run.log_path, f"[error] {type(exc).__name__}: {exc}")
    run.run_status.write(
        state="failed",
        current_step="failed",
        error=f"{type(exc).__name__}: {exc}",
    )
    # Include trace_health_warnings accumulated pre-exception for auto-recovery.
    # ``run_meta`` goes out on this path too: the ladder up to the failure is
    # what says which rung refused, so dropping it here would leave the
    # failure cases -- the ones worth explaining -- as the only blind spot.
    print(
        json.dumps(
            {
                "tool": "tracelens_analysis",
                "session_id": run.session_id,
                "run_id": run.run_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "cli_log_path": str(run.log_path),
                "status_path": str(run.status_path),
                "trace_health_warnings": run.trace_health_warnings,
                "run_meta": run.run_meta,
            },
            indent=2,
            sort_keys=True,
        )
    )
