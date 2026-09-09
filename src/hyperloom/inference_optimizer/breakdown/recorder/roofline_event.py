# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``roofline`` action: recorded the same way wherever it belongs."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .event_fields import (
    analysis_detail as _analysis_detail,
    as_dict as _as_dict,
    as_list as _as_list,
    clip as _clip,
    failure_row as _failure_row,
    int_or_none as _int_or_none,
    now_iso_seconds as _now_iso,
    worst_status as _worst_status,
)
from .event_ids import event_id
from .event_rows import group_rows, rows_for_event, sort_rows, wire_rows
from .event_sink import RecordSink
from .event_timeline import finish_event, open_event

log = logging.getLogger(__name__)

EVENT_TYPE = "roofline"
EVENT_KIND = "roofline"

#: The component segment of a standalone roofline event id. The phase segment
#: is the phase that dispatched it, which is why it is a parameter.
EVENT_COMPONENT = "roofline"

PRODUCER = "orchestrator"

#: The event-level section, holding one fragment per event rather than per
#: action. It is separate from :data:`SECTION_ACTION` because the two are
#: counted differently: an event has one timeline sequence and, when a phase
#: dispatched roofline twice in a cycle, several actions -- so a section serving
#: both would put a row with no action in it among the actions.
SECTION_EVENT = "roofline_event"

SECTION_ACTION = "roofline_action"
SECTION_PROFILE_RUN = "roofline_profile_run"
SECTION_ANALYSIS_RUN = "roofline_analysis_run"

ROW_ACTION = "action"
ROW_PROFILE_RUN = "profile_run"
ROW_ANALYSIS_RUN = "analysis_run"

# ``trace_files`` reaches 424 entries on multi-rank xDiT runs (p99 424, p50 2), which would be ~85 KiB of paths per
# profile run.
_MAX_SAMPLE_TRACE_FILES = 4

# Trace-structure issues are prose written for an operator; a handful is enough to characterize a degraded trace and
# the count carries the rest.
_MAX_TRACE_ISSUES = 8

# Every profile run row names why it ran, so a multi-attempt roofline can be read without re-deriving the retry reason
# from log text.
PROFILE_ATTEMPT_INITIAL = "initial"
PROFILE_ATTEMPT_AFTER_EXCEPTION = "retry_after_exception"
PROFILE_ATTEMPT_AFTER_BAD_RETURN = "retry_after_bad_return"
PROFILE_ATTEMPT_AFTER_FAILURE = "retry_after_failure"
PROFILE_ATTEMPT_AFTER_NO_TRACE = "retry_after_no_trace"
PROFILE_ATTEMPT_AFTER_CAPTURE_ONLY = "retry_after_capture_only"
PROFILE_ATTEMPT_AFTER_ZERO_OPS = "retry_after_zero_ops"
PROFILE_ATTEMPT_COMPUTE_BOUND = "compute_bound_reprofile"

ANALYSIS_ATTEMPT_INITIAL = "initial"
ANALYSIS_ATTEMPT_N26_RETRY = "n26_steady_state_retry"
ANALYSIS_ATTEMPT_COMPUTE_BOUND = "compute_bound_reprofile"

# The two halves of a roofline action, named because ``failed_substep`` reports which one failed and the crash path
# has only the in-flight half to go on.
SUBSTEP_PROFILE = "profile"
SUBSTEP_ANALYSIS = "analysis"
# The executor names its failure exits after the step that produced them (``profile_no_trace``, ``trace_analyze``,
# ...), which is finer than the two halves above; these are the spellings that mean the analysis half.
_ANALYSIS_PHASES = frozenset({SUBSTEP_ANALYSIS, "trace_analyze"})

__all__ = [
    "ANALYSIS_ATTEMPT_COMPUTE_BOUND",
    "ANALYSIS_ATTEMPT_INITIAL",
    "ANALYSIS_ATTEMPT_N26_RETRY",
    "EVENT_COMPONENT",
    "EVENT_KIND",
    "EVENT_TYPE",
    "PRODUCER",
    "PROFILE_ATTEMPT_AFTER_BAD_RETURN",
    "PROFILE_ATTEMPT_AFTER_CAPTURE_ONLY",
    "PROFILE_ATTEMPT_AFTER_EXCEPTION",
    "PROFILE_ATTEMPT_AFTER_FAILURE",
    "PROFILE_ATTEMPT_AFTER_NO_TRACE",
    "PROFILE_ATTEMPT_AFTER_ZERO_OPS",
    "PROFILE_ATTEMPT_COMPUTE_BOUND",
    "PROFILE_ATTEMPT_INITIAL",
    "SECTION_ACTION",
    "SECTION_ANALYSIS_RUN",
    "SECTION_EVENT",
    "SECTION_PROFILE_RUN",
    "SUBSTEP_ANALYSIS",
    "SUBSTEP_PROFILE",
    "RooflineEventRecorder",
    "assemble_roofline_action",
    "assemble_roofline_ext",
    "make_roofline_recorder",
    "roofline_event_id",
]


def roofline_event_id(phase: str, macro_cycle: Any) -> str:
    """Build the event id of the rooflines one phase dispatched in one cycle."""
    return event_id(phase, macro_cycle, EVENT_COMPONENT)


def _rank_of(path: str) -> str:
    """Extract the rank token from a per-rank trace filename."""
    name = Path(str(path)).name
    for token in name.replace("-", "_").split("_"):
        if token.startswith("rank") and token[4:].isdigit():
            return token[4:]
    return "unknown"


def _summarize_trace_files(profile_result: dict[str, Any]) -> dict[str, Any]:
    """Summarize the profile's trace file set without carrying every path."""
    files = [str(row) for row in _as_list(profile_result.get("trace_files")) if row]
    by_rank: dict[str, int] = {}
    for path in files:
        rank = _rank_of(path)
        by_rank[rank] = by_rank.get(rank, 0) + 1
    return {
        "main_path": str(profile_result.get("main_trace_path") or ""),
        "trace_dir": str(profile_result.get("trace_dir") or ""),
        "file_count": len(files),
        "rank_count": len([rank for rank in by_rank if rank != "unknown"]),
        "files_by_rank": by_rank,
        "sample_files": files[:_MAX_SAMPLE_TRACE_FILES],
        "selection_reason": str(profile_result.get("profile_trace_selection_reason") or ""),
    }


def _summarize_trace_health(profile_result: dict[str, Any]) -> dict[str, Any]:
    """Project ``trace_health`` into the action's bounded health block."""
    health = _as_dict(profile_result.get("trace_health"))
    issues = [_clip(row) for row in _as_list(health.get("issues"))]
    return {
        "zero_ops": bool(health.get("zero_ops")),
        "capture_traces_present": bool(health.get("capture_traces_present")),
        "per_kernel_attribution_degraded": bool(health.get("per_kernel_attribution_degraded")),
        "issue_count": len(issues),
        "issues": issues[:_MAX_TRACE_ISSUES],
    }


def _summarize_validate(profile_result: dict[str, Any]) -> dict[str, Any]:
    """Project the structured profile-trace validation into the run row."""
    validate = _as_dict(profile_result.get("trace_validate"))
    if not validate:
        return {}
    checks = [_as_dict(row) for row in _as_list(validate.get("checks")) if isinstance(row, dict)]
    verdict = _as_dict(validate.get("verdict"))
    usable_by = [str(name) for name in _as_list(verdict.get("usable_by"))]
    return {
        # Carried as two independent axes.
        "usable_by": usable_by,
        "decode_conclusions_valid": verdict.get("decode_conclusions_valid"),
        "silently_wrong": verdict.get("silently_wrong"),
        "blocking_reasons": [_clip(row) for row in _as_list(verdict.get("blocking_reasons"))],
        "warnings": [_clip(row) for row in _as_list(verdict.get("warnings"))],
        "recommended_steady_state_mode": verdict.get("recommended_steady_state_mode"),
        "modes_that_would_fail": verdict.get("modes_that_would_fail"),
        "steady_state_forecast": _as_dict(validate.get("steady_state_forecast")),
        "hot_kernel_list_would_be_suppressed": verdict.get("hot_kernel_list_would_be_suppressed"),
        "thresholds_effective": _as_dict(verdict.get("thresholds_effective")),
        "probe_status": str(validate.get("probe_status") or ""),
        "probe_error": _clip(validate.get("probe_error") or ""),
        "checked_at": str(validate.get("checked_at") or ""),
        "failed_check_ids": [str(row.get("check_id") or "") for row in checks if row.get("status") == "failed"],
        "checks": checks,
    }


class RooflineEventRecorder:
    """Records one roofline action's facts into whichever event owns it."""

    def __init__(
        self,
        sink: RecordSink,
        *,
        task_id: str = "",
        task_kind: str = "",
        reason: str = "",
        framework: str = "",
        params: dict[str, Any] | None = None,
        owns_event: bool = True,
    ):
        """Bind a recorder to one action inside one event."""
        self._sink = sink
        self._t0 = time.monotonic()
        self._start_time = _now_iso()
        self._owns_event = bool(owns_event)
        self._sequence: int | None = None
        self._closed = False
        self._substep = SUBSTEP_PROFILE
        params = _as_dict(params)
        # ``arm`` names the configuration the run measured, which only a roofline dispatch does.
        kind = str(task_kind or "")
        arm = (
            ""
            if kind not in ("", EVENT_TYPE)
            else ("baseline" if str(reason or "") == "prelude_initial" else "current_best")
        )
        self._task_id = str(task_id or "")
        self._action_id = self._task_id or "unnamed"
        self._sink.record(
            SECTION_ACTION,
            {
                "task_id": self._task_id,
                "start_time": self._start_time,
                "in_flight_substep": self._substep,
                "request": {
                    "task_id": self._task_id,
                    "task_kind": kind,
                    "reason": str(reason or ""),
                    "arm": arm,
                    "framework": str(framework or ""),
                    "workspace_path": str(params.get("workspace_path") or ""),
                },
            },
            row_type=ROW_ACTION,
            natural_ids=self._action_id,
        )

    @property
    def event_id(self) -> str:
        """str: The event this action's rows belong to."""
        return self._sink.event_id

    @property
    def task_id(self) -> str:
        """str: The task id separating this action from others in its event."""
        return self._task_id

    def _record_action(self, payload: Mapping[str, Any]) -> None:
        """Update this action's own row."""
        self._sink.record(SECTION_ACTION, payload, row_type=ROW_ACTION, natural_ids=self._action_id)

    # ---- lifecycle -------------------------------------------------------

    def begin(self, *, max_profile_attempts: int) -> None:
        """Record the retry budget and, when this action owns the event, open it."""
        self._record_action({"max_profile_attempts": int(max_profile_attempts)})
        if not self._owns_event:
            return
        self._sequence = open_event(
            event_type=EVENT_TYPE,
            event=self.event_id,
            event_section=SECTION_EVENT,
            producer=PRODUCER,
            kind=EVENT_KIND,
            start_time=self._start_time,
            ext={"in_flight_substep": self._substep},
        )

    def record_profile_run(
        self,
        *,
        run_index: int,
        attempt_reason: str,
        status: str,
        started_at: str,
        duration_sec: float | None,
        disable_cuda_graph: bool,
        profile_result: dict[str, Any] | None = None,
        failure: dict[str, Any] | None = None,
    ) -> None:
        """Record one profile attempt."""
        result = _as_dict(profile_result)
        self._sink.record(
            SECTION_PROFILE_RUN,
            {
                "task_id": self._task_id,
                "run_index": int(run_index),
                "effective": False,
                "attempt_reason": str(attempt_reason),
                "status": str(status),
                "start_time": str(started_at or ""),
                "end_time": _now_iso(),
                "duration_sec": duration_sec,
                "disable_cuda_graph": bool(disable_cuda_graph),
                "failure": failure,
                "validate": _summarize_validate(result),
            },
            row_type=ROW_PROFILE_RUN,
            natural_ids=(self._action_id, str(int(run_index))),
        )
        if disable_cuda_graph:
            self._record_action({"eager_fallback_applied": True})

    def adopt_profile_run(
        self,
        *,
        run_index: int,
        profile_result: dict[str, Any] | None,
        recovered: bool = False,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Mark one profile attempt as the one the action carried forward."""
        result = _as_dict(profile_result)
        self._substep = SUBSTEP_ANALYSIS
        self._record_action(
            {
                "profile_effective_run_index": int(run_index),
                "recovered": bool(recovered),
                "in_flight_substep": self._substep,
                "profile_effective_run": {
                    "run_index": int(run_index),
                    "status": str(result.get("status") or ""),
                    "framework": str(result.get("framework") or ""),
                    "model": str(result.get("model") or ""),
                    "workspace": str(result.get("workspace") or ""),
                    "report_path": str(result.get("report_path") or ""),
                    "trace": _summarize_trace_files(result),
                    "trace_health": _summarize_trace_health(result),
                    "framework_rewrite_candidate_count": _int_or_none(result.get("framework_rewrite_candidate_count")),
                    "params": {
                        key: params[key]
                        for key in ("workspace_path", "reason", "framework", "num_prompts", "request_rate")
                        if isinstance(params, dict) and key in params
                    },
                },
            }
        )

    def record_analysis_run(
        self,
        *,
        run_index: int,
        attempt_reason: str,
        status: str,
        started_at: str,
        duration_sec: float | None,
        trace_input: str,
        requested_steady_state_mode: str = "",
        ta_result: dict[str, Any] | None = None,
        failure: dict[str, Any] | None = None,
    ) -> None:
        """Record one trace-analysis attempt."""
        result = _as_dict(ta_result)
        meta = _as_dict(result.get("analysis_meta"))
        self._sink.record(
            SECTION_ANALYSIS_RUN,
            {
                "task_id": self._task_id,
                "run_index": int(run_index),
                "effective": False,
                "attempt_reason": str(attempt_reason),
                "status": str(status),
                "start_time": str(started_at or ""),
                "end_time": _now_iso(),
                "duration_sec": duration_sec,
                "route": str(meta.get("route") or ""),
                "tool": str(meta.get("tool") or ""),
                "requested_steady_state_mode": str(requested_steady_state_mode or meta.get("steady_state_mode") or ""),
                "trace_input": str(trace_input or ""),
                "hot_kernel_count": len(
                    [row for row in _as_list(result.get("hot_kernels_top15") or result.get("hot_kernels")) if row]
                ),
                "failure": failure,
            },
            row_type=ROW_ANALYSIS_RUN,
            natural_ids=(self._action_id, str(int(run_index))),
        )

    def adopt_analysis_run(
        self,
        *,
        run_index: int,
        ta_result: dict[str, Any] | None,
        trace_input: str,
    ) -> None:
        """Mark one analysis attempt as the one the action concluded from."""
        result = _as_dict(ta_result)
        payload: dict[str, Any] = {
            "analysis_effective_run_index": int(run_index),
            "analysis_effective_run": {
                "run_index": int(run_index),
                "trace_input": str(trace_input or ""),
                "orchestrator_mode": str(result.get("orchestrator_mode") or ""),
                "orchestrator_error": _clip(result.get("orchestrator_error")),
                **_analysis_detail(result),
            },
        }
        n26 = _as_dict(result.get("n26_auto_retry"))
        if n26:
            payload["n26_auto_retry"] = n26
        self._record_action(payload)

    def record_compute_bound_reprofile(self, *, attempted: bool, adopted: bool, reason: str = "") -> None:
        """Record the multi-node compute-bound re-profile decision."""
        self._record_action(
            {
                "compute_bound_reprofile": {
                    "attempted": bool(attempted),
                    "adopted": bool(adopted),
                    "reason": _clip(reason),
                }
            }
        )

    def finish_succeeded(
        self,
        *,
        snapshot_id: Any,
        hot_kernel_count: int,
        kernel_attribution_degraded: bool,
        cached: dict[str, Any] | None,
        trace_path: str,
    ) -> None:
        """Close the action as succeeded and record the promoted artifacts."""
        promoted = _as_dict(cached)
        # Zero routable candidates is a completed roofline that cannot advance kernel work, which is a different
        # operational state from a clean run.
        self._close(
            status="degraded" if kernel_attribution_degraded else "succeeded",
            payload={
                "outcome": {
                    "snapshot_id": _int_or_none(promoted.get("roofline_snapshot_id") or snapshot_id),
                    "hot_kernel_count": int(hot_kernel_count),
                    "kernel_attribution_degraded": bool(kernel_attribution_degraded),
                    "profile_trace": str(trace_path or ""),
                    "steady_state_trace": str(promoted.get("steady_state_trace") or ""),
                    "analysis_md_path": str(promoted.get("analysis_md_path") or ""),
                    "candidates_path": str(promoted.get("candidates_path") or ""),
                    "kernel_roofline_path": str(promoted.get("kernel_roofline_path") or ""),
                }
            },
        )

    def finish_failed(self, *, phase: str, error_class: str = "", message: Any = "") -> None:
        """Close the action as failed, naming the sub-step that failed."""
        # Matched against the analysis spellings rather than prefixed, because the crash path passes the in-flight
        # substep itself: ``"analysis"`` does not start with ``"trace_analyze"``, so a crash after the profile had
        # already been adopted was reported as a profile failure.
        named = str(phase or "")
        analysis = named in _ANALYSIS_PHASES or named.startswith("trace_analyze")
        self._close(
            status="failed",
            payload={
                "failed_substep": SUBSTEP_ANALYSIS if analysis else SUBSTEP_PROFILE,
                "failure": _failure_row(
                    phase=phase,
                    error_class=error_class or f"{phase}_failed",
                    message=message,
                ),
            },
        )

    def finish_crashed(self, exc: BaseException) -> None:
        """Close an action whose executor raised instead of returning a result."""
        if self._closed:
            return
        self.finish_failed(
            phase=self._substep or EVENT_TYPE,
            error_class=type(exc).__name__,
            message=f"roofline action raised: {exc!r}",
        )

    def _close(self, *, status: str, payload: Mapping[str, Any]) -> None:
        """Record the action's terminal facts and, when it owns the event, close it."""
        if self._closed:
            return
        self._closed = True
        end_time = _now_iso()
        self._record_action(
            {
                **payload,
                "status": str(status),
                "in_flight_substep": None,
                "end_time": end_time,
                "duration_sec": round(time.monotonic() - self._t0, 3),
            }
        )
        if not self._owns_event:
            return
        from .assembler import roofline_event_parts

        ext, derived = assemble_roofline_ext(roofline_event_parts(), event=self.event_id)
        finish_event(
            event_type=EVENT_TYPE,
            event=self.event_id,
            sequence=self._sequence,
            status=derived or status,
            ext=ext,
            kind=EVENT_KIND,
            start_time=self._start_time,
            end_time=end_time,
        )


def assemble_roofline_action(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
    task_id: str,
) -> dict[str, Any] | None:
    """Assemble one roofline action out of its recorded rows."""
    actions = assemble_roofline_actions(parts, event=event)
    wanted = str(task_id or "")
    for action in actions:
        if str(action.get("task_id") or "") == wanted:
            return action
    return None


def assemble_roofline_actions(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> list[dict[str, Any]]:
    """Assemble every roofline action belonging to one event."""
    action_rows = sort_rows(
        rows_for_event(parts.get(SECTION_ACTION) or [], event),
        keys=("start_time", "task_id"),
    )
    profiles = group_rows(
        sort_rows(rows_for_event(parts.get(SECTION_PROFILE_RUN) or [], event), keys=("run_index",)),
        "task_id",
    )
    analyses = group_rows(
        sort_rows(rows_for_event(parts.get(SECTION_ANALYSIS_RUN) or [], event), keys=("run_index",)),
        "task_id",
    )

    actions: list[dict[str, Any]] = []
    for row in action_rows:
        task = str(row.get("task_id") or "")
        profile_index = _int_or_none(row.get("profile_effective_run_index"))
        analysis_index = _int_or_none(row.get("analysis_effective_run_index"))
        profile_runs = _mark_effective(wire_rows(profiles.get(task, []), drop=("event_id", "task_id")), profile_index)
        analysis_runs = _mark_effective(wire_rows(analyses.get(task, []), drop=("event_id", "task_id")), analysis_index)
        actions.append(
            {
                "task_id": task,
                "status": str(row.get("status") or "running"),
                "start_time": str(row.get("start_time") or ""),
                "end_time": str(row.get("end_time") or ""),
                "duration_sec": row.get("duration_sec"),
                "in_flight_substep": row.get("in_flight_substep"),
                "failed_substep": row.get("failed_substep"),
                "request": _as_dict(row.get("request")),
                "profile": {
                    "attempt_count": len(profile_runs),
                    "max_attempts": _int_or_none(row.get("max_profile_attempts")) or 0,
                    "effective_run_index": profile_index,
                    "recovered": bool(row.get("recovered")),
                    "eager_fallback_applied": bool(row.get("eager_fallback_applied")),
                    "runs": profile_runs,
                    "effective_run": _as_dict(row.get("profile_effective_run")),
                },
                "analysis": {
                    "attempt_count": len(analysis_runs),
                    "effective_run_index": analysis_index,
                    "n26_auto_retry": row.get("n26_auto_retry"),
                    "compute_bound_reprofile": _as_dict(row.get("compute_bound_reprofile"))
                    or {"attempted": False, "adopted": False, "reason": ""},
                    "runs": analysis_runs,
                    "effective_run": _as_dict(row.get("analysis_effective_run")),
                },
                "outcome": _as_dict(row.get("outcome")),
                "failure": _as_dict(row.get("failure")) or None,
            }
        )
    return actions


def assemble_roofline_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble one roofline event's ``ext`` out of its recorded rows."""
    actions = assemble_roofline_actions(parts, event=event)
    return {"actions": actions}, _worst_status([str(action.get("status") or "") for action in actions])


def _mark_effective(runs: list[dict[str, Any]], effective_index: int | None) -> list[dict[str, Any]]:
    """Stamp ``effective`` onto the one run the action adopted."""
    for run in runs:
        run["effective"] = effective_index is not None and _int_or_none(run.get("run_index")) == effective_index
    return runs


def make_roofline_recorder(
    sink: RecordSink | None,
    *,
    task_id: str = "",
    task_kind: str = "",
    reason: str = "",
    framework: str = "",
    params: dict[str, Any] | None = None,
    owns_event: bool = True,
) -> RooflineEventRecorder | None:
    """Build a recorder, or ``None`` when one cannot be constructed."""
    if sink is None:
        return None
    try:
        return RooflineEventRecorder(
            sink,
            task_id=task_id,
            task_kind=task_kind,
            reason=reason,
            framework=framework,
            params=params,
            owns_event=owns_event,
        )
    except Exception:  # noqa: BLE001 — observability cannot change roofline behavior
        log.warning(
            "roofline timeline: recorder construction failed; this action's facts will be missing from the event",
            exc_info=True,
        )
        return None
