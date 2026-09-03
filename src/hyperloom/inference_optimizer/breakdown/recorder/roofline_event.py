# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``roofline`` action: recorded the same way wherever it belongs.

Roofline is the executor two callers reach for. A phase with no timeline event
of its own dispatches it and it leaves an event behind; the KERNEL entry calls
it inline as its re-profile, and that run belongs to the kernel event, because
a sibling event would claim an autonomy the inline run does not have.

The difference between those two modes is one string. The recorder writes the
same three sections either way, tagged with whichever event id its sink was
built for, and assembly decides where the rows land: under ``actions[]`` of a
roofline event, or under ``forge.reprofile.run`` of the kernel event that
called it. Teaching the executor to ask "am I inline right now" would put every
caller in its signature; lifting the event id out of it costs one parameter.

The action id is a second axis the recorder needs and the kernel event does
not. One event id is ``{phase}:{macro_cycle}:roofline``, and a phase can
dispatch roofline more than once in a cycle -- so the event holds an array of
actions and the task id is what separates them. That is also why it is a
fragment key rather than a segment of the event id: it names a row within an
event, not an event.

Multiplicity inside one action stays inside it. Every retry the executor
performs -- the profile attempt loop, the N26 steady-state re-analysis, the
multi-node compute-bound re-profile -- is a row, and the row the action
actually carried forward is flagged ``effective``. An action that re-profiled
three times is one action with three profile runs, not three actions.
"""

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

# ``trace_files`` reaches 424 entries on multi-rank xDiT runs (p99 424, p50 2),
# which would be ~85 KiB of paths per profile run. The rank histogram plus a few
# samples answers the same questions (is the file count plausible, did every rank
# report) at a bounded cost.
_MAX_SAMPLE_TRACE_FILES = 4

# Trace-structure issues are prose written for an operator; a handful is enough
# to characterize a degraded trace and the count carries the rest.
_MAX_TRACE_ISSUES = 8

# Every profile run row names why it ran, so a multi-attempt roofline can be read
# without re-deriving the retry reason from log text.
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
    "RooflineEventRecorder",
    "assemble_roofline_action",
    "assemble_roofline_ext",
    "make_roofline_recorder",
    "roofline_event_id",
]


def roofline_event_id(phase: str, macro_cycle: Any) -> str:
    """Build the event id of the rooflines one phase dispatched in one cycle.

    Args:
        phase (str): The coordinator phase that dispatched the action.
        macro_cycle (Any): The macro cycle it was dispatched in.

    Returns:
        str: The event id, ``{phase}:{macro_cycle}:roofline``.

    Raises:
        ValueError: If either segment is malformed.
    """
    return event_id(phase, macro_cycle, EVENT_COMPONENT)


def _rank_of(path: str) -> str:
    """Extract the rank token from a per-rank trace filename.

    xDiT tensor/sequence-parallel profiles write one trace per rank, named with
    a ``rank<N>`` / ``_<N>.pt.trace.json.gz`` suffix. Grouping by rank turns a
    424-entry path list into a histogram that shows whether every rank reported.

    Args:
        path (str): A trace file path.

    Returns:
        str: The rank token, or ``"unknown"`` when no rank is encoded.
    """
    name = Path(str(path)).name
    for token in name.replace("-", "_").split("_"):
        if token.startswith("rank") and token[4:].isdigit():
            return token[4:]
    return "unknown"


def _summarize_trace_files(profile_result: dict[str, Any]) -> dict[str, Any]:
    """Summarize the profile's trace file set without carrying every path.

    Args:
        profile_result (dict[str, Any]): The profile sub-step result dict.

    Returns:
        dict[str, Any]: The resolved main path, the file count, a per-rank
            histogram, a bounded sample of paths, and the selection reason.
    """
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
    """Project ``trace_health`` into the action's bounded health block.

    Carries the three booleans the executor branches on, the structured
    per-check rows the profile validator emits, and a clipped slice of the
    operator-facing issue prose.

    Args:
        profile_result (dict[str, Any]): The profile sub-step result dict.

    Returns:
        dict[str, Any]: The health summary.
    """
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
    """Project the structured profile-trace validation into the run row.

    The validator runs per profile attempt, so its verdict is stored on the run
    row rather than on the effective-run summary: "attempt 1 recorded no graph
    launches, attempt 2 did" is only answerable when each attempt keeps the
    verdict computed against the trace it produced.

    Args:
        profile_result (dict[str, Any]): The profile sub-step result dict.

    Returns:
        dict[str, Any]: The validation block, empty when the validator did not
            run.
    """
    validate = _as_dict(profile_result.get("trace_validate"))
    if not validate:
        return {}
    checks = [_as_dict(row) for row in _as_list(validate.get("checks")) if isinstance(row, dict)]
    verdict = _as_dict(validate.get("verdict"))
    usable_by = [str(name) for name in _as_list(verdict.get("usable_by"))]
    return {
        # Carried as two independent axes. A trace can be routable and still
        # yield a false decode conclusion, so a single grade cannot say which of
        # the two a given attempt failed -- and the silently-wrong case is
        # exactly the one where the analysis reports no trouble at all.
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
    """Records one roofline action's facts into whichever event owns it.

    The recorder holds a sink and a task id and nothing else. It never learns
    whether it is writing into its own event or into an enclosing one, which is
    the point: :meth:`close` writes a timeline entry only when the action owns
    the event, and that is a property of how the recorder was built.
    """

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
        """Bind a recorder to one action inside one event.

        Args:
            sink (RecordSink): Where the rows go, which decides the event they
                belong to.
            task_id (str): The dispatched task id, which separates this action
                from the others in the same event.
            task_kind (str): The dispatched task's kind. The executor is also
                reused by the GEMM shape-capture path, which is not a roofline
                dispatch, so the action states which one it was rather than
                leaving a consumer to infer it from an absent reason.
            reason (str): Dispatch reason (``prelude_initial`` /
                ``close_post_opt`` / an LLM-proposed reason), which also selects
                the profiled arm.
            framework (str): Resolved serving framework.
            params (dict[str, Any] | None): The task params, read for workspace
                and arm context.
            owns_event (bool): Whether this action is the event. An action
                called inline by a phase that owns a timeline event is not, so
                it records rows and writes no entry of its own.
        """
        self._sink = sink
        self._t0 = time.monotonic()
        self._start_time = _now_iso()
        self._owns_event = bool(owns_event)
        self._sequence: int | None = None
        self._closed = False
        self._substep = "profile"
        params = _as_dict(params)
        # ``arm`` names the configuration the run measured, which only a roofline
        # dispatch does. A roofline task may legitimately carry no reason, so the
        # claim is withheld by kind rather than by an empty reason -- otherwise
        # shape capture, which reuses this executor with no reason of its own,
        # would be recorded as having measured current_best.
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
        """Update this action's own row.

        Args:
            payload (Mapping[str, Any]): The fields this call knows.
        """
        self._sink.record(SECTION_ACTION, payload, row_type=ROW_ACTION, natural_ids=self._action_id)

    # ---- lifecycle -------------------------------------------------------

    def begin(self, *, max_profile_attempts: int) -> None:
        """Record the retry budget and, when this action owns the event, open it.

        Args:
            max_profile_attempts (int): The profile retry budget, recorded so a
                run that exhausted it is distinguishable from one that stopped
                early.
        """
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
        """Record one profile attempt.

        Args:
            run_index (int): 1-based attempt index within this action.
            attempt_reason (str): Why this attempt ran (a ``PROFILE_ATTEMPT_*``
                value).
            status (str): ``succeeded`` / ``recovered`` / ``failed``.
            started_at (str): ISO start of the attempt.
            duration_sec (float | None): Wall-clock seconds the attempt took.
            disable_cuda_graph (bool): Whether the attempt booted eager.
            profile_result (dict[str, Any] | None): The attempt's result dict,
                when it returned one.
            failure (dict[str, Any] | None): Failure row for a failed attempt.
        """
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
        """Mark one profile attempt as the one the action carried forward.

        Which attempt was adopted is recorded on the action rather than stamped
        onto the runs, so the flag cannot end up on two of them; assembly reads
        it back onto the row it names.

        Args:
            run_index (int): The adopted attempt's index.
            profile_result (dict[str, Any] | None): The adopted attempt's result
                dict.
            recovered (bool): True when the attempt reported a non-success status
                but still produced a usable trace.
            params (dict[str, Any] | None): The profile params that produced the
                adopted trace.
        """
        result = _as_dict(profile_result)
        self._substep = "analysis"
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
        """Record one trace-analysis attempt.

        Args:
            run_index (int): 1-based attempt index within this action.
            attempt_reason (str): Why this attempt ran (an ``ANALYSIS_ATTEMPT_*``
                value).
            status (str): ``succeeded`` / ``failed``.
            started_at (str): ISO start of the attempt.
            duration_sec (float | None): Wall-clock seconds the attempt took.
            trace_input (str): The trace the attempt analyzed.
            requested_steady_state_mode (str): Steady-state mode asked of the
                splitter.
            ta_result (dict[str, Any] | None): The attempt's result dict, when it
                returned one.
            failure (dict[str, Any] | None): Failure row for a failed attempt.
        """
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
        """Mark one analysis attempt as the one the action concluded from.

        Args:
            run_index (int): The adopted attempt's index.
            ta_result (dict[str, Any] | None): The adopted attempt's result dict.
            trace_input (str): The trace the adopted attempt analyzed.
        """
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
        """Record the multi-node compute-bound re-profile decision.

        Args:
            attempted (bool): Whether a re-profile was attempted.
            adopted (bool): Whether its result was adopted.
            reason (str): Why it went the way it did.
        """
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
        """Close the action as succeeded and record the promoted artifacts.

        Args:
            snapshot_id (Any): The roofline snapshot id the recorder bumped to.
            hot_kernel_count (int): Hot kernels handed to candidate dispatch.
            kernel_attribution_degraded (bool): True when zero hot kernels are an
                attribution artifact rather than a real absence.
            cached (dict[str, Any] | None): The promoted ``last_trace_analyze``
                cache.
            trace_path (str): The profile trace the conclusion rests on.
        """
        promoted = _as_dict(cached)
        # Zero routable candidates is a completed roofline that cannot advance
        # kernel work, which is a different operational state from a clean run.
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
        """Close the action as failed, naming the sub-step that failed.

        Args:
            phase (str): The failed sub-step (``profile`` / ``profile_no_trace``
                / ``profile_capture_only`` / ``profile_zero_ops`` /
                ``trace_analyze``).
            error_class (str): The failure class the action reported.
            message (Any): The failure message.
        """
        self._close(
            status="failed",
            payload={
                "failed_substep": "analysis" if str(phase or "").startswith("trace_analyze") else "profile",
                "failure": _failure_row(
                    phase=phase,
                    error_class=error_class or f"{phase}_failed",
                    message=message,
                ),
            },
        )

    def finish_crashed(self, exc: BaseException) -> None:
        """Close an action whose executor raised instead of returning a result.

        Distinguishes "the executor blew up" from "the session was killed
        mid-roofline", which would otherwise both read as a dangling
        ``status="running"`` event.

        Args:
            exc (BaseException): The exception propagating out of the action.
        """
        if self._closed:
            return
        self.finish_failed(
            phase=self._substep or EVENT_TYPE,
            error_class=type(exc).__name__,
            message=f"roofline action raised: {exc!r}",
        )

    def _close(self, *, status: str, payload: Mapping[str, Any]) -> None:
        """Record the action's terminal facts and, when it owns the event, close it.

        Args:
            status (str): The status the action ended on.
            payload (Mapping[str, Any]): The terminal fields to record.
        """
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
    """Assemble one roofline action out of its recorded rows.

    Exists as its own entry point because the kernel event needs exactly this:
    one action, the one its re-profile dispatched, to place under
    ``forge.reprofile.run``.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The roofline sections as
            read back from the spool.
        event (str): The event id whose rows to select.
        task_id (str): The action to assemble.

    Returns:
        dict[str, Any] | None: The assembled action, or ``None`` when the event
            holds no action with that task id.
    """
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
    """Assemble every roofline action belonging to one event.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The roofline sections as
            read back from the spool.
        event (str): The event id whose rows to select.

    Returns:
        list[dict[str, Any]]: The actions, ordered by when they started.
    """
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
    """Assemble one roofline event's ``ext`` out of its recorded rows.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The roofline sections as
            read back from the spool.
        event (str): The event id to assemble.

    Returns:
        tuple[dict[str, Any], str]: The ``ext`` payload, holding one entry per
            action the event owns, and the status derived from them. A phase can
            dispatch roofline more than once in a macro cycle, so the event that
            holds them takes the worst of their statuses: an event reading
            ``succeeded`` while one of its actions failed would hide the failure
            behind the retry that recovered from it.
    """
    actions = assemble_roofline_actions(parts, event=event)
    return {"actions": actions}, _worst_status([str(action.get("status") or "") for action in actions])


def _mark_effective(runs: list[dict[str, Any]], effective_index: int | None) -> list[dict[str, Any]]:
    """Stamp ``effective`` onto the one run the action adopted.

    Args:
        runs (list[dict[str, Any]]): The action's run rows, ordered.
        effective_index (int | None): The adopted run's index, or ``None`` when
            the action adopted none.

    Returns:
        list[dict[str, Any]]: The rows, each with ``effective`` set.
    """
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
    """Build a recorder, or ``None`` when one cannot be constructed.

    Roofline behavior must not depend on the recorder existing, so construction
    failures degrade to "no event" rather than propagating -- as does an absent
    sink, which is what a caller with no session bound has.

    Args:
        sink (RecordSink | None): Where the rows go, or ``None`` to decline.
        task_id (str): Dispatched roofline task id.
        task_kind (str): The dispatched task's kind.
        reason (str): Dispatch reason.
        framework (str): Resolved serving framework.
        params (dict[str, Any] | None): The roofline task params.
        owns_event (bool): Whether this action owns the event it writes into.

    Returns:
        RooflineEventRecorder | None: The recorder, or ``None`` when it could
            not be built.
    """
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
        log.debug("roofline timeline: recorder construction failed", exc_info=True)
        return None
