# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real-time SBD V6 ``roofline`` timeline event recorder.

One ``roofline`` event is written per dispatched roofline action, at the moment
the action runs -- there is no export-time projection and no backfill. The
event's ``profile`` / ``analysis`` sub-steps are nested in ``ext`` rather than
emitted as sibling events: neither is dispatchable on its own, so a sibling
event would imply an autonomy that does not exist.

Multiplicity is expressed inside one event, not by emitting more events. Every
retry the executor performs (profile attempt loop, N26 steady-state re-analysis,
multi-node compute-bound re-profile) appends a row to ``runs`` and the row that
the action actually adopted is flagged ``effective`` and summarized under
``effective_run``. A session that dispatches roofline seven times therefore
produces seven events, while a roofline that internally re-profiled three times
produces one event with three profile runs.

The recorder flushes after every state change, so a session killed mid-roofline
leaves an event with ``status="running"`` naming the sub-step that was in
flight. Every write is best-effort: observability must never change roofline
behavior, so failures are logged and parked in the V6 write-warnings sidecar.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ._timeline_fields import (
    analysis_detail as _analysis_detail,
    as_dict as _as_dict,
    as_list as _as_list,
    clip as _clip,
    failure_row as _failure_row,
    flush_event as _flush_event,
    int_or_none as _int_or_none,
    now_iso_seconds as _now_iso,
)

log = logging.getLogger(__name__)

# ``trace_files`` reaches 424 entries on multi-rank xDiT runs (p99 424, p50 2),
# which would be ~85 KiB of paths per profile run. The rank histogram plus a few
# samples answers the same questions (is the file count plausible, did every rank
# report) at a bounded cost.
_MAX_SAMPLE_TRACE_FILES = 4

# Trace-structure issues are prose written for an operator; a handful is enough
# to characterize a degraded trace and the count carries the rest.
_MAX_TRACE_ISSUES = 8

_EVENT_TYPE = "roofline"

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


def _rank_of(path: str) -> str:
    """Extract the rank token from a per-rank trace filename.

    xDiT tensor/sequence-parallel profiles write one trace per rank, named with
    a ``rank<N>`` / ``_<N>.pt.trace.json.gz`` suffix. Grouping by rank turns a
    424-entry path list into a histogram that shows whether every rank reported.

    Args:
        path: A trace file path.

    Returns:
        The rank token as a string, or ``"unknown"`` when no rank is encoded.
    """
    name = Path(str(path)).name
    for token in name.replace("-", "_").split("_"):
        if token.startswith("rank") and token[4:].isdigit():
            return token[4:]
    return "unknown"


def _summarize_trace_files(profile_result: dict[str, Any]) -> dict[str, Any]:
    """Summarize the profile's trace file set without carrying every path.

    Args:
        profile_result: The profile sub-step result dict.

    Returns:
        A dict with the resolved main path, the file count, a per-rank
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
    """Project ``trace_health`` into the event's bounded health block.

    Carries the three booleans the executor branches on, the structured
    per-check rows the profile validator emits, and a clipped slice of the
    operator-facing issue prose.

    Args:
        profile_result: The profile sub-step result dict.

    Returns:
        The health summary dict.
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
    """Project the structured profile-trace validation into the event.

    The validator runs per profile attempt, so its verdict is stored on the run
    row rather than on the effective-run summary: "attempt 1 recorded no graph
    launches, attempt 2 did" is only answerable when each attempt keeps the
    verdict computed against the trace it produced.

    Args:
        profile_result: The profile sub-step result dict.

    Returns:
        The validation block, or an empty dict when the validator did not run.
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


class RooflineTimelineRecorder:
    """Accumulates and flushes one SBD V6 ``roofline`` timeline event.

    The recorder owns a single event dict for the lifetime of one roofline
    action. ``write_timeline_event`` stamps its storage sequence back onto that
    dict, so re-writing the same object updates the same file in place -- which
    is what lets the recorder flush on every state change without depending on
    a "latest event of this type" lookup that concurrent rooflines would race.
    """

    def __init__(
        self,
        session_dir: Path | str,
        *,
        task_id: str = "",
        task_kind: str = "",
        reason: str = "",
        framework: str = "",
        params: dict[str, Any] | None = None,
    ):
        """Start an in-flight roofline event.

        Args:
            session_dir: Session root the timeline lives under.
            task_id: Dispatched roofline task id.
            task_kind: The dispatched task's kind. The executor is also reused by
                the GEMM shape-capture path, which is not a roofline dispatch, so
                the event states which one it was rather than leaving a consumer
                to infer it from an absent reason.
            reason: Dispatch reason (``prelude_initial`` / ``close_post_opt`` /
                an LLM-proposed reason), which also selects the profiled arm.
            framework: Resolved serving framework.
            params: The roofline task params, read for workspace / arm context.
        """
        self._session_dir = Path(session_dir)
        self._t0 = time.monotonic()
        params = _as_dict(params)
        # ``arm`` names the configuration the run measured, which only a roofline
        # dispatch does. A roofline task may legitimately carry no reason, so the
        # claim is withheld by kind rather than by an empty reason -- otherwise
        # shape capture, which reuses this executor with no reason of its own,
        # would be recorded as having measured current_best.
        kind = str(task_kind or "")
        arm = (
            ""
            if kind not in ("", _EVENT_TYPE)
            else ("baseline" if str(reason or "") == "prelude_initial" else "current_best")
        )
        self._event: dict[str, Any] = {
            "type": _EVENT_TYPE,
            "kind": _EVENT_TYPE,
            "status": "running",
            "start_time": _now_iso(),
            "end_time": "",
            "ext": {
                "in_flight_substep": "profile",
                "failed_substep": None,
                "request": {
                    "task_id": str(task_id or ""),
                    "task_kind": kind,
                    "reason": str(reason or ""),
                    "arm": arm,
                    "framework": str(framework or ""),
                    "workspace_path": str(params.get("workspace_path") or ""),
                },
                "profile": {
                    "attempt_count": 0,
                    "max_attempts": 0,
                    "effective_run_index": None,
                    "recovered": False,
                    "eager_fallback_applied": False,
                    "runs": [],
                    "effective_run": {},
                },
                "analysis": {
                    "attempt_count": 0,
                    "effective_run_index": None,
                    "n26_auto_retry": None,
                    "compute_bound_reprofile": {"attempted": False, "adopted": False, "reason": ""},
                    "runs": [],
                    "effective_run": {},
                },
                "outcome": {},
                "failure": None,
            },
        }

    # ---- internals -------------------------------------------------------

    @property
    def _ext(self) -> dict[str, Any]:
        return self._event["ext"]

    def _flush(self, component: str) -> None:
        """Persist the event, parking any writer failure for the next export."""
        _flush_event(self._session_dir, self._event, component=f"roofline.{component}")

    # ---- lifecycle -------------------------------------------------------

    def begin(self, *, max_profile_attempts: int) -> None:
        """Write the in-flight event so a killed session still shows the stage.

        Args:
            max_profile_attempts: The profile retry budget, recorded so a run
                that exhausted it is distinguishable from one that stopped early.
        """
        self._ext["profile"]["max_attempts"] = int(max_profile_attempts)
        self._flush("begin")

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
        """Append one profile attempt.

        Args:
            run_index: 1-based attempt index within this roofline.
            attempt_reason: Why this attempt ran (a ``PROFILE_ATTEMPT_*`` value).
            status: ``succeeded`` / ``recovered`` / ``failed``.
            started_at: ISO start of the attempt.
            duration_sec: Wall-clock seconds the attempt took.
            disable_cuda_graph: Whether the attempt booted eager.
            profile_result: The attempt's result dict, when it returned one.
            failure: Failure row for a failed attempt.
        """
        result = _as_dict(profile_result)
        run: dict[str, Any] = {
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
        }
        runs = self._ext["profile"]["runs"]
        runs.append(run)
        self._ext["profile"]["attempt_count"] = len(runs)
        if disable_cuda_graph:
            self._ext["profile"]["eager_fallback_applied"] = True
        self._flush("profile_run")

    def adopt_profile_run(
        self,
        *,
        run_index: int,
        profile_result: dict[str, Any] | None,
        recovered: bool = False,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Mark one profile attempt as the one the action carried forward.

        Args:
            run_index: The adopted attempt's index.
            profile_result: The adopted attempt's result dict.
            recovered: True when the attempt reported a non-success status but
                still produced a usable trace.
            params: The profile params that produced the adopted trace.
        """
        result = _as_dict(profile_result)
        profile = self._ext["profile"]
        profile["effective_run_index"] = int(run_index)
        profile["recovered"] = bool(recovered)
        for run in profile["runs"]:
            run["effective"] = int(run.get("run_index") or 0) == int(run_index)
        profile["effective_run"] = {
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
        }
        self._ext["in_flight_substep"] = "analysis"
        self._flush("profile_adopt")

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
        """Append one trace-analysis attempt.

        Args:
            run_index: 1-based attempt index within this roofline.
            attempt_reason: Why this attempt ran (an ``ANALYSIS_ATTEMPT_*`` value).
            status: ``succeeded`` / ``failed``.
            started_at: ISO start of the attempt.
            duration_sec: Wall-clock seconds the attempt took.
            trace_input: The trace the attempt analyzed.
            requested_steady_state_mode: Steady-state mode asked of the splitter.
            ta_result: The attempt's result dict, when it returned one.
            failure: Failure row for a failed attempt.
        """
        result = _as_dict(ta_result)
        meta = _as_dict(result.get("analysis_meta"))
        run: dict[str, Any] = {
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
        }
        runs = self._ext["analysis"]["runs"]
        runs.append(run)
        self._ext["analysis"]["attempt_count"] = len(runs)
        self._flush("analysis_run")

    def adopt_analysis_run(
        self,
        *,
        run_index: int,
        ta_result: dict[str, Any] | None,
        trace_input: str,
    ) -> None:
        """Mark one analysis attempt as the one the action concluded from.

        Args:
            run_index: The adopted attempt's index.
            ta_result: The adopted attempt's result dict.
            trace_input: The trace the adopted attempt analyzed.
        """
        result = _as_dict(ta_result)
        analysis = self._ext["analysis"]
        analysis["effective_run_index"] = int(run_index)
        for run in analysis["runs"]:
            run["effective"] = int(run.get("run_index") or 0) == int(run_index)
        n26 = _as_dict(result.get("n26_auto_retry"))
        if n26:
            analysis["n26_auto_retry"] = n26
        analysis["effective_run"] = {
            "run_index": int(run_index),
            "trace_input": str(trace_input or ""),
            "orchestrator_mode": str(result.get("orchestrator_mode") or ""),
            "orchestrator_error": _clip(result.get("orchestrator_error")),
            **_analysis_detail(result),
        }
        self._flush("analysis_adopt")

    def record_compute_bound_reprofile(self, *, attempted: bool, adopted: bool, reason: str = "") -> None:
        """Record the multi-node compute-bound re-profile decision."""
        self._ext["analysis"]["compute_bound_reprofile"] = {
            "attempted": bool(attempted),
            "adopted": bool(adopted),
            "reason": _clip(reason),
        }
        self._flush("compute_bound")

    def finish_succeeded(
        self,
        *,
        snapshot_id: Any,
        hot_kernel_count: int,
        kernel_attribution_degraded: bool,
        cached: dict[str, Any] | None,
        trace_path: str,
    ) -> None:
        """Close the event as succeeded and record the promoted artifacts.

        Args:
            snapshot_id: The roofline snapshot id the recorder bumped to.
            hot_kernel_count: Hot kernels handed to candidate dispatch.
            kernel_attribution_degraded: True when zero hot kernels are an
                attribution artifact rather than a real absence.
            cached: The promoted ``last_trace_analyze`` cache.
            trace_path: The profile trace the conclusion rests on.
        """
        promoted = _as_dict(cached)
        self._ext["in_flight_substep"] = None
        self._ext["outcome"] = {
            "snapshot_id": _int_or_none(promoted.get("roofline_snapshot_id") or snapshot_id),
            "hot_kernel_count": int(hot_kernel_count),
            "kernel_attribution_degraded": bool(kernel_attribution_degraded),
            "profile_trace": str(trace_path or ""),
            "steady_state_trace": str(promoted.get("steady_state_trace") or ""),
            "analysis_md_path": str(promoted.get("analysis_md_path") or ""),
            "candidates_path": str(promoted.get("candidates_path") or ""),
            "kernel_roofline_path": str(promoted.get("kernel_roofline_path") or ""),
        }
        # Zero routable candidates is a completed roofline that cannot advance
        # kernel work, which is a different operational state from a clean run.
        self._event["status"] = "degraded" if kernel_attribution_degraded else "succeeded"
        self._event["end_time"] = _now_iso()
        self._ext["duration_sec"] = round(time.monotonic() - self._t0, 3)
        self._flush("finish")

    def finish_crashed(self, exc: BaseException) -> None:
        """Close an event whose action raised instead of returning a result.

        Distinguishes "the executor blew up" from "the session was killed
        mid-roofline", which would otherwise both read as a dangling
        ``status="running"`` event.

        Args:
            exc: The exception propagating out of the action.
        """
        if self._event.get("end_time"):
            return
        self.finish_failed(
            phase=str(self._ext.get("in_flight_substep") or "roofline"),
            error_class=type(exc).__name__,
            message=f"roofline action raised: {exc!r}",
        )

    def finish_failed(self, *, phase: str, error_class: str = "", message: Any = "") -> None:
        """Close the event as failed, naming the sub-step that failed.

        Args:
            phase: The failed sub-step (``profile`` / ``profile_no_trace`` /
                ``profile_capture_only`` / ``profile_zero_ops`` / ``trace_analyze``).
            error_class: The failure class the action reported.
            message: The failure message.
        """
        substep = "analysis" if str(phase or "").startswith("trace_analyze") else "profile"
        self._ext["in_flight_substep"] = None
        self._ext["failed_substep"] = substep
        self._ext["failure"] = _failure_row(
            phase=phase,
            error_class=error_class or f"{phase}_failed",
            message=message,
        )
        self._event["status"] = "failed"
        self._event["end_time"] = _now_iso()
        self._ext["duration_sec"] = round(time.monotonic() - self._t0, 3)
        self._flush("finish_failed")


def make_roofline_recorder(
    session_dir: Path | str,
    *,
    task_id: str = "",
    task_kind: str = "",
    reason: str = "",
    framework: str = "",
    params: dict[str, Any] | None = None,
) -> RooflineTimelineRecorder | None:
    """Build a recorder, or ``None`` when one cannot be constructed.

    Roofline behavior must not depend on the recorder existing, so construction
    failures degrade to "no event" rather than propagating.

    Declines to record when the session directory did not resolve to a real
    session: ``_resolve_session_dir`` falls back to ``Path(".")``, and writing
    the timeline there would drop events into whatever the working directory
    happens to be rather than into the session.

    Args:
        session_dir: Session root the timeline lives under.
        task_id: Dispatched roofline task id.
        task_kind: The dispatched task's kind.
        reason: Dispatch reason.
        framework: Resolved serving framework.
        params: The roofline task params.

    Returns:
        The recorder, or ``None`` when it could not be built.
    """
    try:
        root = Path(session_dir)
        if not root.name or not root.is_dir():
            log.debug("roofline timeline: unresolved session dir %r; not recording", str(session_dir))
            return None
        return RooflineTimelineRecorder(
            session_dir,
            task_id=task_id,
            task_kind=task_kind,
            reason=reason,
            framework=framework,
            params=params,
        )
    except Exception:  # noqa: BLE001 — observability cannot change roofline behavior
        log.debug("roofline timeline: recorder construction failed", exc_info=True)
        return None


__all__ = [
    "ANALYSIS_ATTEMPT_COMPUTE_BOUND",
    "ANALYSIS_ATTEMPT_INITIAL",
    "ANALYSIS_ATTEMPT_N26_RETRY",
    "PROFILE_ATTEMPT_AFTER_BAD_RETURN",
    "PROFILE_ATTEMPT_AFTER_CAPTURE_ONLY",
    "PROFILE_ATTEMPT_AFTER_EXCEPTION",
    "PROFILE_ATTEMPT_AFTER_FAILURE",
    "PROFILE_ATTEMPT_AFTER_NO_TRACE",
    "PROFILE_ATTEMPT_AFTER_ZERO_OPS",
    "PROFILE_ATTEMPT_COMPUTE_BOUND",
    "PROFILE_ATTEMPT_INITIAL",
    "RooflineTimelineRecorder",
    "make_roofline_recorder",
]
