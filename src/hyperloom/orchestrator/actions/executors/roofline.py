# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Roofline composite ActionRunner."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from hyperloom.common.provenance import detect_kineto_backend
from hyperloom_common.timeutil import now_iso
from ...phases.machine_state import LifecycleMark, record_lifecycle_event
from ...loop.sub_agent_runner import RunnerContext
from ...state.analysis_failures import remember_roofline_failure, remember_trace_failure
from hyperloom.inference_optimizer.trace.task_progress import report_progress
from ._multi_node_env import is_multi_node
from ..stop_attribution import ORCHESTRATOR_CANCELLED_CLASS
from hyperloom.inference_optimizer.breakdown.recorder.event_ids import INLINE_EVENT_PARAM
from hyperloom.inference_optimizer.breakdown.recorder.roofline_event import (
    ANALYSIS_ATTEMPT_COMPUTE_BOUND,
    ANALYSIS_ATTEMPT_INITIAL,
    ANALYSIS_ATTEMPT_N26_RETRY,
    PRODUCER as _RECORDER_PRODUCER,
    PROFILE_ATTEMPT_COMPUTE_BOUND,
    PROFILE_ATTEMPT_INITIAL,
    AnalysisRunOutcome,
    ProfileRunOutcome,
    RooflineConclusion,
    RooflineRequest,
    make_roofline_recorder,
    roofline_arm,
    roofline_event_id,
)

log = logging.getLogger(__name__)

#: Task params copied into a capture-failure diagnosis. A capture failure is judged against the shape that was
#: being captured, so the diagnosis has to carry the shape rather than expect a reader to rejoin it by task id.
_CAPTURE_DIAGNOSIS_WORKLOAD_KEYS = (
    "benchmark_mode",
    "framework",
    "precision",
    "model_path",
    "tp",
    "concurrency",
    "isl",
    "osl",
    "max_model_len",
)
#: Per-source cap on the raw text kept in a capture-failure diagnosis. Generous on purpose: this file is the only
#: record of the failure, since nothing downstream re-runs the arm to reproduce it.
_CAPTURE_DIAGNOSIS_SOURCE_BYTES = 32768
#: Suffixes the live trace resolver will consider. One of these already present in a task's own run directory is
#: a leftover the resolver can pick instead of what this run produces.
_TRACE_SUFFIXES = (".pt.trace.json.gz", ".pt.trace.json", ".trace.json.gz", ".trace.json")
#: Enough leftovers to establish the pattern; the preflight record is context, not an inventory.
_PREFLIGHT_STALE_TRACE_LIMIT = 20
_TERMINAL_PROFILE_ERRORS = frozenset(
    {
        ORCHESTRATOR_CANCELLED_CLASS,
        "primary_rank_trace_missing",
    }
)
_TERMINAL_CAPTURE_REASONS = frozenset(
    {
        "api_port_allocation_failed",
        "capture_status_missing",
        "capture_status_unreadable",
        "profiler_output_unconfigured",
    }
)

# Env switch for the multi-node compute-bound auto re-profile (default on; set to "0" to disable).
_AUTO_COMPUTE_BOUND_ENV = "HYPERLOOM_PROFILE_AUTO_COMPUTE_BOUND"


async def _reap_session_orphans(session_dir: Path | str) -> list[int]:
    """Reap this session's own orphaned serving processes."""
    from ._server_lifecycle import reap_orphaned_servers

    return await asyncio.to_thread(reap_orphaned_servers, Path(session_dir))


def _gpu_trace_unsupported_reason(profile_result: dict[str, Any]) -> str:
    """Why this stack can never record GPU kernels, or empty when the capture merely failed this time. Demands a
    parsed trace carrying host ops beside zero kernels, so one transient empty capture cannot condemn the session.
    """
    if not isinstance(profile_result, dict):
        return ""
    measures = ((profile_result.get("trace_validate") or {}).get("verdict") or {}).get("measures") or {}
    kernel_count = measures.get("kernel_count") if isinstance(measures, dict) else None
    if not isinstance(kernel_count, int) or kernel_count != 0:
        return ""
    health = profile_result.get("trace_health")
    if not isinstance(health, dict) or health.get("zero_ops") is not False:
        return ""
    return (
        f"the profiler recorded {kernel_count} GPU kernels beside a populated host timeline, "
        "so this stack cannot capture GPU traces at all"
    )


def _trace_is_high_idle(ta_result: dict[str, Any]) -> bool:
    """Whether trace_analyze flagged the profiled step as host-bound (high GPU idle), i.e. carries a ``high_gpu_idle_pct`` trace-health warning."""
    if not isinstance(ta_result, dict):
        return False
    for w in ta_result.get("trace_health_warnings") or []:
        if isinstance(w, dict) and w.get("code") == "high_gpu_idle_pct":
            return True
    return False


# seconds + ``+00:00`` (canonical helper; kept importable for callers).
_now_iso = functools.partial(now_iso, "seconds")


# Auto-recover from TraceLens steady_state_chunk_* failures: re-issue ONCE with the first non-empty mode from the
# warning's ``non_empty_modes``.
_AUTO_RETRY_WARNING_CODES = frozenset(
    {
        "steady_state_chunk_missing",
        # low-quality or zero-GPU-events chunk; same recovery path via ``non_empty_modes``.
        "steady_state_chunk_low_quality",
    }
)


def _extract_steady_state_retry_mode(
    ta_result: dict[str, Any],
) -> "tuple[str, dict[str, Any]] | None":
    """Inspect a failed trace_analyze result for a steady-state recovery hint."""
    if not isinstance(ta_result, dict):
        return None
    warnings = ta_result.get("trace_health_warnings") or []
    if not isinstance(warnings, list):
        return None
    for w in warnings:
        if not isinstance(w, dict):
            continue
        if w.get("code") not in _AUTO_RETRY_WARNING_CODES:
            continue
        # Splitter-accepted alternates (non_empty_modes / available_modes).
        modes = w.get("non_empty_modes") or w.get("available_modes") or []
        if not isinstance(modes, list):
            continue
        for candidate in modes:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip(), w
    return None


def _extract_trace_path(profile_result: dict[str, Any]) -> str:
    """Pick the trace path like Coordinator's ``promote_to_shared_state``: prefer ``main_trace_path``, else
    ``trace_files[0]`` for legacy results.
    """
    if not isinstance(profile_result, dict):
        return ""
    if profile_result.get("trace_input_ready") is False:
        return ""
    direct = profile_result.get("main_trace_path")
    if direct:
        return str(direct)
    files = profile_result.get("trace_files")
    if isinstance(files, (list, tuple)) and files:
        first = files[0]
        if first:
            return str(first)
    return ""


def _fail(
    recorder: Any,
    phase: str,
    error: str,
    sub_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Close the timeline event as failed, then build the canonical failure result.

    Every failure exit goes through this, so one added later cannot be the one that leaves the event dangling.
    """
    if recorder is not None:
        recorder.finish_failed(phase=phase, message=error)
    out: dict[str, Any] = {
        "status": "failed",
        "error_class": f"{phase}_failed",
        "error": error,
        "phase": phase,
        "executed_at_iso": _now_iso(),
    }
    if isinstance(sub_result, dict):
        out["sub_result"] = {
            k: sub_result.get(k)
            for k in (
                "status",
                "error",
                "error_class",
                "main_trace_path",
                "trace_files",
                "analysis_md_path",
                "hot_kernels",
            )
            if k in sub_result
        }
    return out


def _profile_err_text(profile_result: Any) -> str:
    """Flatten a profile result's error fields into one blob for cuda-graph capture-failure detection."""
    if not isinstance(profile_result, dict):
        return ""
    parts = [str(profile_result.get(k) or "") for k in ("error", "error_class", "error_excerpt", "stderr_tail")]
    sub = profile_result.get("sub_result")
    if isinstance(sub, dict):
        parts += [str(sub.get(k) or "") for k in ("error", "error_class")]
    return "\n".join(parts)


def _profile_server_log_tail(profile_result: Any, max_bytes: int = 16384) -> str:
    """Return the tail of the newest engine ``server.log`` for a profile run."""
    if not isinstance(profile_result, dict):
        return ""
    base = profile_result.get("trace_dir") or profile_result.get("workspace")
    if not base:
        return ""
    try:
        from .benchmark_result import _find_server_logs

        logs = _find_server_logs(Path(str(base)))
        if not logs:
            return ""
        return logs[0].read_bytes()[-max_bytes:].decode("utf-8", "replace")
    except (OSError, ImportError):
        return ""


def _marker_hit_context(sources: dict[str, str], marker: str, radius: int = 6) -> list[dict[str, Any]]:
    """Return the lines around each source's first hit on ``marker``.

    The classifier matches lowercased substrings, so it can be fooled; the excerpt is the original text, because
    that is what a reader needs to overrule a classification they think is wrong.
    """
    needle = marker.split(" + ")[0].lower()
    hits: list[dict[str, Any]] = []
    for name, text in sources.items():
        if not text:
            continue
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if needle in line.lower():
                hits.append(
                    {
                        "source": name,
                        "line_no": idx + 1,
                        "excerpt": lines[max(0, idx - radius) : idx + radius + 1],
                    }
                )
                break
    return hits


def _trace_dir_inventory(profile_result: Any, limit: int = 200) -> dict[str, Any]:
    """List what a failed profile left on disk, so a truncated export reads differently from no export at all."""
    if not isinstance(profile_result, dict):
        return {}
    base = profile_result.get("trace_dir") or profile_result.get("workspace")
    if not base:
        return {}
    root = Path(str(base))
    try:
        entries = sorted(p for p in root.rglob("*") if p.is_file())
    except OSError as exc:
        # The inventory is evidence about a failure, so it reports its own trouble rather than raising over it.
        return {"root": str(root), "error": repr(exc)}
    files: list[dict[str, Any]] = []
    for p in entries[:limit]:
        try:
            files.append({"path": str(p.relative_to(root)), "size_bytes": p.stat().st_size})
        except OSError:
            continue
    out: dict[str, Any] = {"root": str(root), "file_count": len(entries), "files": files}
    if len(entries) > limit:
        out["truncated"] = True
    return out


def _iso_from_epoch(ts: float) -> str:
    """Render a filesystem timestamp in the same UTC ISO form the rest of the event uses."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _preflight_probe(session_dir: Path, task_id: str, *, reaped: Sequence[int]) -> dict[str, Any]:
    """Describe the conditions the action is about to profile under. Never raises.

    Reported, not acted on: a leftover server, a nearly-full disk or a trace file already sitting in this task's
    own output directory all change how the result should be read, but deciding what to do about them is not a
    measurement stage's call.
    """
    out: dict[str, Any] = {
        "orphans_reaped": list(reaped),
        "orphans_reaped_count": len(reaped),
    }
    try:
        usage = shutil.disk_usage(session_dir)
        out["disk"] = {
            "path": str(session_dir),
            "free_bytes": usage.free,
            "total_bytes": usage.total,
            "free_pct": round(100.0 * usage.free / usage.total, 2) if usage.total else None,
        }
    except OSError as exc:
        out["disk"] = {"path": str(session_dir), "error": repr(exc)}

    # A trace already under this task's own run directory is what later surfaces as selfcert's
    # ``production_pick_probe`` disagreement: the live resolver can pick the leftover instead of what this run
    # produces, and by then there is no way to tell which run the numbers came from.
    stale: list[dict[str, Any]] = []
    if task_id:
        try:
            for run_dir in sorted(session_dir.glob(f"runs/*/{task_id}")):
                for p in sorted(run_dir.rglob("*")):
                    if len(stale) >= _PREFLIGHT_STALE_TRACE_LIMIT:
                        break
                    if p.is_file() and str(p).endswith(_TRACE_SUFFIXES):
                        info = p.stat()
                        stale.append(
                            {
                                "path": str(p.relative_to(session_dir)),
                                "size_bytes": info.st_size,
                                "mtime_iso": _iso_from_epoch(info.st_mtime),
                            }
                        )
        except OSError as exc:
            out["stale_traces_error"] = repr(exc)
    out["stale_traces"] = stale
    out["stale_trace_count"] = len(stale)
    return out


def _drain_instrumentation(executor: Any) -> dict[str, Any] | None:
    """Take the profile executor's per-attempt instrumentation report, if it keeps one."""
    drain = getattr(executor, "drain_instrumentation_report", None)
    if not callable(drain):
        return None
    report = drain()
    return report if isinstance(report, dict) else None


def _server_liveness_probe(session_dir: Path, task_id: str) -> dict[str, Any]:
    """Report, without touching them, whether the servers this task started outlived the profile. Never raises.

    ``dead_with_pidfile`` is the one this exists for: teardown unlinks the pidfile, so a pidfile naming a dead
    process means the engine died instead of being torn down. That run can still have exported a complete trace,
    which is exactly why the result dict alone cannot show it.
    """
    from hyperloom.common.proctree import running

    from ._server_lifecycle import _looks_like_server_process

    runs_dir = session_dir / "runs"
    if not runs_dir.is_dir():
        return {}
    entries: list[dict[str, Any]] = []
    try:
        pid_files = sorted(runs_dir.rglob("*.pid"))
    except OSError as exc:
        return {"error": repr(exc)}
    for pid_file in pid_files:
        if task_id and task_id not in str(pid_file):
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").split()[0])
        except (OSError, IndexError, ValueError):
            continue
        alive = running(pid)
        entries.append(
            {
                "pid_file": str(pid_file.relative_to(session_dir)),
                "pid": pid,
                "alive": alive,
                # A live pid that no longer looks like a server is pid reuse, not a surviving engine.
                "is_server": _looks_like_server_process(pid) if alive else False,
            }
        )
    if not entries:
        return {"pidfiles": 0}
    return {
        "pidfiles": len(entries),
        "alive": sum(1 for e in entries if e["alive"]),
        "dead_with_pidfile": sum(1 for e in entries if not e["alive"]),
        "entries": entries,
    }


def _write_capture_failure_diagnosis(path: Path, payload: dict[str, Any]) -> str:
    """Write the capture-failure evidence file; return its path, or ``""`` when it could not be written.

    A failure to write the evidence must not displace the capture failure it describes, so this degrades to an
    empty path and lets the caller's message stand on its own.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        log.warning("roofline: could not write capture-failure diagnosis to %s: %r", path, exc)
        return ""
    return str(path)


async def _reported(label: str, start: Callable[[], Awaitable[Any]], **fields: Any) -> Any:
    """Announce a roofline sub-step, then await it.

    Every sub-step goes through this, so a call site added later cannot silently be the one that reports nothing.
    """
    await report_progress(unit="roofline_step", label=label, status="started", **fields)
    return await start()


@dataclass(frozen=True)
class _ProfileOutcome:
    """The profile a roofline adopts: its result, trace, degraded-status warning and launch params."""

    result: dict[str, Any]
    trace_path: str
    warning: dict[str, Any] | None
    params: dict[str, Any]

    @classmethod
    def of(cls, result: dict[str, Any], params: dict[str, Any]) -> _ProfileOutcome:
        """The outcome of adopting ``result``; a non-succeeded status that still produced a trace is recovered."""
        recovered = result.get("status") != "succeeded"
        warning = {key: result.get(key) for key in ("status", "error_class", "error")} if recovered else None
        return cls(result=result, trace_path=_extract_trace_path(result), warning=warning, params=params)


@dataclass(frozen=True)
class _RunEnd:
    """How one profile or analysis attempt ended, as its breakdown row records it."""

    status: str
    result: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None


@dataclass(frozen=True)
class _AnalysisAttempt:
    """Which trace-analysis attempt a run is: its reported label, recorded reason and failure-message context."""

    label: str
    attempt_reason: str
    context: str = ""


@dataclass(frozen=True)
class _AnalysisOutcome:
    """The trace analysis a roofline concludes from: the request it ran, its result and its run index."""

    payload: dict[str, Any]
    result: dict[str, Any]
    run_index: int


@dataclass(frozen=True)
class _AnalysedProfile:
    """A profile and the successful analysis of its trace: shared state only ever records the two together."""

    profile: _ProfileOutcome
    analysis: _AnalysisOutcome


@dataclass(frozen=True)
class _ProfileArm:
    """The configuration every profile attempt of one action runs under.

    Fixed for the whole action: roofline profiles the arm it was handed. Re-booting eager after a capture crash would
    produce a trace of a configuration nobody asked to measure, and choosing a configuration is enablement's job.
    ``HYPERLOOM_PROFILE_DISABLE_CUDA_GRAPH`` stays as an operator override, read once per action.
    """

    framework: str
    env_disable_cuda_graph: str

    @classmethod
    def resolve(cls, framework: str) -> _ProfileArm:
        """Read the operator override for an action profiling ``framework``."""
        return cls(framework, os.environ.get("HYPERLOOM_PROFILE_DISABLE_CUDA_GRAPH", "").strip())

    @property
    def disable_cuda_graph(self) -> bool:
        """bool: Whether the override turns graph capture off."""
        return self.env_disable_cuda_graph.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class _Unusable:
    """Why the profile cannot be carried forward."""

    stage: str
    error_class: str
    message: str
    #: What the profile returned, when it returned a dict.
    result: dict[str, Any] | None = None
    #: The text a cuda-graph capture failure is recognised in; empty where it cannot apply.
    evidence: dict[str, str] = field(default_factory=dict)

    @property
    def failure(self) -> dict[str, str]:
        """dict: The failure block of the profile's run row."""
        return {"stage": self.stage, "error_class": self.error_class, "message": self.message}


#: What a profile that reported success can still hand back unusable, checked in order:
#: (test on the result and its trace path, stage, error class, message).
_UNUSABLE_TRACES: tuple[tuple[Callable[[dict[str, Any], str], bool], str, str, str], ...] = (
    (
        lambda result, trace: not trace,
        "profile_no_trace",
        "no_trace",
        "profile succeeded but no trace_path in result (missing both main_trace_path and trace_files[0])",
    ),
    (
        lambda result, trace: result.get("profile_trace_selection_reason") == "capture_only_fallback",
        "profile_capture_only",
        "capture_only",
        "profile produced only CUDA-graph capture sidecars under capture_traces/ (no annotated steady-state trace); "
        "the steady-state splitter cannot use these",
    ),
    (
        lambda result, trace: bool((result.get("trace_health") or {}).get("zero_ops")),
        "profile_zero_ops",
        "zero_ops",
        "profile produced a metadata-only trace (PyTorch Profiler Op count == 0); the active capture window never "
        "overlapped execution",
    ),
)


def _raised(exc: Exception) -> _Unusable:
    """Classify a profile call that raised; only ``repr(exc)`` is there to recognise a capture failure in."""
    message = f"profile_executor raised: {exc!r}"
    return _Unusable("profile", type(exc).__name__, message, evidence={"last_error": message})


def _classify_profile(result: Any) -> _Unusable | None:
    """Classify what one profile call returned; ``None`` when its trace can be analysed."""
    if not isinstance(result, dict):
        message = f"profile_executor returned non-dict: {type(result).__name__}"
        return _Unusable("profile", "bad_return", message)
    trace_path = _extract_trace_path(result)
    if result.get("status") != "succeeded":
        error = str(result.get("error") or "profile sub-step failed")
        capture_reason = str((result.get("trace_capture") or {}).get("reason") or "")
        if result.get("error_class") in _TERMINAL_PROFILE_ERRORS or capture_reason in _TERMINAL_CAPTURE_REASONS:
            # Checked before the trace is adopted: a cancelled profile can already have flushed one, and
            # analysing it would run the analysis and a re-profile inside the cancelled scope.
            return _Unusable("profile", str(result.get("error_class") or ""), error, result)
        # A duplicate stop_profile failure can arrive after a trace was already flushed successfully.
        if trace_path:
            return None
        evidence = {
            "last_error": error,
            "profile_error": _profile_err_text(result),
            "server_log_tail": _profile_server_log_tail(result),
        }
        return _Unusable("profile", str(result.get("error_class") or ""), error, result, evidence)
    for matches, stage, error_class, message in _UNUSABLE_TRACES:
        if matches(result, trace_path):
            return _Unusable(stage, error_class, message, result)
    return None


async def _end_profile_run(
    recorder: Any,
    run_index: int,
    session_dir: Path,
    arm: _ProfileArm,
    end: _RunEnd,
) -> None:
    """Row one profile attempt with the process and patch state it left behind."""
    from .profile import profile_executor

    if recorder is None:
        return
    recorder.end_profile_run(
        run_index,
        ProfileRunOutcome(
            status=end.status,
            disable_cuda_graph=arm.disable_cuda_graph,
            profile_result=end.result,
            failure=end.failure,
            # Probed on every attempt, not only the failed ones: the run this is meant to catch is the one that reports
            # success.
            server_liveness=await asyncio.to_thread(_server_liveness_probe, session_dir, recorder.task_id),
            # Drained per attempt, so every attempt carries which patchers ran and what they returned -- including the
            # attempts that raised, where no result dict exists to carry it.
            instrumentation=_drain_instrumentation(profile_executor),
        ),
    )


async def _adopt_profile(
    recorder: Any,
    run_index: int,
    session_dir: Path,
    arm: _ProfileArm,
    profiled: _ProfileOutcome,
) -> _ProfileOutcome:
    """Row the profile whose trace the action carries forward, and adopt it."""
    result, params = profiled.result, profiled.params
    recovered = result.get("status") != "succeeded"
    if recovered:
        log.warning(
            "roofline profile returned status=%r but produced trace=%s; continuing to trace_analyze",
            result.get("status"),
            profiled.trace_path,
        )
    status = "recovered" if recovered else "succeeded"
    await _end_profile_run(recorder, run_index, session_dir, arm, _RunEnd(status, result))
    if recorder is not None:
        recorder.adopt_profile_run(run_index=run_index, profile_result=result, recovered=recovered, params=params)
    return profiled


async def _fail_on_capture(
    recorder: Any,
    task: Any,
    session_dir: Path,
    arm: _ProfileArm,
    unusable: _Unusable,
) -> dict[str, Any] | None:
    """Fail on a classified cuda-graph capture failure, leaving its evidence on disk; ``None`` when there is none.

    Roofline does not re-boot eager here: a measurement stage that edits its own configuration reports a number for
    an arm that was never requested. The split between ``instrumentation`` (the profiler collided with capture, so the arm itself is
    fine) and ``config`` (these server args cannot capture at all) decides who owns the fix, so it is recorded
    rather than acted on here.
    """
    from .benchmark_run.bringup import _classify_cuda_graph_capture_failure

    # With capture already off, a marker in the log tail says nothing about this run -- the tail can span an earlier
    # boot -- and "does not re-profile with graph capture disabled" would be nonsense to read on such a run.
    sources = unusable.evidence
    if arm.disable_cuda_graph or not sources:
        return None
    category, marker = _classify_cuda_graph_capture_failure(*sources.values())
    if not category:
        return None
    task_id = str(getattr(task, "task_id", "") or "") or "unknown"
    params = dict(task.params or {})
    diagnosis = _write_capture_failure_diagnosis(
        session_dir / "diagnostics" / f"roofline_cuda_graph_capture_{task_id}.json",
        {
            "category": category,
            "matched_marker": marker,
            "task_id": task_id,
            "recorded_at_iso": _now_iso(),
            "config": {
                "framework": arm.framework,
                "disable_cuda_graph": arm.disable_cuda_graph,
                "env_disable_cuda_graph": arm.env_disable_cuda_graph,
                "extra_server_args": params.get("extra_server_args"),
                "workload": {k: params.get(k) for k in _CAPTURE_DIAGNOSIS_WORKLOAD_KEYS if k in params},
            },
            "marker_hits": _marker_hit_context(sources, marker),
            "sources": {k: v[-_CAPTURE_DIAGNOSIS_SOURCE_BYTES:] for k, v in sources.items() if v},
            "trace_dir": await asyncio.to_thread(_trace_dir_inventory, unusable.result),
        },
    )
    message = (
        f"cuda-graph capture failed ({category}-rooted; marker={marker!r}); roofline profiles the configuration it "
        f"was given and does not re-profile with graph capture disabled. Evidence: {diagnosis or '<unwritten>'}"
    )
    log.warning("roofline: %s", message)
    return _fail(recorder, f"profile_cuda_graph_capture_{category}", message, unusable.result)


async def _compute_bound_profile(
    recorder: Any, session_dir: Path, arm: _ProfileArm, cb_ctx: RunnerContext
) -> tuple[int, Any]:
    """Run and row the compute-bound profile; return its run index and what the profiler returned.

    Rowed before anything raises out of here: the fail-soft handler only narrates the outcome, and an attempt the
    event never rows is an attempt ``attempt_count`` does not count.
    """
    from .profile import profile_executor

    run_index = recorder.begin_profile_run(attempt_reason=PROFILE_ATTEMPT_COMPUTE_BOUND) if recorder is not None else 0
    try:
        cb_profile = await _reported("profile_compute_bound", lambda: profile_executor(cb_ctx))
    except Exception as exc:
        message = f"compute-bound re-profile raised: {exc!r}"
        failure = {"stage": "profile", "error_class": type(exc).__name__, "message": message}
        await _end_profile_run(recorder, run_index, session_dir, arm, _RunEnd("failed", failure=failure))
        raise
    result = cb_profile if isinstance(cb_profile, dict) else None
    if _extract_trace_path(cb_profile):
        await _end_profile_run(recorder, run_index, session_dir, arm, _RunEnd("succeeded", result))
    else:
        message = "compute-bound re-profile produced no trace path"
        failure = {"stage": "profile_no_trace", "error_class": "no_trace", "message": message}
        await _end_profile_run(recorder, run_index, session_dir, arm, _RunEnd("failed", result, failure))
    return run_index, cb_profile


async def _compute_bound_analysis(recorder: Any, session_dir: Path, payload: dict[str, Any]) -> _AnalysisOutcome | None:
    """Run and row the compute-bound re-analysis; return it when it succeeded, else ``None``.

    Rowed before anything raises out of here, for the same reason as the re-profile.
    """
    from .trace_analyze import trace_analyze_handler

    run_index = (
        recorder.begin_analysis_run(attempt_reason=ANALYSIS_ATTEMPT_COMPUTE_BOUND) if recorder is not None else 0
    )
    try:
        result = await _reported(
            "trace_analyze_compute_bound", lambda: trace_analyze_handler(payload, session_dir=session_dir)
        )
    except Exception as exc:
        message = f"compute-bound re-analysis raised: {exc!r}"
        failure = {"stage": "trace_analyze", "error_class": type(exc).__name__, "message": message}
        _end_analysis_run(recorder, run_index, payload, _RunEnd("failed", failure=failure))
        raise
    if isinstance(result, dict) and result.get("status") == "ok":
        _end_analysis_run(recorder, run_index, payload, _RunEnd("succeeded", result))
        return _AnalysisOutcome(payload=payload, result=result, run_index=run_index)
    if isinstance(result, dict):
        message, row_result = str(result.get("error") or "compute-bound re-analysis failed"), result
    else:
        message, row_result = f"non-dict result: {type(result).__name__}", None
    failure = {"stage": "trace_analyze", "error_class": "compute_bound_reanalyze", "message": message}
    _end_analysis_run(recorder, run_index, payload, _RunEnd("failed", row_result, failure))
    return None


def _end_analysis_run(recorder: Any, run_index: int, payload: dict[str, Any], end: _RunEnd) -> None:
    """Row one trace-analysis attempt against the request it ran."""
    if recorder is not None:
        recorder.end_analysis_run(
            run_index,
            AnalysisRunOutcome(
                status=end.status,
                trace_input=str(payload["trace_input"]),
                requested_steady_state_mode=str(payload.get("steady_state_mode") or ""),
                ta_result=end.result,
                failure=end.failure,
            ),
        )


async def _preflight(recorder: Any, session_dir: Path) -> None:
    """Reap this session's orphaned servers, then record the conditions the action is about to profile under.

    An explore variant boots its server with ``cleanup=false`` to keep it hot and tears it down in a ``finally`` --
    which never runs if the driver process dies.
    """
    reaped = await _reap_session_orphans(session_dir)
    if reaped:
        log.warning("roofline: preflight reaped %d orphaned server pid(s) before profiling: %s", len(reaped), reaped)
    if recorder is not None:
        recorder.record_preflight(
            await asyncio.to_thread(_preflight_probe, session_dir, recorder.task_id, reaped=reaped)
        )


def _analysis_payload(task_params: dict[str, Any], *, trace_path: str, framework: str) -> dict[str, Any]:
    """Build the trace_analyze request for the adopted trace.

    The arm is named explicitly so neither the snapshot's ceiling precision nor the recorded workload relies on a
    transient current_best inference. Each roofline writes its own report so the PRELUDE baseline snapshot is never
    overwritten: prelude keeps the default file, close_post_opt the "after" file, every other reason a rolling one.
    """
    reason = str(task_params.get("reason") or "")
    payload: dict[str, Any] = {"trace_input": str(trace_path), "framework": framework}
    if task_params.get("workspace_path") not in (None, ""):
        payload["workspace_path"] = task_params["workspace_path"]
    payload["roofline_arm"] = roofline_arm(reason)
    output_name = {"prelude_initial": "", "close_post_opt": "kernel_roofline_opt.json"}.get(
        reason, "kernel_roofline_current.json"
    )
    if output_name:
        payload["roofline_output_name"] = output_name
    return payload


def _hot_kernels(ta_result: dict[str, Any]) -> list[Any]:
    """The hot kernels an analysis surfaced."""
    return ta_result.get("hot_kernels_top15") or ta_result.get("hot_kernels") or []


def _flag_attribution_degraded(ta_result: dict[str, Any], profile_result: dict[str, Any]) -> bool:
    """Warn on an ok analysis whose zero hot kernels are an attribution artifact; return whether it is one.

    Zero hot kernels on a trace whose health says per-kernel attribution degraded means cuda-graph capture folded
    per-kernel time into hipGraphLaunch wrappers.
    """
    trace_health = profile_result.get("trace_health") or {}
    if _hot_kernels(ta_result) or not trace_health.get("per_kernel_attribution_degraded"):
        return False
    warning = {
        "code": "cuda_graph_attribution_degraded",
        "severity": "warning",
        "message": (
            "trace_analyze returned 0 hot kernels: the profile trace "
            "has no execute_*/user_annotation events, so per-kernel "
            "device time is folded into hipGraphLaunch wrappers under "
            "cuda-graph capture (#431). Re-profile without capture (append "
            "--disable-cuda-graph to EXTRA_SGLANG_ARGS, or "
            "--compilation-config.cudagraph_mode NONE to EXTRA_VLLM_ARGS) so "
            "per-step annotations fire, or enable a capture-fold fallback over capture_traces/."
        ),
        "capture_traces_present": bool(trace_health.get("capture_traces_present")),
    }
    ta_result["trace_health_warnings"] = [*(ta_result.get("trace_health_warnings") or []), warning]
    return True


def _wants_compute_bound_reprofile(ta_result: dict[str, Any]) -> bool:
    """Whether an analysis is the multi-node host-bound shape a compute-bound re-profile exists for.

    A host-bound (high-idle) trace under PD-disagg + DP yields zero kernel candidates because the per-rank per-step
    batch is tiny.
    """
    return (
        not _hot_kernels(ta_result)
        and is_multi_node()
        and os.environ.get(_AUTO_COMPUTE_BOUND_ENV, "1").strip() != "0"
        and _trace_is_high_idle(ta_result)
    )


class RooflineExecutor:
    """Production composite ActionRunner."""

    def __init__(self, *, shared_state: Any):
        """Initialize the executor with a required SharedState reference."""
        if shared_state is None:
            raise ValueError(
                "RooflineExecutor requires a SharedState reference; "
                "construct RooflineExecutor(shared_state=...) from cli._register_executors"
            )
        self.shared_state = shared_state

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the roofline action, closing its timeline event either way."""
        from hyperloom.inference_optimizer.session.session_binding import named_session_scope

        with named_session_scope(ctx.extra["session_dir"]):
            return await self._run_recorded(ctx)

    async def _run_recorded(self, ctx: RunnerContext) -> dict[str, Any]:
        """Open the event, run the action, and close the event either way."""
        params = ctx.task.params or {}
        recorder = make_roofline_recorder(
            self._resolve_sink(ctx),
            RooflineRequest(
                task_id=str(getattr(ctx.task, "task_id", "") or ""),
                task_kind=str(getattr(ctx.task, "kind", "") or ""),
                reason=str(params.get("reason") or ""),
                framework=self._resolve_framework(ctx),
                params=params,
            ),
            owns_event=not str(params.get(INLINE_EVENT_PARAM) or ""),
        )
        try:
            result = await self._execute(ctx, recorder=recorder)
        except BaseException as exc:
            if recorder is not None:
                recorder.finish_crashed(exc)
            raise
        remember_roofline_failure(self.shared_state, params, result)
        return result

    def _resolve_sink(self, ctx: RunnerContext) -> Any:
        """Decide which event this run's rows belong to."""
        from hyperloom.inference_optimizer.breakdown.recorder.event_sink import EventSink
        from hyperloom.inference_optimizer.session.session_binding import session_is_bound

        params = ctx.task.params or {}
        if not session_is_bound():
            log.warning(
                "roofline timeline: no session bound; this action's whole event will be "
                "missing from the breakdown. The coordinator binds at startup, so this "
                "means either that never happened or the context did not name a session"
            )
            return None
        inline = str(params.get(INLINE_EVENT_PARAM) or "")
        event = inline or roofline_event_id(
            str(getattr(self.shared_state, "phase", "") or "unphased"),
            int(getattr(self.shared_state, "macro_cycle", 0) or 0),
        )
        return EventSink(event, producer=_RECORDER_PRODUCER)

    async def _execute(self, ctx: RunnerContext, *, recorder: Any) -> dict[str, Any]:
        """Run the roofline action for the given context."""
        session_dir = Path(ctx.extra["session_dir"])
        started = time.monotonic()
        self._record_lifecycle(session_dir, status="START", detail="auto-roofline: profile + TraceLens")
        arm = _ProfileArm.resolve(self._resolve_framework(ctx))
        if recorder is not None:
            recorder.begin()
        await _preflight(recorder, session_dir)

        profiled = await self._profile_until_usable(ctx, recorder, arm)
        if not isinstance(profiled, _ProfileOutcome):
            return profiled
        self._record_profile_findings(profiled)
        payload = _analysis_payload(ctx.task.params or {}, trace_path=profiled.trace_path, framework=arm.framework)

        analysis = await self._analyze(ctx, recorder, payload)
        if not isinstance(analysis, _AnalysisOutcome):
            remember_trace_failure(self.shared_state, payload, analysis)
            return analysis
        attribution_degraded = _flag_attribution_degraded(analysis.result, profiled.result)
        analysed = _AnalysedProfile(profiled, analysis)
        if _wants_compute_bound_reprofile(analysis.result):
            analysed = await self._compute_bound_reprofile(ctx, recorder, arm, analysis) or analysed
        return self._conclude(ctx, recorder, analysed, attribution_degraded=attribution_degraded, started=started)

    async def _profile_until_usable(
        self, ctx: RunnerContext, recorder: Any, arm: _ProfileArm
    ) -> _ProfileOutcome | dict[str, Any]:
        """Profile once and adopt its trace, or return the failure result.

        A failed profile is not re-run: the same configuration fails the same way, and the session degrades instead.
        """
        from .profile import profile_executor

        session_dir = Path(ctx.extra["session_dir"])
        run = recorder.begin_profile_run(attempt_reason=PROFILE_ATTEMPT_INITIAL) if recorder is not None else 0
        profile_ctx = self._wrap_profile_ctx(ctx, disable_cuda_graph=arm.disable_cuda_graph, framework=arm.framework)
        returned: Any = None
        try:
            returned = await _reported("profile", lambda: profile_executor(profile_ctx))
        except Exception as exc:  # isolation boundary: one profile sub-executor run
            log.warning("roofline: profile_executor raised", exc_info=True)
            unusable = _raised(exc)
        else:
            unusable = _classify_profile(returned)
        if unusable is None:
            return await _adopt_profile(
                recorder, run, session_dir, arm, _ProfileOutcome.of(returned, dict(profile_ctx.task.params or {}))
            )
        log.warning("roofline profile failed (%s): %s", unusable.error_class, unusable.message)
        await _end_profile_run(recorder, run, session_dir, arm, _RunEnd("failed", unusable.result, unusable.failure))
        captured = await _fail_on_capture(recorder, ctx.task, session_dir, arm, unusable)
        if captured is not None:
            return captured
        return _fail(recorder, unusable.stage, unusable.message, returned)

    def _record_profile_findings(self, profiled: _ProfileOutcome) -> None:
        """Record what the profile shows regardless of its analysis: Kineto backend, stack verdict, rewrite evidence."""
        from ._framework_rewrite_evidence import promote_evidence_path

        state = self.shared_state
        backend = detect_kineto_backend(profiled.trace_path)
        if backend:
            fingerprint = dict(state.stack_fingerprint_meta or {})
            if fingerprint.get("kineto_backend") != backend:
                fingerprint["kineto_backend"] = backend
                state.stack_fingerprint_meta = fingerprint
        unsupported = "" if state.gpu_trace_unsupported_reason else _gpu_trace_unsupported_reason(profiled.result)
        if unsupported:
            if backend:
                unsupported = f"{unsupported} (Kineto backend: {backend})"
            state.gpu_trace_unsupported_reason = unsupported
            log.error(
                "roofline: %s; automatic profile/roofline enqueues will be suppressed (stack=%s)",
                unsupported,
                state.stack_fingerprint_meta or "(unknown)",
            )
        # The host-side rewrite evidence is what the framework specialist is given instead of guessing landing points
        # from source.
        evidence_path = promote_evidence_path(state, profiled.result)
        if evidence_path:
            log.info(
                "roofline: promoted host-side rewrite evidence (%s candidate(s)) -> %s",
                profiled.result.get("framework_rewrite_candidate_count"),
                evidence_path,
            )

    async def _analyze(
        self, ctx: RunnerContext, recorder: Any, payload: dict[str, Any]
    ) -> _AnalysisOutcome | dict[str, Any]:
        """Analyse the adopted trace, re-splitting it once when TraceLens names a usable steady-state mode (N26).

        Returns the analysis the action concludes from, or the failure result.
        """
        analysis = await self._trace_analyze_once(
            ctx, recorder, payload, _AnalysisAttempt("trace_analyze", ANALYSIS_ATTEMPT_INITIAL)
        )
        if isinstance(analysis, _AnalysisOutcome) and analysis.result.get("status") != "ok":
            hint = _extract_steady_state_retry_mode(analysis.result)
            if hint is not None:
                analysis = await self._n26_retry(ctx, recorder, payload, *hint)
        if isinstance(analysis, _AnalysisOutcome) and analysis.result.get("status") != "ok":
            error = str(analysis.result.get("error") or "trace_analyze sub-step failed")
            return _fail(recorder, "trace_analyze", error, analysis.result)
        return analysis

    async def _n26_retry(
        self,
        ctx: RunnerContext,
        recorder: Any,
        payload: dict[str, Any],
        retry_mode: str,
        source_warning: dict[str, Any],
    ) -> _AnalysisOutcome | dict[str, Any]:
        """Re-split the same trace with the mode the recovery warning names and analyse it once (no re-benchmark)."""
        from_mode = source_warning.get("requested_mode") or "mixed"
        busy_ratio = source_warning.get("busy_ratio")
        threshold = source_warning.get("threshold")
        warning_code = source_warning.get("code", "")
        log.warning(
            "roofline: N26 auto-retry — adjusting steady-state window "
            "(mode %s -> %s%s); re-analyzing same trace without re-benchmarking. "
            "This is a self-healing step, NOT a failure — monitoring should "
            "expect a brief pause here.",
            from_mode,
            retry_mode,
            (
                f", busy_ratio={busy_ratio * 100:.2f}% < threshold={threshold * 100:.0f}%"
                if busy_ratio is not None and threshold is not None
                else (f", warning={warning_code}" if warning_code else "")
            ),
        )
        retry_payload = {**payload, "steady_state_mode": retry_mode}
        attempt = _AnalysisAttempt(
            "trace_analyze_n26_retry", ANALYSIS_ATTEMPT_N26_RETRY, context=f" on N26 auto-retry (mode={retry_mode})"
        )
        analysis = await self._trace_analyze_once(ctx, recorder, retry_payload, attempt)
        if isinstance(analysis, _AnalysisOutcome):
            status = analysis.result.get("status")
            log.log(
                logging.INFO if status == "ok" else logging.ERROR,
                "roofline: N26 auto-retry completed (mode %s -> %s, status=%s).",
                from_mode,
                retry_mode,
                status,
            )
            analysis.result.setdefault(
                "n26_auto_retry",
                {"applied": True, "from_mode": from_mode, "to_mode": retry_mode, "source_warning_code": warning_code},
            )
        return analysis

    async def _trace_analyze_once(
        self,
        ctx: RunnerContext,
        recorder: Any,
        payload: dict[str, Any],
        attempt: _AnalysisAttempt,
    ) -> _AnalysisOutcome | dict[str, Any]:
        """Run and row one trace analysis; return it, or the failure result when the handler produced no result.

        ``attempt.context`` names the attempt in the failure message.
        """
        from .trace_analyze import trace_analyze_handler

        session_dir = Path(ctx.extra["session_dir"])
        run_index = recorder.begin_analysis_run(attempt_reason=attempt.attempt_reason) if recorder is not None else 0
        try:
            result = await _reported(attempt.label, lambda: trace_analyze_handler(payload, session_dir=session_dir))
        except Exception as exc:  # isolation boundary: one trace analysis run
            log.warning("roofline: trace_analyze_handler raised%s", attempt.context, exc_info=True)
            error_class, message = type(exc).__name__, f"trace_analyze_handler raised{attempt.context}: {exc!r}"
        else:
            if isinstance(result, dict):
                ok = result.get("status") == "ok"
                failure = {
                    "stage": "trace_analyze",
                    "error_class": str(result.get("error_class") or ""),
                    "message": str(result.get("error") or "trace_analyze sub-step failed"),
                }
                end = _RunEnd("succeeded" if ok else "failed", result, None if ok else failure)
                _end_analysis_run(recorder, run_index, payload, end)
                return _AnalysisOutcome(payload=payload, result=result, run_index=run_index)
            error_class = "bad_return"
            message = f"trace_analyze_handler returned non-dict{attempt.context}: {type(result).__name__}"
        log.error("roofline: %s", message)
        failure = {"stage": "trace_analyze", "error_class": error_class, "message": message}
        _end_analysis_run(recorder, run_index, payload, _RunEnd("failed", failure=failure))
        return _fail(recorder, "trace_analyze", message)

    async def _compute_bound_reprofile(
        self, ctx: RunnerContext, recorder: Any, arm: _ProfileArm, analysis: _AnalysisOutcome
    ) -> _AnalysedProfile | None:
        """Re-profile a host-bound trace once with DP attention stripped; return the pair to adopt, or ``None``.

        Fail-soft: whatever goes wrong, the original analysis stands.
        """
        from ._multi_node_server_lifecycle import _COMPUTE_BOUND_PROFILE_ENV, _scoped_env

        log.info(
            "roofline: host-bound trace (0 hot kernels + high GPU idle); "
            "attempting one compute-bound re-profile (DP-attention stripped)"
        )
        adopted: _AnalysedProfile | None = None
        outcome = "re-profile produced no usable trace"
        with _scoped_env({_COMPUTE_BOUND_PROFILE_ENV: "1"}):
            try:
                adopted, outcome = await self._compute_bound_attempt(ctx, recorder, arm, analysis.payload)
            except Exception as exc:  # isolation boundary: one optional compute-bound re-profile
                outcome = f"re-profile raised: {exc!r}"
                log.warning("roofline: compute-bound re-profile failed (%s); keeping original", exc, exc_info=True)
            finally:
                if recorder is not None:
                    recorder.record_compute_bound_reprofile(attempted=True, adopted=adopted is not None, reason=outcome)
        return adopted

    async def _compute_bound_attempt(
        self, ctx: RunnerContext, recorder: Any, arm: _ProfileArm, payload: dict[str, Any]
    ) -> tuple[_AnalysedProfile | None, str]:
        """Profile and analyse once under the compute-bound override; return the pair to adopt and why."""
        session_dir = Path(ctx.extra["session_dir"])
        cb_ctx = self._wrap_profile_ctx(ctx, disable_cuda_graph=arm.disable_cuda_graph, framework=arm.framework)
        profile_run, cb_profile = await _compute_bound_profile(recorder, session_dir, arm, cb_ctx)
        cb_trace = _extract_trace_path(cb_profile)
        if not cb_trace:
            return None, "re-profile produced no usable trace"
        cb_payload = {
            "trace_input": cb_trace,
            "framework": payload["framework"],
            **{key: payload[key] for key in ("roofline_arm", "roofline_output_name") if key in payload},
        }
        analysis = await _compute_bound_analysis(recorder, session_dir, cb_payload)
        if analysis is None:
            return None, "re-profile produced no usable trace"
        cb_hot = _hot_kernels(analysis.result)
        if not cb_hot:
            log.info("roofline: compute-bound re-profile still host-bound / no hot kernels; keeping original result")
            return None, "still host-bound: re-analysis surfaced no hot kernels"
        log.info(
            "roofline: compute-bound re-profile surfaced %d hot kernel(s); adopting it for candidate dispatch",
            len(cb_hot),
        )
        params = dict(cb_ctx.task.params or {})
        if recorder is not None:
            recorder.adopt_profile_run(run_index=profile_run, profile_result=cb_profile, params=params)
        profile = _ProfileOutcome(result=cb_profile, trace_path=cb_trace, warning=None, params=params)
        return _AnalysedProfile(profile, analysis), f"adopted: surfaced {len(cb_hot)} hot kernel(s)"

    def _conclude(
        self,
        ctx: RunnerContext,
        recorder: Any,
        analysed: _AnalysedProfile,
        *,
        attribution_degraded: bool,
        started: float,
    ) -> dict[str, Any]:
        """Record the concluded pair, close the event and build the result; ``recovered`` if the trace was salvaged."""
        profile, analysis = analysed.profile, analysed.analysis
        status = "succeeded" if profile.warning is None else "recovered"
        trace_path = str(analysis.payload["trace_input"])
        hot = _hot_kernels(analysis.result)
        self.shared_state.last_profile_trace = trace_path
        self.shared_state.last_profile_launch_evidence_path = str(profile.result.get("launch_evidence_path") or "")
        self.shared_state.last_profile_status = status
        self.shared_state.record_profile_workload(profile.params, arm=analysis.payload["roofline_arm"])
        self.shared_state.record_trace_analyze(analysis.payload, analysis.result)
        cached = self.shared_state.last_trace_analyze or {}
        if recorder is not None:
            recorder.adopt_analysis_run(run_index=analysis.run_index, ta_result=analysis.result, trace_input=trace_path)
        self._record_lifecycle(
            Path(ctx.extra["session_dir"]),
            status="END",
            artifacts={
                "trace_input": trace_path,
                "analysis_md_path": str(cached.get("analysis_md_path") or ""),
                "candidates_path": str(cached.get("candidates_path") or ""),
                "kernel_roofline_path": str(cached.get("kernel_roofline_path") or ""),
            },
            detail=f"hot_kernels={len(hot)}",
            duration_s=time.monotonic() - started,
        )
        result = {
            "status": status,
            "executed_at_iso": _now_iso(),
            "snapshot_id": cached.get("roofline_snapshot_id"),
            "last_profile_trace": trace_path,
            "steady_state_trace": cached.get("steady_state_trace", ""),
            "analysis_md_path": cached.get("analysis_md_path", ""),
            "kernel_roofline_path": cached.get("kernel_roofline_path", ""),
            "profile_workspace": profile.result.get("workspace"),
            # True when trace_analyze produced 0 hot kernels because cuda-graph folding stripped per-kernel
            # attribution.
            "kernel_attribution_degraded": attribution_degraded,
            **({"profile_warning": profile.warning} if profile.warning is not None else {}),
        }
        if recorder is not None:
            snapshots = self.shared_state.roofline_snapshots
            recorder.finish_succeeded(
                RooflineConclusion(
                    snapshot_id=cached.get("roofline_snapshot_id"),
                    hot_kernel_count=len(hot),
                    kernel_attribution_degraded=attribution_degraded,
                    cached=cached,
                    trace_path=trace_path,
                    # The snapshot this run just appended. Read here rather than in the recorder because the history is
                    # capped and later runs evict entries, so the numbers have to be taken while this run is the latest.
                    snapshot=snapshots[-1] if isinstance(snapshots, list) and snapshots else None,
                )
            )
        return result

    def _record_lifecycle(self, session_dir: Path, **event: Any) -> None:
        """Emit a roofline lifecycle event and persist state.

        The auto-roofline path bypasses Coordinator._handle_request, so the action emits its own START / END pair.
        """
        try:
            record_lifecycle_event(self.shared_state, LifecycleMark(step="roofline", **event))
            if session_dir.name and session_dir.is_dir() and (session_dir / "state.json").exists():
                self.shared_state.save(session_dir)
        except OSError:
            log.debug("roofline: lifecycle %s emit failed", event.get("status"), exc_info=True)

    # Helpers (instance methods so tests can subclass / monkeypatch)
    def _resolve_framework(self, ctx: RunnerContext) -> str:
        """Resolve the active framework: task params > FRAMEWORK env > shared_state.framework."""
        params = ctx.task.params or {}
        fw = str(params.get("framework") or "").strip()
        if fw:
            return fw
        fw = os.environ.get("FRAMEWORK", "").strip()
        if fw:
            return fw
        return str(getattr(self.shared_state, "framework", "") or "").strip()

    @staticmethod
    def _wrap_profile_ctx(
        parent_ctx: RunnerContext,
        *,
        disable_cuda_graph: bool = False,
        framework: str = "",
    ) -> RunnerContext:
        """Construct a child RunnerContext for profile_executor."""
        from ...state.task_registry import Task

        parent_task = parent_ctx.task
        params = dict(parent_task.params or {})
        if disable_cuda_graph:
            from .benchmark_run.bringup import _with_cuda_graph_disabled

            params["base_extra_args"] = _with_cuda_graph_disabled(
                str(params.get("base_extra_args") or ""),
                framework or str(params.get("framework") or ""),
            )
        sub_task = Task(
            task_id=f"{parent_task.task_id}-profile",
            kind="profile",
            state="running",
            params=params,
            idempotency_key=f"{parent_task.idempotency_key}-profile",
            requires_lanes=list(parent_task.requires_lanes or []),
            lease_ttl_sec=parent_task.lease_ttl_sec,
        )
        return RunnerContext(
            task=sub_task,
            lease=parent_ctx.lease,
            extra=dict(parent_ctx.extra or {}),
        )


__all__ = [
    "RooflineExecutor",
]
