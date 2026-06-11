# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Roofline composite ActionRunner — Roofline-v2 N2 (design §8.4).

Pipeline action: orchestrates the atomic ``profile`` + ``trace_analyze``
sub-steps and produces a fresh TraceLens snapshot (``last_profile_trace`` +
``last_trace_analyze.analysis_md_text`` + monotonic ``roofline_snapshot_id``).

Design constraints (§4/§6):

* No LLM in the executor — pure orchestration; interpretation happens in the
  main Orchestration context.
* No structured ``RooflineAnalysis`` — SharedState carries verbatim
  ``analysis_md_text`` (cached by C1, kept after D1 revert).
* Atomic: both sub-steps succeed or the task fails. If profile succeeds but
  trace_analyze fails, ``last_profile_trace`` is still promoted but the
  ``last_trace_analyze`` cache stays empty.
* Bypasses SubAgentRunner: sub-steps run as plain coroutines (no double task
  accounting); the trace_path / status / args promotions trace_analyze needs
  are reproduced inline.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..sub_agent_runner import RunnerContext

log = logging.getLogger(__name__)

_PROFILE_MAX_ATTEMPTS = 3


def _now_iso() -> str:
    """Return the current UTC time as a second-precision ISO-8601 string.

    Returns:
        str: The current UTC timestamp formatted with ``timespec="seconds"``.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Auto-recover from TraceLens steady_state_chunk_* failures: when
# trace_analyze fails with one of these warnings carrying ``non_empty_modes``,
# re-issue ONCE with the first non-empty mode (from TraceLens's splitter, not
# a local heuristic). Single-retry to prevent loops.
_AUTO_RETRY_WARNING_CODES = frozenset({
    "steady_state_chunk_empty",
    "steady_state_chunk_missing",
    # low-quality chunk (non-empty but busy_ratio below threshold with a
    # better alternate); same recovery path via ``non_empty_modes``.
    "steady_state_chunk_low_quality",
})


def _extract_steady_state_retry_mode(
    ta_result: dict[str, Any],
) -> "tuple[str, dict[str, Any]] | None":
    """Inspect a failed trace_analyze result for a steady-state recovery hint.

    Returns ``(mode, warning_dict)`` when a recovery warning carries an
    alternate in ``non_empty_modes`` / ``available_modes`` (first one picked,
    splitter-sorted); ``None`` otherwise (caller falls to ``_failed()``).
    """
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
        # Both warnings name the splitter-accepted alternates
        # (non_empty_modes / available_modes).
        modes = (
            w.get("non_empty_modes")
            or w.get("available_modes")
            or []
        )
        if not isinstance(modes, list):
            continue
        for candidate in modes:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip(), w
    return None


def _extract_trace_path(profile_result: dict[str, Any]) -> str:
    """Pick the trace path like Coordinator's ``_promote_to_shared_state``:
    prefer ``main_trace_path`` (merged), else ``trace_files[0]``."""
    if not isinstance(profile_result, dict):
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


def _failed(
    phase: str,
    error: str,
    *,
    sub_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the canonical failure result dict.

    ``phase`` names the failed sub-step (profile / profile_no_trace /
    trace_analyze); ``sub_result`` is pruned to known keys for audit.
    """
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
            for k in ("status", "error", "error_class",
                      "main_trace_path", "trace_files",
                      "analysis_md_path", "hot_kernels")
            if k in sub_result
        }
    return out


class RooflineExecutor:
    """Production composite ActionRunner.

    One ``await self(ctx)`` call: run profile (failure → ``_failed`` with no
    SharedState mutation), inline-promote ``last_profile_trace`` /
    ``last_profile_status`` / ``last_profile_args``, run trace_analyze
    (failure → ``_failed`` + cleared trace_analyze cache), then on success
    ``record_trace_analyze`` (C1 path) and return snapshot_id /
    last_profile_trace / analysis_md_path for audit.
    """

    def __init__(self, *, shared_state: Any):
        """Initialize the executor with a required SharedState reference.

        Args:
            shared_state (Any): The SharedState instance the executor mutates
                (profile fields, trace_analyze cache). Must not be ``None``.

        Raises:
            ValueError: If ``shared_state`` is ``None``.
        """
        if shared_state is None:
            raise ValueError(
                "RooflineExecutor requires a SharedState reference; "
                "construct via make_roofline_executor(shared_state=...) "
                "from cli._register_executors"
            )
        self.shared_state = shared_state

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        # atom: the profile sub-step produces *.pt.trace.json.gz that
        # TraceLens consumes unchanged, so this falls through to the
        # sglang/vllm path. Lazy imports keep shell-out/yaml off module load.
        from ..kernel_request_handlers import trace_analyze_handler
        from .profile import profile_executor

        session_dir = self._resolve_session_dir(ctx)
        # #266: time the composite so the END lifecycle event reports how
        # long the auto-roofline TraceLens run took.
        _lc_t0 = time.monotonic()

        # #266: emit a paired START so the auto-roofline path (which bypasses
        # Coordinator._handle_request) does not show a lone END. Without it the
        # operator sees nothing for the whole profile-retry + TraceLens run —
        # potentially minutes — then a sudden END. Best-effort, never blocks the
        # run; the END below carries the produced artifact paths + duration.
        try:
            self.shared_state.record_lifecycle_event(
                step="roofline",
                status="START",
                detail="auto-roofline: profile + TraceLens",
            )
            _sd0 = Path(session_dir)
            if _sd0.name and _sd0.is_dir() and (_sd0 / "state.json").exists():
                self.shared_state.save(_sd0)
        except Exception:  # noqa: BLE001 — defensive
            log.debug("roofline: lifecycle START emit failed", exc_info=True)

        # ---- Step 1: profile (with retry) ------------------------------------
        # sglang's torch profiler on MI300X/ROCm is unstable: ~86% per-attempt
        # failure rate (SIGQUIT / "Profiling is not in progress" / engine init
        # crash). Retry up to _PROFILE_MAX_ATTEMPTS times; each call to
        # profile_executor manages its own server lifecycle, so a fresh attempt
        # starts with a clean profiling state.
        profile_result: dict[str, Any] | None = None
        trace_path = ""
        last_error = ""
        # Track the last failure kind so the no-trace contract is preserved
        # (profile_no_trace_failed) instead of collapsing into profile_failed.
        last_phase = "profile"
        for attempt in range(1, _PROFILE_MAX_ATTEMPTS + 1):
            profile_ctx = self._wrap_profile_ctx(ctx)
            try:
                profile_result = await profile_executor(profile_ctx)
            except Exception as exc:  # noqa: BLE001
                last_phase = "profile"
                last_error = f"profile_executor raised: {exc!r}"
                log.warning(
                    "roofline profile attempt %d/%d failed (exception): %s",
                    attempt, _PROFILE_MAX_ATTEMPTS, last_error,
                )
                continue
            if not isinstance(profile_result, dict):
                last_phase = "profile"
                last_error = (
                    f"profile_executor returned non-dict: "
                    f"{type(profile_result).__name__}"
                )
                log.warning(
                    "roofline profile attempt %d/%d failed (bad return): %s",
                    attempt, _PROFILE_MAX_ATTEMPTS, last_error,
                )
                continue
            if profile_result.get("status") != "succeeded":
                last_phase = "profile"
                last_error = str(
                    profile_result.get("error") or "profile sub-step failed"
                )
                log.warning(
                    "roofline profile attempt %d/%d failed: %s",
                    attempt, _PROFILE_MAX_ATTEMPTS, last_error,
                )
                continue
            trace_path = _extract_trace_path(profile_result)
            if not trace_path:
                last_phase = "profile_no_trace"
                last_error = (
                    "profile succeeded but no trace_path in result "
                    "(missing both main_trace_path and trace_files[0])"
                )
                log.warning(
                    "roofline profile attempt %d/%d: no trace path",
                    attempt, _PROFILE_MAX_ATTEMPTS,
                )
                continue
            # Success
            if attempt > 1:
                log.info(
                    "roofline profile succeeded on attempt %d/%d",
                    attempt, _PROFILE_MAX_ATTEMPTS,
                )
            break
        else:
            return _failed(
                last_phase,
                f"all {_PROFILE_MAX_ATTEMPTS} profile attempts failed; "
                f"last: {last_error}",
                sub_result=profile_result,
            )

        # Inline-promote only the profile fields trace_analyze needs (not
        # current_best / cumulative_gain). Do NOT clear last_trace_analyze
        # here: record_trace_analyze derives the next snapshot_id from it, so
        # clearing would reset the monotonic counter (§5/§8.7 invariant). The
        # clear happens only on the trace_analyze failure path below.
        self.shared_state.last_profile_trace = str(trace_path)
        self.shared_state.last_profile_status = "succeeded"
        self.shared_state.last_profile_args = str(
            (ctx.task.params or {}).get("base_extra_args") or ""
        )

        # ---- Step 2: trace_analyze ----------------------------------------
        ta_payload: dict[str, Any] = {"trace_input": str(trace_path)}
        try:
            ta_result = await trace_analyze_handler(
                ta_payload,
                session_dir=session_dir,
            )
        except Exception as exc:  # noqa: BLE001
            # Stale-cache invariant: clear the cache so the prompt shows
            # "(no TraceLens snapshot yet)" instead of advice tied to the
            # previous trace.
            self.shared_state.last_trace_analyze = {}
            return _failed("trace_analyze", f"trace_analyze_handler raised: {exc!r}")
        if not isinstance(ta_result, dict):
            self.shared_state.last_trace_analyze = {}
            return _failed(
                "trace_analyze",
                f"trace_analyze_handler returned non-dict: {type(ta_result).__name__}",
            )

        # auto-retry: on a recovery warning naming an alternate mode the
        # splitter can serve, re-issue ONCE with that mode rather than
        # failing (single-retry enforced by the handler idempotency key + the
        # local gate below).
        retry_hint: "tuple[str, dict[str, Any]] | None" = None
        if ta_result.get("status") != "ok":
            retry_hint = _extract_steady_state_retry_mode(ta_result)
        if retry_hint is not None:
            retry_mode, source_warning = retry_hint
            ta_payload_retry: dict[str, Any] = {
                "trace_input": str(trace_path),
                "steady_state_mode": retry_mode,
                # Marker against retry loops; single-retry is enforced by not
                # re-entering this block regardless of the second outcome.
                "_n26_auto_retry": True,
                "_n26_retry_from_mode": (
                    source_warning.get("requested_mode") or ""
                ),
            }
            try:
                ta_result = await trace_analyze_handler(
                    ta_payload_retry,
                    session_dir=session_dir,
                )
            except Exception as exc:  # noqa: BLE001
                self.shared_state.last_trace_analyze = {}
                return _failed(
                    "trace_analyze",
                    (
                        f"trace_analyze_handler raised on N26 auto-retry "
                        f"(mode={retry_mode}): {exc!r}"
                    ),
                )
            if not isinstance(ta_result, dict):
                self.shared_state.last_trace_analyze = {}
                return _failed(
                    "trace_analyze",
                    (
                        f"trace_analyze_handler returned non-dict on N26 "
                        f"auto-retry (mode={retry_mode}): "
                        f"{type(ta_result).__name__}"
                    ),
                )
            # Stamp ``n26_auto_retry`` so the recorder / prompt surface
            # "snapshot came from an auto-retry" (best-effort).
            if isinstance(ta_result, dict):
                ta_result.setdefault("n26_auto_retry", {
                    "applied": True,
                    "from_mode": (
                        source_warning.get("requested_mode") or "mixed"
                    ),
                    "to_mode": retry_mode,
                    "source_warning_code": source_warning.get("code"),
                })

        if ta_result.get("status") != "ok":
            self.shared_state.last_trace_analyze = {}
            return _failed(
                "trace_analyze",
                str(ta_result.get("error") or "trace_analyze sub-step failed"),
                sub_result=ta_result,
            )

        # #431: trace_analyze status=ok but ZERO hot kernels means cuda-graph capture folded per-kernel time into hipGraphLaunch wrappers (degraded input, not a TraceLens failure). Append a trace_health_warnings entry so the LLM re-profiles in eager mode instead of reading top=[] as "no kernels".
        hot = (
            ta_result.get("hot_kernels_top15")
            or ta_result.get("hot_kernels")
            or []
        )
        trace_health = profile_result.get("trace_health") or {}
        attribution_degraded = bool(
            not hot and trace_health.get("per_kernel_attribution_degraded")
        )
        if attribution_degraded:
            warning = {
                "code": "cuda_graph_attribution_degraded",
                "severity": "warning",
                "message": (
                    "trace_analyze returned 0 hot kernels: the profile trace "
                    "has no execute_*/user_annotation events, so per-kernel "
                    "device time is folded into hipGraphLaunch wrappers under "
                    "cuda-graph capture (#431). Re-profile in eager mode "
                    "(append --enforce-eager to EXTRA_SGLANG_ARGS / "
                    "EXTRA_VLLM_ARGS) so per-step annotations fire, or enable "
                    "a capture-fold fallback over capture_traces/."
                ),
                "capture_traces_present": bool(
                    trace_health.get("capture_traces_present")
                ),
            }
            health = list(ta_result.get("trace_health_warnings") or [])
            health.append(warning)
            ta_result["trace_health_warnings"] = health

        # Cache via the C1 recorder (bumps roofline_snapshot_id by one, writes analysis_md_text / analysis_md_path).
        self.shared_state.record_trace_analyze(ta_payload, ta_result)
        cached = self.shared_state.last_trace_analyze or {}

        # #266: the auto-roofline TraceLens run does NOT pass through
        # Coordinator._handle_request, so emit its lifecycle event here so
        # operators still see "TraceLens finished -> analysis at <path>".
        # Best-effort. The event is recorded into the coordinator's shared
        # SharedState object, so it is durable as soon as ANY later
        # coordinator save runs; the explicit save below is only a fast-path
        # to flush it immediately when a real session dir already exists
        # (tests may resolve session_dir to "."). Note: if auto-roofline ever
        # became the very first writer of state.json, this in-memory event
        # would rely on that later coordinator save to reach disk.
        try:
            self.shared_state.record_lifecycle_event(
                step="roofline",
                status="END",
                artifacts={
                    "trace_input": str(trace_path),
                    "analysis_md_path": str(cached.get("analysis_md_path") or ""),
                    "candidates_path": str(cached.get("candidates_path") or ""),
                    "kernel_roofline_path": str(
                        cached.get("kernel_roofline_path") or ""
                    ),
                },
                detail=f"hot_kernels={len(hot)}",
                duration_s=time.monotonic() - _lc_t0,
            )
            sd = Path(session_dir)
            if sd.name and sd.is_dir() and (sd / "state.json").exists():
                self.shared_state.save(sd)
        except Exception:  # noqa: BLE001 — defensive
            log.debug("roofline: lifecycle emit failed", exc_info=True)

        return {
            "status": "succeeded",
            "executed_at_iso": _now_iso(),
            "snapshot_id": cached.get("roofline_snapshot_id"),
            "last_profile_trace": str(trace_path),
            "analysis_md_path": cached.get("analysis_md_path", ""),
            "kernel_roofline_path": cached.get("kernel_roofline_path", ""),
            "profile_workspace": profile_result.get("workspace"),
            # #431: False on a healthy run; True when trace_analyze produced
            # 0 hot kernels because cuda-graph folding stripped per-kernel
            # attribution (a ``cuda_graph_attribution_degraded`` entry was
            # appended to trace_health_warnings for the prompt/audit).
            "kernel_attribution_degraded": attribution_degraded,
        }

    # Helpers (instance methods so tests can subclass / monkeypatch)
    @staticmethod
    def _resolve_session_dir(ctx: RunnerContext) -> Path:
        """Resolve the session directory from the runner context.

        Args:
            ctx (RunnerContext): The runner context whose ``extra`` may carry a
                ``session_dir`` entry.

        Returns:
            Path: The configured session directory, or ``Path(".")`` when none
                is present.
        """
        sd = ctx.extra.get("session_dir") if ctx.extra else None
        return Path(sd) if sd else Path(".")

    @staticmethod
    def _wrap_profile_ctx(parent_ctx: RunnerContext) -> RunnerContext:
        """Construct a child RunnerContext for profile_executor.

        Bypasses SubAgentRunner's child Task creation (avoids double task
        accounting); the child carries kind="profile" + same params and
        inherits the lease (no profile_lane re-acquire).
        """
        from ..task_registry import Task
        parent_task = parent_ctx.task
        sub_task = Task(
            task_id=f"{parent_task.task_id}-profile",
            kind="profile",
            state="running",
            params=dict(parent_task.params or {}),
            idempotency_key=f"{parent_task.idempotency_key}-profile",
            requires_lanes=list(parent_task.requires_lanes or []),
            allowed_tools=list(parent_task.allowed_tools or []),
            side_effects=list(parent_task.side_effects or []),
            lease_ttl_sec=parent_task.lease_ttl_sec,
        )
        return RunnerContext(
            task=sub_task,
            lease=parent_ctx.lease,
            extra=dict(parent_ctx.extra or {}),
        )


def make_roofline_executor(*, shared_state: Any) -> RooflineExecutor:
    """Production factory used by `cli._register_executors`."""
    return RooflineExecutor(shared_state=shared_state)


__all__ = [
    "RooflineExecutor",
    "make_roofline_executor",
]
