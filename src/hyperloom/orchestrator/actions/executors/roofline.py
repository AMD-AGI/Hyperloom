# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Roofline composite ActionRunner.

Pipeline action: orchestrates the atomic ``profile`` + ``trace_analyze``
sub-steps and produces a fresh TraceLens snapshot (``last_profile_trace`` +
``last_trace_analyze.analysis_md_text`` + monotonic ``roofline_snapshot_id``).

Design constraints:

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

import asyncio
import functools
import logging
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from hyperloom.common.timeutil import now_iso
from ...loop.sub_agent_runner import RunnerContext
from ...trace.task_progress import report_progress
from ._multi_node_env import is_multi_node

log = logging.getLogger(__name__)

_PROFILE_MAX_ATTEMPTS = 3

# Settle time after reclaiming GPUs before the next profile attempt. A SIGKILLed
# server's VRAM is not returned by the KFD the instant the process dies, so an
# immediate retry can still read the old free-memory figure and refuse to boot.
_GPU_RECLAIM_SETTLE_S = 20.0

# Env switch for the multi-node compute-bound auto re-profile (default on; set to
# "0" to disable). Only ever consulted on multi-node runs.
_AUTO_COMPUTE_BOUND_ENV = "HYPERLOOM_PROFILE_AUTO_COMPUTE_BOUND"


async def _reclaim_gpus_for_retry(session_dir: Path | str, *, attempt: int) -> None:
    """Free GPUs held by an orphaned server before the next profile attempt.

    The profile retry loop otherwise changes nothing between attempts, so a
    boot that failed because the GPUs were already occupied fails identically
    on every attempt — three refusals inside three minutes, all against a
    squatter that is still very much alive.

    Two best-effort stages, cheapest first:

    1. ``reap_orphaned_servers`` on this session's own ``runs/`` pidfiles. This
       is cmdline-gated and scoped to pidfiles this session wrote, so it cannot
       touch a co-located session. It is the common case: ``explore`` boots
       variant servers with ``cleanup=false`` to keep them hot, and its
       ``finally`` teardown never runs if the driver dies first.
    2. ``kill_stale_gpu_owners`` — recover's soft-cleanup stage, for a squatter
       whose pidfile is already gone.

    Never raises: reclaiming is an optimisation on the retry path, and a failure
    here must not mask the underlying profile error.

    Args:
        session_dir: Session directory whose ``runs/`` subtree owns the pidfiles.
        attempt: The attempt that just failed, for log correlation.
    """
    from ._server_lifecycle import reap_orphaned_servers
    from .recover import kill_stale_gpu_owners, probe_gpu_free_mb

    try:
        reaped = await asyncio.to_thread(reap_orphaned_servers, Path(session_dir))
    except Exception:  # noqa: BLE001 — best-effort
        log.debug("roofline: orphan reap failed", exc_info=True)
        reaped = []
    try:
        killed = await asyncio.to_thread(kill_stale_gpu_owners)
    except Exception:  # noqa: BLE001 — best-effort
        log.debug("roofline: stale-owner cleanup failed", exc_info=True)
        killed = []

    if not reaped and not killed:
        log.warning(
            "roofline: attempt %d hit insufficient GPU memory but found no "
            "reclaimable owner; the VRAM belongs to something outside this "
            "session and retrying will not help",
            attempt,
        )
        return

    log.warning(
        "roofline: attempt %d hit insufficient GPU memory; reaped=%s killed=%s; settling %.0fs before retry",
        attempt,
        reaped,
        [p.get("pid") for p in killed],
        _GPU_RECLAIM_SETTLE_S,
    )
    await asyncio.sleep(_GPU_RECLAIM_SETTLE_S)
    try:
        free_mb = await asyncio.to_thread(probe_gpu_free_mb)
        log.info("roofline: post-reclaim free VRAM: %s", free_mb)
    except Exception:  # noqa: BLE001 — best-effort
        log.debug("roofline: post-reclaim probe failed", exc_info=True)


def _trace_is_high_idle(ta_result: dict[str, Any]) -> bool:
    """Whether trace_analyze flagged the profiled step as host-bound (high GPU
    idle), i.e. carries a ``high_gpu_idle_pct`` trace-health warning.

    Args:
        ta_result: The trace_analyze result dict.

    Returns:
        bool: True when a high-GPU-idle warning is present.
    """
    if not isinstance(ta_result, dict):
        return False
    for w in ta_result.get("trace_health_warnings") or []:
        if isinstance(w, dict) and w.get("code") == "high_gpu_idle_pct":
            return True
    return False


# seconds + ``+00:00`` (canonical helper; kept importable for callers).
_now_iso = functools.partial(now_iso, "seconds")


# Auto-recover from TraceLens steady_state_chunk_* failures: re-issue ONCE with
# the first non-empty mode from the warning's ``non_empty_modes``.
_AUTO_RETRY_WARNING_CODES = frozenset(
    {
        "steady_state_chunk_empty",
        "steady_state_chunk_missing",
        # low-quality chunk; same recovery path via ``non_empty_modes``.
        "steady_state_chunk_low_quality",
    }
)


def _extract_steady_state_retry_mode(
    ta_result: dict[str, Any],
) -> "tuple[str, dict[str, Any]] | None":
    """Inspect a failed trace_analyze result for a steady-state recovery hint.

    Args:
        ta_result: The failed trace_analyze result to inspect for warnings.

    Returns:
        ``(mode, warning_dict)`` when a recovery warning carries an alternate
        in ``non_empty_modes`` / ``available_modes`` (first one picked,
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
        # Splitter-accepted alternates (non_empty_modes / available_modes).
        modes = w.get("non_empty_modes") or w.get("available_modes") or []
        if not isinstance(modes, list):
            continue
        for candidate in modes:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip(), w
    return None


def _extract_trace_path(profile_result: dict[str, Any]) -> str:
    """Pick the trace path like Coordinator's ``_promote_to_shared_state``:
    prefer ``main_trace_path`` (merged), else ``trace_files[0]``.

    Args:
        profile_result: The profile sub-step result to read the trace from.

    Returns:
        The resolved trace path, or an empty string if none is present.
    """
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

    Args:
        phase: Name of the failed sub-step.
        error: Human-readable error message.
        sub_result: The failed sub-step's result, pruned to known keys for
            audit when provided.

    Returns:
        The canonical failure result dict.
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
    """Flatten a profile result's error fields into one blob for cuda-graph
    capture-failure detection."""
    if not isinstance(profile_result, dict):
        return ""
    parts = [str(profile_result.get(k) or "") for k in ("error", "error_class", "error_excerpt", "stderr_tail")]
    sub = profile_result.get("sub_result")
    if isinstance(sub, dict):
        parts += [str(sub.get(k) or "") for k in ("error", "error_class")]
    return "\n".join(parts)


def _profile_server_log_tail(profile_result: Any, max_bytes: int = 16384) -> str:
    """Return the tail of the newest engine ``server.log`` for a profile run.

    Some crashes surface only in the engine ``server.log`` (not the profile
    result fields), so it feeds the cuda-graph capture-failure detector.
    Best-effort: returns "" on any miss.
    """
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
        """Run the roofline action for the given context.

        Performs a profile sub-step and feeds the resulting trace into
        TraceLens analysis to produce a roofline characterization.

        Args:
            ctx: Runner context carrying the task and session metadata.

        Returns:
            A result dict describing the roofline outcome and artifacts.
        """
        # atom: the profile sub-step produces *.pt.trace.json.gz that TraceLens
        # consumes unchanged. Lazy imports keep shell-out/yaml off module load.
        from ...kernel.request_handlers import trace_analyze_handler
        from .profile import profile_executor

        # Every sub-step below goes through this, so a call site added later
        # cannot silently be the one that reports nothing. Reported on entry,
        # not on completion: a sub-step that never returns is exactly the case
        # the heartbeat has to be able to show.
        async def _reported(
            label: str,
            start: Callable[[], Awaitable[Any]],
            **fields: Any,
        ) -> Any:
            """Announce a roofline sub-step, then await it.

            Args:
                label (str): Sub-step name carried on the progress note.
                start (Callable[[], Awaitable[Any]]): Zero-argument factory
                    returning the sub-step awaitable.
                **fields (Any): Extra note fields, e.g. ``index`` / ``total``.

            Returns:
                Any: Whatever the sub-step returned, unchanged.
            """
            await report_progress(
                unit="roofline_step",
                label=label,
                status="started",
                **fields,
            )
            return await start()

        session_dir = self._resolve_session_dir(ctx)
        # Time the composite so the END lifecycle event reports its duration.
        _lc_t0 = time.monotonic()

        # Emit a paired START so the auto-roofline path (which bypasses
        # Coordinator._handle_request) does not show a lone END. Best-effort.
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

        # ---- Profile (with retry) --------------------------------------------
        # sglang's torch profiler on MI300X/ROCm is unstable, so retry up to
        # _PROFILE_MAX_ATTEMPTS times; each profile_executor call manages its own
        # server lifecycle so a fresh attempt starts clean.
        profile_result: dict[str, Any] | None = None
        trace_path = ""
        last_error = ""
        profile_warning: dict[str, Any] | None = None
        successful_profile_params: dict[str, Any] = {}
        # Track the last failure kind so the no-trace contract is preserved
        # (profile_no_trace_failed) instead of collapsing into profile_failed.
        last_phase = "profile"
        # After a cuda-graph capture crash the next attempt boots eager so the
        # torch-profiler stream capture cannot collide. Operators can arm it
        # upfront too: the profiler cannot decompose cuda-graph replay, so a
        # graph-mode trace reports near-zero GPU time and trips the idle gate.
        disable_cuda_graph = os.environ.get("HYPERLOOM_PROFILE_DISABLE_CUDA_GRAPH", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        # Resolve framework so the eager fallback picks the correct flag (vLLM
        # --enforce-eager, sglang --disable-cuda-graph).
        framework = self._resolve_framework(ctx)
        from .baseline import (
            _disable_cuda_graph_flag,
            _is_cuda_graph_capture_failure,
            _is_insufficient_gpu_memory,
        )

        eager_flag = _disable_cuda_graph_flag(framework)

        # Preflight: an explore variant boots its server with ``cleanup=false``
        # to keep it hot and tears it down in a ``finally`` — which never runs
        # if the driver process dies. Nothing else clears the pidfile, because
        # reap_orphaned_servers is called only at coordinator startup/resume, so
        # the stale server keeps the VRAM until the NEXT session boots. Reap
        # this session's own orphans before asking for eight GPUs.
        #
        # Safe to do unconditionally here: roofline holds ``profile_lane``,
        # which resource_lock declares mutually exclusive with
        # ``benchmark_lane`` / ``server_lifecycle`` / ``gpu_research_lane``. No
        # task in this session may legitimately be holding a server while we
        # run, so any surviving pidfile is by definition an orphan. The reap is
        # additionally scoped to this session's own ``runs/`` pidfiles and gated
        # on a cmdline match, so a co-located session is never touched.
        from ._server_lifecycle import reap_orphaned_servers

        try:
            _pre_reaped = await asyncio.to_thread(reap_orphaned_servers, Path(session_dir))
            if _pre_reaped:
                log.warning(
                    "roofline: preflight reaped %d orphaned server pid(s) before profiling: %s",
                    len(_pre_reaped),
                    _pre_reaped,
                )
        except Exception:  # noqa: BLE001 — best-effort
            log.debug("roofline: preflight orphan reap failed", exc_info=True)

        for attempt in range(1, _PROFILE_MAX_ATTEMPTS + 1):
            profile_ctx = self._wrap_profile_ctx(
                ctx,
                disable_cuda_graph=disable_cuda_graph,
                framework=framework,
            )
            try:
                profile_result = await _reported(
                    "profile",
                    lambda: profile_executor(profile_ctx),
                    index=attempt,
                    total=_PROFILE_MAX_ATTEMPTS,
                )
            except Exception as exc:  # noqa: BLE001
                last_phase = "profile"
                last_error = f"profile_executor raised: {exc!r}"
                log.warning(
                    "roofline profile attempt %d/%d failed (exception): %s",
                    attempt,
                    _PROFILE_MAX_ATTEMPTS,
                    last_error,
                )
                if not disable_cuda_graph and _is_cuda_graph_capture_failure(last_error):
                    disable_cuda_graph = True
                    log.warning(
                        "roofline: cuda-graph capture failure detected; next attempt boots eager (%s)",
                        eager_flag,
                    )
                elif attempt < _PROFILE_MAX_ATTEMPTS and _is_insufficient_gpu_memory(last_error):
                    await _reclaim_gpus_for_retry(session_dir, attempt=attempt)
                continue
            if not isinstance(profile_result, dict):
                last_phase = "profile"
                last_error = f"profile_executor returned non-dict: {type(profile_result).__name__}"
                log.warning(
                    "roofline profile attempt %d/%d failed (bad return): %s",
                    attempt,
                    _PROFILE_MAX_ATTEMPTS,
                    last_error,
                )
                continue
            trace_path = _extract_trace_path(profile_result)
            if profile_result.get("status") != "succeeded":
                if trace_path:
                    # A duplicate stop_profile failure can arrive after a trace
                    # was already flushed successfully.
                    profile_warning = {
                        "status": profile_result.get("status"),
                        "error_class": profile_result.get("error_class"),
                        "error": profile_result.get("error"),
                    }
                    log.warning(
                        "roofline profile attempt %d/%d returned status=%r "
                        "but produced trace=%s; continuing to trace_analyze",
                        attempt,
                        _PROFILE_MAX_ATTEMPTS,
                        profile_result.get("status"),
                        trace_path,
                    )
                    successful_profile_params = dict(profile_ctx.task.params or {})
                    break
                last_phase = "profile"
                last_error = str(profile_result.get("error") or "profile sub-step failed")
                log.warning(
                    "roofline profile attempt %d/%d failed: %s",
                    attempt,
                    _PROFILE_MAX_ATTEMPTS,
                    last_error,
                )
                if not disable_cuda_graph and _is_cuda_graph_capture_failure(
                    last_error,
                    _profile_err_text(profile_result),
                    _profile_server_log_tail(profile_result),
                ):
                    disable_cuda_graph = True
                    log.warning(
                        "roofline: cuda-graph capture failure detected; next attempt boots eager (%s)",
                        eager_flag,
                    )
                elif attempt < _PROFILE_MAX_ATTEMPTS and _is_insufficient_gpu_memory(
                    last_error,
                    _profile_err_text(profile_result),
                    _profile_server_log_tail(profile_result),
                ):
                    await _reclaim_gpus_for_retry(session_dir, attempt=attempt)
                continue
            if not trace_path:
                last_phase = "profile_no_trace"
                last_error = (
                    "profile succeeded but no trace_path in result (missing both main_trace_path and trace_files[0])"
                )
                log.warning(
                    "roofline profile attempt %d/%d: no trace path",
                    attempt,
                    _PROFILE_MAX_ATTEMPTS,
                )
                continue
            # A capture-only profile yielded only CUDA-graph capture sidecars
            # (no annotated steady-state trace). This is transient, so re-profile
            # with the SAME graph-capture settings (do NOT escalate to eager,
            # which changes the measured workload). If every attempt stays
            # capture-only, fail with an accurate message.
            if profile_result.get("profile_trace_selection_reason") == "capture_only_fallback":
                last_phase = "profile_capture_only"
                last_error = (
                    "profile produced only CUDA-graph capture sidecars under "
                    "capture_traces/ (no annotated steady-state trace); the "
                    "steady-state splitter cannot use these — re-profile needed"
                )
                log.warning(
                    "roofline profile attempt %d/%d: capture-only trace "
                    "(%s); re-profiling (same graph-capture settings)",
                    attempt,
                    _PROFILE_MAX_ATTEMPTS,
                    trace_path,
                )
                continue
            # Op count == 0: the torch-profiler active window captured no ops
            # (metadata-only trace). Re-profile rather than cache an empty
            # snapshot; if every attempt is zero-ops, fail with an accurate message.
            if bool((profile_result.get("trace_health") or {}).get("zero_ops")):
                last_phase = "profile_zero_ops"
                last_error = (
                    "profile produced a metadata-only trace (PyTorch Profiler "
                    "Op count == 0); the active capture window never overlapped "
                    "execution — re-profile needed"
                )
                log.warning(
                    "roofline profile attempt %d/%d: zero-ops trace (%s); re-profiling",
                    attempt,
                    _PROFILE_MAX_ATTEMPTS,
                    trace_path,
                )
                continue
            # Success
            if attempt > 1:
                log.info(
                    "roofline profile succeeded on attempt %d/%d",
                    attempt,
                    _PROFILE_MAX_ATTEMPTS,
                )
            successful_profile_params = dict(profile_ctx.task.params or {})
            break
        else:
            return _failed(
                last_phase,
                f"all {_PROFILE_MAX_ATTEMPTS} profile attempts failed; last: {last_error}",
                sub_result=profile_result,
            )

        # Resolve the profiled arm explicitly so neither the snapshot's ceiling
        # precision nor the recorded workload relies on a transient current_best
        # inference: PRELUDE measures the baseline arm; all other reasons
        # measure current_best.
        _task_params = ctx.task.params or {}
        _reason = str(_task_params.get("reason") or "")
        roofline_arm = "baseline" if _reason == "prelude_initial" else "current_best"

        # Inline-promote only the profile fields trace_analyze needs. Do NOT
        # clear last_trace_analyze here: record_trace_analyze derives the next
        # snapshot_id from it. The clear happens only on the failure path below.
        self.shared_state.last_profile_trace = str(trace_path)
        self.shared_state.last_profile_status = "succeeded"
        self.shared_state.record_profile_workload(
            successful_profile_params or ctx.task.params or {},
            arm=roofline_arm,
        )
        # The host-side rewrite evidence is produced by the profile sub-step and is
        # what the framework specialist is given instead of guessing landing points
        # from source. It is independent of the trace: no kernel timeline can say
        # which host-side work is redundant.
        from ._framework_rewrite_evidence import promote_evidence_path

        _evidence_path = promote_evidence_path(self.shared_state, profile_result)
        if _evidence_path:
            log.info(
                "roofline: promoted host-side rewrite evidence (%s candidate(s)) -> %s",
                profile_result.get("framework_rewrite_candidate_count"),
                _evidence_path,
            )

        # ---- trace_analyze -------------------------------------------------
        # Route each roofline to its own report so the PRELUDE baseline snapshot
        # is never overwritten: prelude keeps the default file, close_post_opt
        # writes the "after" file, every other reason writes a rolling current one.
        if _reason == "prelude_initial":
            roofline_output_name = ""
        elif _reason == "close_post_opt":
            roofline_output_name = "kernel_roofline_opt.json"
        else:
            roofline_output_name = "kernel_roofline_current.json"
        ta_payload: dict[str, Any] = {
            "trace_input": str(trace_path),
            "framework": framework,
        }
        workspace_path = _task_params.get("workspace_path")
        if workspace_path not in (None, ""):
            ta_payload["workspace_path"] = workspace_path
        if roofline_arm:
            ta_payload["roofline_arm"] = roofline_arm
        if roofline_output_name:
            ta_payload["roofline_output_name"] = roofline_output_name
        try:
            ta_result = await _reported(
                "trace_analyze",
                lambda: trace_analyze_handler(ta_payload, session_dir=session_dir),
            )
        except Exception as exc:  # noqa: BLE001
            # Clear the cache so the prompt shows no snapshot rather than advice
            # tied to the previous trace.
            self.shared_state.last_trace_analyze = {}
            return _failed("trace_analyze", f"trace_analyze_handler raised: {exc!r}")
        if not isinstance(ta_result, dict):
            self.shared_state.last_trace_analyze = {}
            return _failed(
                "trace_analyze",
                f"trace_analyze_handler returned non-dict: {type(ta_result).__name__}",
            )

        # N26 auto-retry: on a recovery warning naming an alternate mode, re-split
        # the SAME trace with that mode and re-issue trace_analyze ONCE (no
        # re-benchmark). Single-retry is enforced by the handler idempotency key
        # + the local gate below.
        retry_hint: "tuple[str, dict[str, Any]] | None" = None
        if ta_result.get("status") != "ok":
            retry_hint = _extract_steady_state_retry_mode(ta_result)
        if retry_hint is not None:
            retry_mode, source_warning = retry_hint
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
            ta_payload_retry: dict[str, Any] = {
                "trace_input": str(trace_path),
                "framework": framework,
                "steady_state_mode": retry_mode,
                # Marker against retry loops.
                "_n26_auto_retry": True,
                "_n26_retry_from_mode": from_mode,
            }
            if "workspace_path" in ta_payload:
                ta_payload_retry["workspace_path"] = ta_payload["workspace_path"]
            if roofline_arm:
                ta_payload_retry["roofline_arm"] = roofline_arm
            if roofline_output_name:
                ta_payload_retry["roofline_output_name"] = roofline_output_name
            try:
                ta_result = await _reported(
                    "trace_analyze_n26_retry",
                    lambda: trace_analyze_handler(ta_payload_retry, session_dir=session_dir),
                )
            except Exception as exc:  # noqa: BLE001
                self.shared_state.last_trace_analyze = {}
                log.error(
                    "roofline: N26 auto-retry FAILED (mode=%s): %r — "
                    "this is a genuine trace_analyze failure after window adjustment.",
                    retry_mode,
                    exc,
                )
                return _failed(
                    "trace_analyze",
                    (f"trace_analyze_handler raised on N26 auto-retry (mode={retry_mode}): {exc!r}"),
                )
            if not isinstance(ta_result, dict):
                self.shared_state.last_trace_analyze = {}
                log.error(
                    "roofline: N26 auto-retry returned non-dict (mode=%s, type=%s) — "
                    "genuine failure after window adjustment.",
                    retry_mode,
                    type(ta_result).__name__,
                )
                return _failed(
                    "trace_analyze",
                    (
                        f"trace_analyze_handler returned non-dict on N26 "
                        f"auto-retry (mode={retry_mode}): "
                        f"{type(ta_result).__name__}"
                    ),
                )
            retry_ok = ta_result.get("status") == "ok"
            log.info(
                "roofline: N26 auto-retry completed (mode %s -> %s, status=%s).",
                from_mode,
                retry_mode,
                ta_result.get("status"),
            )
            if not retry_ok:
                log.error(
                    "roofline: N26 auto-retry status=%s — genuine failure "
                    "after window adjustment (mode=%s); not a hang.",
                    ta_result.get("status"),
                    retry_mode,
                )
            # Stamp ``n26_auto_retry`` for the recorder / prompt (best-effort).
            if isinstance(ta_result, dict):
                ta_result.setdefault(
                    "n26_auto_retry",
                    {
                        "applied": True,
                        "from_mode": from_mode,
                        "to_mode": retry_mode,
                        "source_warning_code": warning_code,
                    },
                )

        if ta_result.get("status") != "ok":
            self.shared_state.last_trace_analyze = {}
            return _failed(
                "trace_analyze",
                str(ta_result.get("error") or "trace_analyze sub-step failed"),
                sub_result=ta_result,
            )

        # status=ok but ZERO hot kernels means cuda-graph capture folded
        # per-kernel time into hipGraphLaunch wrappers. Append a warning so the
        # LLM re-profiles in eager mode instead of reading top=[] as "no kernels".
        hot = ta_result.get("hot_kernels_top15") or ta_result.get("hot_kernels") or []
        trace_health = profile_result.get("trace_health") or {}
        attribution_degraded = bool(not hot and trace_health.get("per_kernel_attribution_degraded"))
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
                "capture_traces_present": bool(trace_health.get("capture_traces_present")),
            }
            health = list(ta_result.get("trace_health_warnings") or [])
            health.append(warning)
            ta_result["trace_health_warnings"] = health

        # Multi-node compute-bound auto re-profile: a host-bound (high-idle)
        # trace under PD-disagg + DP yields zero kernel candidates because the
        # per-rank per-step batch is tiny. Re-profile ONCE with DP-attention /
        # dp-size stripped (single DP rank at full per-step batch => compute-
        # bound) so kernel candidates can surface. Candidates are still validated
        # on the real served config downstream. Multi-node only, single-shot,
        # fail-soft (any error keeps the original host-bound result).
        if (
            not hot
            and is_multi_node()
            and os.environ.get(_AUTO_COMPUTE_BOUND_ENV, "1").strip() != "0"
            and _trace_is_high_idle(ta_result)
        ):
            from ._multi_node_server_lifecycle import _COMPUTE_BOUND_PROFILE_ENV

            log.info(
                "roofline: host-bound trace (0 hot kernels + high GPU idle); "
                "attempting one compute-bound re-profile (DP-attention stripped)"
            )
            _prev_cb = os.environ.get(_COMPUTE_BOUND_PROFILE_ENV)
            os.environ[_COMPUTE_BOUND_PROFILE_ENV] = "1"
            try:
                cb_ctx = self._wrap_profile_ctx(ctx, framework=framework)
                cb_profile = await _reported(
                    "profile_compute_bound",
                    lambda: profile_executor(cb_ctx),
                )
                cb_trace = _extract_trace_path(cb_profile) if isinstance(cb_profile, dict) else ""
                if cb_trace:
                    cb_payload: dict[str, Any] = {
                        "trace_input": str(cb_trace),
                        "framework": framework,
                    }
                    if roofline_arm:
                        cb_payload["roofline_arm"] = roofline_arm
                    if roofline_output_name:
                        cb_payload["roofline_output_name"] = roofline_output_name
                    cb_ta = await _reported(
                        "trace_analyze_compute_bound",
                        lambda: trace_analyze_handler(cb_payload, session_dir=session_dir),
                    )
                    if isinstance(cb_ta, dict) and cb_ta.get("status") == "ok":
                        cb_hot = cb_ta.get("hot_kernels_top15") or cb_ta.get("hot_kernels") or []
                        if cb_hot:
                            log.info(
                                "roofline: compute-bound re-profile surfaced %d hot "
                                "kernel(s); adopting it for candidate dispatch",
                                len(cb_hot),
                            )
                            ta_result = cb_ta
                            ta_payload = cb_payload
                            hot = cb_hot
                            trace_path = cb_trace
                            successful_profile_params = dict(cb_ctx.task.params or {})
                            self.shared_state.last_profile_trace = str(cb_trace)
                            self.shared_state.record_profile_workload(
                                successful_profile_params,
                                arm=roofline_arm,
                            )
                        else:
                            log.info(
                                "roofline: compute-bound re-profile still host-bound "
                                "/ no hot kernels; keeping original result"
                            )
            except Exception as exc:  # noqa: BLE001 — fail-soft
                log.warning(
                    "roofline: compute-bound re-profile failed (%s); keeping original",
                    exc,
                )
            finally:
                if _prev_cb is None:
                    os.environ.pop(_COMPUTE_BOUND_PROFILE_ENV, None)
                else:
                    os.environ[_COMPUTE_BOUND_PROFILE_ENV] = _prev_cb

        # Cache via the C1 recorder (bumps roofline_snapshot_id by one, writes analysis_md_text / analysis_md_path).
        self.shared_state.record_trace_analyze(ta_payload, ta_result)
        cached = self.shared_state.last_trace_analyze or {}

        # The auto-roofline TraceLens run does NOT pass through
        # Coordinator._handle_request, so emit its lifecycle event here.
        # Best-effort; the explicit save below is a fast-path to flush it when a
        # real session dir already exists.
        try:
            self.shared_state.record_lifecycle_event(
                step="roofline",
                status="END",
                artifacts={
                    "trace_input": str(trace_path),
                    "analysis_md_path": str(cached.get("analysis_md_path") or ""),
                    "candidates_path": str(cached.get("candidates_path") or ""),
                    "kernel_roofline_path": str(cached.get("kernel_roofline_path") or ""),
                },
                detail=f"hot_kernels={len(hot)}",
                duration_s=time.monotonic() - _lc_t0,
            )
            sd = Path(session_dir)
            if sd.name and sd.is_dir() and (sd / "state.json").exists():
                self.shared_state.save(sd)
        except Exception:  # noqa: BLE001 — defensive
            log.debug("roofline: lifecycle emit failed", exc_info=True)

        result = {
            "status": "succeeded",
            "executed_at_iso": _now_iso(),
            "snapshot_id": cached.get("roofline_snapshot_id"),
            "last_profile_trace": str(trace_path),
            "steady_state_trace": cached.get("steady_state_trace", ""),
            "analysis_md_path": cached.get("analysis_md_path", ""),
            "kernel_roofline_path": cached.get("kernel_roofline_path", ""),
            "profile_workspace": profile_result.get("workspace"),
            # True when trace_analyze produced 0 hot kernels because cuda-graph
            # folding stripped per-kernel attribution.
            "kernel_attribution_degraded": attribution_degraded,
        }
        if profile_warning is not None:
            result["profile_recovered"] = True
            result["profile_warning"] = profile_warning
        return result

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

    def _resolve_framework(self, ctx: RunnerContext) -> str:
        """Resolve the active framework: task params > FRAMEWORK env >
        shared_state.framework.
        """
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
        """Construct a child RunnerContext for profile_executor.

        Bypasses SubAgentRunner's child Task creation; the child carries
        kind="profile" + same params and inherits the lease. When
        ``disable_cuda_graph`` is set, the framework-correct eager flag is folded
        into ``base_extra_args``.

        Args:
            parent_ctx: The parent runner context to derive the child from.
            disable_cuda_graph: Whether to inject the eager fallback flag.
            framework: Framework name used to choose the correct eager flag.

        Returns:
            A child ``RunnerContext`` for the profile sub-step.
        """
        from ...state.task_registry import Task

        parent_task = parent_ctx.task
        params = dict(parent_task.params or {})
        if disable_cuda_graph:
            from .baseline import _with_cuda_graph_disabled

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
            side_effects=list(parent_task.side_effects or []),
            lease_ttl_sec=parent_task.lease_ttl_sec,
        )
        return RunnerContext(
            task=sub_task,
            lease=parent_ctx.lease,
            extra=dict(parent_ctx.extra or {}),
        )


def make_roofline_executor(*, shared_state: Any) -> RooflineExecutor:
    """Production factory used by `cli._register_executors`.

    Args:
        shared_state: The SharedState instance the executor will mutate.

    Returns:
        A configured ``RooflineExecutor``.
    """
    return RooflineExecutor(shared_state=shared_state)


__all__ = [
    "RooflineExecutor",
    "make_roofline_executor",
]
