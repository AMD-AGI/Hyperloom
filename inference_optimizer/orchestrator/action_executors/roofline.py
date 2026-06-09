# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Roofline composite ActionRunner — Roofline-v2 N2 (design §8.4).

`roofline` is a macro / pipeline action: its executor orchestrates the
two atomic sub-steps `profile` and `trace_analyze` and produces a
fresh TraceLens snapshot (`last_profile_trace` +
`last_trace_analyze.analysis_md_text` + monotonic
`roofline_snapshot_id`).

**Design constraints** (per §4 / §6 of the design doc):

* No LLM is invoked inside the executor — `roofline` is pure
  orchestration. Any "interpret the report" work happens in the main
  Orchestration LLM context where `analysis.md` full text is rendered
  into the prompt by §8.7's `_format_analysis_md_full`.
* No structured `RooflineAnalysis` dict is written — `SharedState`
  carries the verbatim `analysis_md_text` (cached by C1, kept after
  D1 revert) and the main LLM reads it directly.
* Atomic semantics — either both sub-steps succeed (cache written,
  task succeeded) or the whole task fails with `status="failed"`. If
  profile succeeds but trace_analyze fails, `last_profile_trace` is
  still promoted (profile artifact is independently useful) but
  `last_trace_analyze` cache stays empty (the C1 path that writes it
  is never reached).
* Bypasses SubAgentRunner for sub-step execution. profile_executor
  and trace_analyze_handler are invoked as plain coroutines so we
  avoid two layers of task accounting / lease re-acquisition. The
  shared_state mutations that Coordinator._promote_to_shared_state
  would normally do for a top-level profile task are reproduced
  inline (limited to the trace_path / status / args fields that
  downstream trace_analyze depends on).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..sub_agent_runner import RunnerContext


log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# auto-recover from TraceLens steady_state_chunk_* failures
# ---------------------------------------------------------------------------
# When trace_analyze fails with a structured
# steady_state_chunk_{empty,missing} warning carrying ``non_empty_modes``,
# re-issue trace_analyze ONCE with the first non-empty mode. The alternate
# mode comes directly from TraceLens's splitter (execution_details.csv),
# not an inference_optimizer-side busy-% heuristic.
#
# Single-retry contract: prevents infinite loops. If the retry also
# fails (or no alternate mode was offered) we propagate the original
# failure unchanged so existing error-handling paths still apply.
_AUTO_RETRY_WARNING_CODES = frozenset({
    "steady_state_chunk_empty",
    "steady_state_chunk_missing",
    # low-quality chunk (non-empty but busy_ratio below threshold AND a
    # materially-higher-busy_ratio alternate exists). Same recovery
    # path; alternate mode comes from the warning's ``non_empty_modes``
    # list, populated by the
    # tracelens_analysis._check_selected_chunk_has_gpu_events_quality gate.
    "steady_state_chunk_low_quality",
})


def _extract_steady_state_retry_mode(
    ta_result: dict[str, Any],
) -> "tuple[str, dict[str, Any]] | None":
    """Inspect a failed trace_analyze result for a structured
    steady-state recovery hint from tracelens_analysis.py.

    Returns ``(mode, warning_dict)`` when the result carries a
    ``steady_state_chunk_empty`` or ``steady_state_chunk_missing``
    warning with at least one alternate mode in ``non_empty_modes``
    (or ``available_modes``, used by the missing-chunk warning).
    Returns ``None`` otherwise -- caller falls through to the
    existing _failed() path.

    The first alternate is picked (TraceLens splitter sorts them
    deterministically via dict-iter order set in
    ``_check_selected_chunk_has_gpu_events``). If you have a use case
    for picking a different one (e.g. operator prefers decode_only
    when both decode_only and prefilldecode are non-empty) we can
    revisit; for now first-non-empty is the simplest correct policy.
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
        # `steady_state_chunk_empty` carries `non_empty_modes`.
        # `steady_state_chunk_missing` carries `available_modes`.
        # Both name the alternates the splitter would accept.
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


# ---------------------------------------------------------------------------
# N2b real executor — orchestrates profile + trace_analyze
# ---------------------------------------------------------------------------
def _extract_trace_path(profile_result: dict[str, Any]) -> str:
    """Pick the trace path the way Coordinator does in
    ``_promote_to_shared_state`` (profile branch).

    Prefer ``main_trace_path`` (merged trace, what TraceLens wants);
    fall back to the first entry of ``trace_files``."""
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

    Phase identifies which sub-step failed: `profile` /
    `profile_no_trace` (profile succeeded but no trace path) /
    `trace_analyze`. `sub_result` carries the sub-step's raw result
    for audit; pruned to known keys to keep result size bounded.
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

    Lifecycle of one ``await self(ctx)`` call:

    1. Resolve session_dir from ``ctx.extra``.
    2. Construct a child `RunnerContext` for the profile sub-step
       (kind="profile") sharing the parent task's params / lease /
       extra so profile_executor sees the expected workspace and
       config. SubAgentRunner has already acquired the
       ``profile_lane`` lease for the parent roofline task (yaml
       declares it), so the child re-uses that lease without
       re-acquiring.
    3. Invoke `profile_executor` and check `status`. On failure
       return `_failed("profile", ...)` immediately; SharedState is
       not mutated.
    4. On profile success, extract `trace_path` and inline-promote
       `last_profile_trace` / `last_profile_status` /
       `last_profile_args` (mirroring the relevant subset of
       Coordinator._promote_to_shared_state's profile branch). Also
       clear `last_trace_analyze` so the stale-cache invariant in
       §5 / §8.5 holds.
    5. Invoke `trace_analyze_handler` (the kernel request handler
       renamed by N1) with `payload={"trace_input": trace_path}`.
       On failure return `_failed("trace_analyze", ...)`. SharedState
       retains the newly-set profile fields but NOT a fresh
       trace_analyze cache.
    6. On trace_analyze success, call
       `SharedState.record_trace_analyze(payload, result)` — this is
       the C1 path (preserved through D1) which writes
       `analysis_md_text` / `roofline_snapshot_id` etc.
    7. Return a status dict carrying `snapshot_id` /
       `last_profile_trace` / `analysis_md_path` so audit (N7) can
       cross-reference the produced snapshot with the LLM-visible
       cache.
    """

    def __init__(self, *, shared_state: Any):
        if shared_state is None:
            raise ValueError(
                "RooflineExecutor requires a SharedState reference; "
                "construct via make_roofline_executor(shared_state=...) "
                "from cli._register_executors"
            )
        self.shared_state = shared_state

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        # atom: the Magpie atom wrapper bridges PROFILE=1 to atom's
        # --torch-profiler-dir CLI flag, so the composite's profile
        # sub-step succeeds and the trace_analyze sub-step consumes the
        # resulting *.pt.trace.json.gz unchanged (TraceLens is
        # framework-agnostic). The executor falls through to the same
        # path as sglang/vllm.
        # Lazy imports avoid pulling shell-out / yaml machinery at
        # module load time (consistent with how the BaselineExecutor
        # subclass is constructed lazily by cli).
        from ..kernel_request_handlers import trace_analyze_handler
        from .profile import profile_executor

        session_dir = self._resolve_session_dir(ctx)
        # #266: time the composite so the END lifecycle event reports how
        # long the auto-roofline TraceLens run took.
        _lc_t0 = time.monotonic()

        # ---- Step 1: profile ------------------------------------------------
        profile_ctx = self._wrap_profile_ctx(ctx)
        try:
            profile_result = await profile_executor(profile_ctx)
        except Exception as exc:  # noqa: BLE001 — surface sub-step errors
            return _failed("profile", f"profile_executor raised: {exc!r}")
        if not isinstance(profile_result, dict):
            return _failed(
                "profile",
                f"profile_executor returned non-dict: {type(profile_result).__name__}",
            )
        if profile_result.get("status") != "succeeded":
            return _failed(
                "profile",
                str(profile_result.get("error") or "profile sub-step failed"),
                sub_result=profile_result,
            )
        trace_path = _extract_trace_path(profile_result)
        if not trace_path:
            return _failed(
                "profile_no_trace",
                "profile succeeded but no trace_path in result "
                "(missing both main_trace_path and trace_files[0])",
                sub_result=profile_result,
            )

        # Inline promote — replicate the subset of
        # Coordinator._promote_to_shared_state profile branch that
        # downstream trace_analyze depends on. We intentionally do
        # NOT touch current_best / cumulative_gain here (those are
        # not required by trace_analyze and are still set later if /
        # when the post-completion audit runs the standard promote
        # path on the outer roofline task).
        #
        # We do NOT clear last_trace_analyze yet — that's done only on
        # the trace_analyze failure path below. Reason: record_trace_analyze
        # in step 2 derives the new roofline_snapshot_id by reading the
        # previous snapshot_id from last_trace_analyze and incrementing
        # by one. Clearing here would reset the counter to 0 every time
        # roofline runs, breaking the §5 / §8.7 single-monotonic-snapshot
        # invariant the prompt re-profile guidance depends on.
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
            # Stale cache invariant: failure leaves last_profile_trace
            # pointing at the new trace but no fresh analysis_md_text
            # for it. Clear the cache so the prompt renderer shows
            # "(no TraceLens snapshot yet)" instead of stale advice
            # tied to the previous trace.
            self.shared_state.last_trace_analyze = {}
            return _failed("trace_analyze", f"trace_analyze_handler raised: {exc!r}")
        if not isinstance(ta_result, dict):
            self.shared_state.last_trace_analyze = {}
            return _failed(
                "trace_analyze",
                f"trace_analyze_handler returned non-dict: {type(ta_result).__name__}",
            )

        # auto-retry: when trace_analyze failed AND the failure
        # carries a steady_state_chunk_{empty,missing} warning with
        # `non_empty_modes`, the TraceLens splitter is telling us
        # exactly which alternate mode it CAN serve. Re-issue ONCE
        # with that mode rather than bubbling a failure up to the
        # operator. We never retry more than once (idempotency_key
        # carries the retry flag so a second auto-retry on the same
        # action would be denied at the handler boundary; we also
        # gate locally below).
        retry_hint: "tuple[str, dict[str, Any]] | None" = None
        if ta_result.get("status") != "ok":
            retry_hint = _extract_steady_state_retry_mode(ta_result)
        if retry_hint is not None:
            retry_mode, source_warning = retry_hint
            ta_payload_retry: dict[str, Any] = {
                "trace_input": str(trace_path),
                "steady_state_mode": retry_mode,
                # Marker so a second iteration of this block (if any
                # downstream code ever reaches here twice) does not
                # cascade into a retry loop. The actual single-retry
                # invariant is enforced by NOT re-entering the
                # `if retry_hint is not None` block below regardless
                # of the second attempt's outcome.
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
            # Stamp the result so the recorder / prompt renderer can
            # surface "this snapshot came from an auto-retry" to the
            # LLM (helps it self-document any subsequent explore
            # decisions). Stamping is best-effort; field naming is
            # under `n26_auto_retry` to keep it discoverable in
            # SharedState dumps.
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

        # #431: trace_analyze can succeed (status=ok) yet return ZERO hot
        # kernels when CUDA/HIP-graph capture folds per-kernel device
        # activity into hipGraphLaunch wrappers, stripping the
        # execute_*/user_annotation events trace_analyze needs to attribute
        # kernels. That is NOT a TraceLens failure (so the N26 status!="ok"
        # retry above never fires) — it is a degraded-input signal. We
        # append a ``trace_health_warnings`` entry: that is the existing
        # channel ``record_trace_analyze`` persists into last_trace_analyze
        # and the orchestration prompt renders as ``warnings=[...]`` so the
        # LLM grounds its next action (re-profile in eager mode, or a future
        # capture-fold fallback over capture_traces/) instead of reading
        # ``top=[]`` as "no optimizable kernels in this trace".
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

        # Cache the trace_analyze result via existing C1 recorder.
        # The recorder bumps roofline_snapshot_id by one against the
        # previous snapshot (kept intact above) and writes
        # analysis_md_text / analysis_md_path.
        self.shared_state.record_trace_analyze(ta_payload, ta_result)
        cached = self.shared_state.last_trace_analyze or {}

        # #266: the auto-roofline TraceLens run does NOT pass through
        # Coordinator._handle_request, so emit its lifecycle event here so
        # operators still see "TraceLens finished -> analysis at <path>".
        # Best-effort; persisted only when running against a real session
        # dir (tests may resolve session_dir to ".").
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

    # ------------------------------------------------------------------
    # Helpers (instance methods so tests can subclass / monkeypatch)
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_session_dir(ctx: RunnerContext) -> Path:
        sd = ctx.extra.get("session_dir") if ctx.extra else None
        return Path(sd) if sd else Path(".")

    @staticmethod
    def _wrap_profile_ctx(parent_ctx: RunnerContext) -> RunnerContext:
        """Construct a child RunnerContext for profile_executor.

        SubAgentRunner normally creates the child Task + workspace,
        but we bypass that to avoid two layers of task accounting.
        The child Task carries kind="profile" + same params so
        BaselineExecutor (profile_executor's parent class) finds
        its config; lease is inherited so we don't re-acquire the
        profile_lane that SubAgentRunner already grabbed for the
        outer roofline task.
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
