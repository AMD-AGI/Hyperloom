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

N2a stub kept (`RooflineStubExecutor` + `make_roofline_stub_executor`)
as the §11 risk-mitigation fallback wiring: operators who want to
temporarily disable the real executor without removing the action
entry can wire the stub instead.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..sub_agent_runner import RunnerContext


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Stub (N2a) — kept for the §11 fallback wiring path
# ---------------------------------------------------------------------------
class RooflineStubExecutor:
    """Stub executor used as a fallback when the real `RooflineExecutor`
    must be disabled (operator override / debug scenarios). Returns
    `succeeded` + `degraded=True` with diagnostic `error` field."""

    def __init__(self, *, shared_state: Any = None):
        self.shared_state = shared_state

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        snapshot_id = 0
        analysis_md_path = ""
        last_profile_trace = ""
        if self.shared_state is not None:
            cached = (
                getattr(self.shared_state, "last_trace_analyze", {}) or {}
            )
            snap_raw = cached.get("roofline_snapshot_id")
            if isinstance(snap_raw, int):
                snapshot_id = snap_raw
            analysis_md_path = str(cached.get("analysis_md_path") or "")
            last_profile_trace = str(
                getattr(self.shared_state, "last_profile_trace", "") or ""
            )
        return {
            "status": "succeeded",
            "degraded": True,
            "error": "roofline_stub_executor_active",
            "executed_at_iso": _now_iso(),
            "snapshot_id": snapshot_id,
            "last_profile_trace": last_profile_trace,
            "analysis_md_path": analysis_md_path,
        }


def make_roofline_stub_executor(
    *, shared_state: Any = None,
) -> RooflineStubExecutor:
    """Stub factory — kept as the explicit safe-fallback wiring path."""
    return RooflineStubExecutor(shared_state=shared_state)


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
        # Lazy imports avoid pulling shell-out / yaml machinery at
        # module load time (consistent with how the BaselineExecutor
        # subclass is constructed lazily by cli).
        from ..kernel_request_handlers import trace_analyze_handler
        from .profile import profile_executor

        session_dir = self._resolve_session_dir(ctx)

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
            # for it. Clear the cache so the prompt renderer (N5) shows
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
        if ta_result.get("status") != "ok":
            self.shared_state.last_trace_analyze = {}
            return _failed(
                "trace_analyze",
                str(ta_result.get("error") or "trace_analyze sub-step failed"),
                sub_result=ta_result,
            )

        # Cache the trace_analyze result via existing C1 recorder.
        # This writes analysis_md_text / roofline_snapshot_id /
        # roofline_baseline_gain_at_snapshot etc. The recorder reads
        # the previous snapshot_id from last_trace_analyze (kept
        # intact above) and bumps it by one, giving us the
        # monotonically-increasing snapshot counter.
        self.shared_state.record_trace_analyze(ta_payload, ta_result)
        cached = self.shared_state.last_trace_analyze or {}

        return {
            "status": "succeeded",
            "executed_at_iso": _now_iso(),
            "snapshot_id": cached.get("roofline_snapshot_id"),
            "last_profile_trace": str(trace_path),
            "analysis_md_path": cached.get("analysis_md_path", ""),
            "profile_workspace": profile_result.get("workspace"),
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
    """Production factory used by `cli._register_executors`.

    Same call-site signature `make_roofline_stub_executor` exposes so
    swapping stub → real is a one-line change in cli.py."""
    return RooflineExecutor(shared_state=shared_state)


__all__ = [
    "RooflineExecutor",
    "RooflineStubExecutor",
    "make_roofline_executor",
    "make_roofline_stub_executor",
]
