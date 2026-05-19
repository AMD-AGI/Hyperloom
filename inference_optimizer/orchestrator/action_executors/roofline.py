"""Roofline composite ActionRunner — Roofline-v2 (design/roofline-v2.md §8.4).

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
  carries the verbatim `analysis_md_text` (cached by C1 / kept after
  D1 revert) and the main LLM reads it directly.
* Atomic semantics — either both sub-steps succeed (cache written,
  task succeeded) or the whole task fails. Partial state (profile
  succeeded but trace_analyze failed) leaves `last_profile_trace`
  populated (profile artifact is still useful) but does **not**
  pretend trace_analyze succeeded.

**N2a (this commit) ships a stub** that registers the action surface
so the main LLM can already `propose_action{roofline}` without
"no_executor" failures and so the C1-cached `analysis.md` /
`roofline_snapshot_id` invariants are observable in cli wiring tests.

The stub returns `status="succeeded"` with `degraded=True` and a
diagnostic `error` field instead of running the real sub-steps —
the actual `profile + trace_analyze` orchestration lands in **N2b**
once we have the sub-task-promote contract worked out (see
design/roofline-v2.md §7.3 sub-commit split principle).

**N2b will replace** `RooflineStubExecutor.__call__` with the
production orchestration logic that:

1. Invokes `profile_executor` against a child `RunnerContext` and
   waits for `status=succeeded`.
2. Manually promotes profile fields (`last_profile_trace`, etc.) on
   `SharedState` since we are bypassing the standard
   `Coordinator._promote_to_shared_state` path.
3. Invokes `trace_analyze_handler` directly with
   `payload={"trace_input": last_profile_trace}` and waits for
   `status="ok"`.
4. Calls `SharedState.record_trace_analyze(payload, result)` to cache
   the report (re-using the C1 mechanism untouched by D1 revert).
5. Returns a status dict carrying the new `roofline_snapshot_id`
   plus pointers to the profile workspace and the analysis.md path.

`RooflineStubExecutor.__call__` keeps the **exact same return-dict
shape** that N2b will produce (modulo `degraded=True` + `error` in
the stub) so prompt rendering (N5) and audit (N7) can be wired
against the stub before N2b lands.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..sub_agent_runner import RunnerContext


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RooflineStubExecutor:
    """N2a stub — registers `roofline` action without running real
    profile / trace_analyze sub-steps.

    N2b will subclass / replace this with the production executor.
    The stub holds a `shared_state` reference (same constructor
    signature as the future `RooflineExecutor`) so cli wiring is
    stable across N2a → N2b.
    """

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
    """N2a factory used by `cli._register_executors`.

    Same call-site signature N2b's `make_roofline_executor` will
    expose, so swapping stub → real is a one-line change in cli.py."""
    return RooflineStubExecutor(shared_state=shared_state)


__all__ = [
    "RooflineStubExecutor",
    "make_roofline_stub_executor",
]
