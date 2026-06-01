"""ActionRunner for the ``conc_sweep`` SWEEP-phase action.

Thin shell around ``orchestrator.conc_sweep.run_conc_sweep`` -- the
pure-Python aggregator + per-variant Magpie loop -- so the same
SubAgentRunner dispatch path that handles ``sweep`` / ``baseline``
can also drive the post-sweep concurrency comparison.

The Coordinator auto-enqueues exactly one ``conc_sweep`` task per
SWEEP phase via ``_enqueue_internal_conc_sweep_task`` right after
the SWEEP-entry sweep task completes (and only when
``shared_state.conc_sweep_enabled`` is True). LLM-proposed
``delegate{action_name='conc_sweep'}`` is denied by PolicyGate.

Inputs (``task.params`` populated by the coordinator):

* ``concs``                 list[int]   — CONC ladder
* ``variant_timeout_sec``   int         — per-variant Magpie timeout
* ``total_budget_sec``      int         — total wall-clock budget
                                          (0 disables the gate)

The executor reloads ``SharedState`` from ``ctx.extra['session_dir']``
to pick up the live ``current_best`` / ``baseline_tput`` / ``isl`` /
``osl`` / ``baseline_config_path`` etc. — these change every tick
and would be stale if pinned at enqueue time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..conc_sweep import run_conc_sweep
from ..shared_state import SharedState


log = logging.getLogger(__name__)


class ConcSweepExecutor:
    """ActionRunner for ``conc_sweep``. See module docstring."""

    async def __call__(self, ctx) -> dict[str, Any]:
        extra = getattr(ctx, "extra", None) or {}
        session_dir_str = str(extra.get("session_dir") or "").strip()
        if not session_dir_str:
            return {
                "status":      "failed",
                "error_class": "missing_session_dir",
                "error":       "conc_sweep: ctx.extra['session_dir'] missing",
            }
        session_dir = Path(session_dir_str)
        try:
            state = SharedState.load_or_init(session_dir)
        except Exception as exc:  # noqa: BLE001 — surface as failure
            return {
                "status":      "failed",
                "error_class": "shared_state_load_failed",
                "error":       f"conc_sweep: SharedState.load_or_init failed: {exc!r}",
            }

        params = ctx.task.params or {}
        # ``None`` lets run_conc_sweep fall back to its DEFAULT_CONCS.
        # Empty list intentionally short-circuits via empty_conc_list
        # so the operator's explicit "no concs" choice is respected.
        concs_raw = params.get("concs")
        if concs_raw is None:
            concs: list[int] | None = (
                list(state.conc_sweep_concs) if state.conc_sweep_concs else None
            )
        else:
            concs = [int(c) for c in concs_raw]

        variant_timeout = int(
            params.get("variant_timeout_sec")
            or state.conc_sweep_variant_timeout_sec
            or 1800
        )
        total_budget = int(
            params.get("total_budget_sec", state.conc_sweep_total_budget_sec)
        )

        payload = await run_conc_sweep(
            state, session_dir,
            concs=concs,
            variant_timeout_sec=variant_timeout,
            total_budget_sec=total_budget,
        )
        # Map run_conc_sweep's skip envelope onto the SubAgentRunner
        # contract: skips are NOT executor failures (no benchmark
        # crashed; the precondition simply wasn't met). The Coordinator
        # records them via the standard ``record_action_attempt`` /
        # decision='discarded' path the same way an empty sweep grid
        # would be.
        if payload.get("status") == "skipped":
            payload = dict(payload)
            payload["status"] = "succeeded"
            payload["was_skipped"] = True
        return payload


conc_sweep_executor = ConcSweepExecutor()


__all__ = [
    "ConcSweepExecutor",
    "conc_sweep_executor",
]
