# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ActionRunner for the ``conc_sweep`` SWEEP-phase action."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...kernel.conc_sweep import run_conc_sweep
from ...state.shared_state import SharedState


class ConcSweepExecutor:
    """Run the coordinator-owned concurrency sweep action."""

    async def __call__(self, ctx) -> dict[str, Any]:
        """Run the concurrency sweep action for the given context."""
        extra = getattr(ctx, "extra", None) or {}
        session_dir_str = str(extra.get("session_dir") or "").strip()
        if not session_dir_str:
            return {
                "status": "failed",
                "error_class": "missing_session_dir",
                "error": "conc_sweep: ctx.extra['session_dir'] missing",
            }
        session_dir = Path(session_dir_str)
        try:
            state = SharedState.load_or_init(session_dir)
        except Exception as exc:  # noqa: BLE001 — surface as failure
            return {
                "status": "failed",
                "error_class": "shared_state_load_failed",
                "error": f"conc_sweep: SharedState.load_or_init failed: {exc!r}",
            }

        params = ctx.task.params or {}
        # ``None`` falls back to the ladder run_conc_sweep resolves for this workload; an empty list short-circuits
        # (respects an explicit "no concs" choice).
        concs_raw = params.get("concs")
        if concs_raw is None:
            concs: list[int] | None = list(state.conc_sweep_concs) if state.conc_sweep_concs else None
        else:
            concs = [int(c) for c in concs_raw]

        variant_timeout = int(params.get("variant_timeout_sec") or state.conc_sweep_variant_timeout_sec or 1800)
        # An explicit ``None`` means "no budget gate" and must survive as None: coercing it to 0 would instead read as
        # "no time left" and skip.
        budget_raw = params.get("total_budget_sec", state.conc_sweep_total_budget_sec)
        total_budget = None if budget_raw is None else int(budget_raw)

        payload = await run_conc_sweep(
            state,
            session_dir,
            concs=concs,
            variant_timeout_sec=variant_timeout,
            total_budget_sec=total_budget,
        )
        # Map run_conc_sweep's skip envelope onto the SubAgentRunner contract: a skip is not an executor failure, so
        # surface as succeeded+was_skipped.
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
