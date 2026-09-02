# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ActionRunner for the ``conc_sweep`` SWEEP-phase action.

Thin shell around ``orchestrator.kernel.conc_sweep.run_conc_sweep``. The
Coordinator auto-enqueues one ``conc_sweep`` task per SWEEP phase via
``_enqueue_internal_conc_sweep_task`` (when ``conc_sweep_enabled``, which is
on by default; disable via ``--no-enable-conc-sweep``); a LLM-proposed
``conc_sweep`` delegate is denied by PolicyGate.

Inputs (``task.params``): ``concs`` (CONC ladder), ``variant_timeout_sec``,
``total_budget_sec`` (``None`` disables the gate; ``<=0`` means no time is left
and the sweep skips without booting a server).

Reloads ``SharedState`` from ``ctx.extra['session_dir']`` to pick up the
live current_best / baseline_tput / isl / osl / baseline_config_path,
which would be stale if pinned at enqueue time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...kernel.conc_sweep import run_conc_sweep
from ...state.shared_state import SharedState


class ConcSweepExecutor:
    """ActionRunner for ``conc_sweep``. See module docstring."""

    async def __call__(self, ctx) -> dict[str, Any]:
        """Run the concurrency sweep action for the given context.

        Args:
            ctx: Action context; ``ctx.extra['session_dir']`` is required.

        Returns:
            A result dict with a ``status`` field (and error metadata on
            failure, such as a missing ``session_dir``).
        """
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
        # ``None`` falls back to the ladder run_conc_sweep resolves for this
        # workload; an empty list short-circuits (respects an explicit "no
        # concs" choice).
        concs_raw = params.get("concs")
        if concs_raw is None:
            concs: list[int] | None = list(state.conc_sweep_concs) if state.conc_sweep_concs else None
        else:
            concs = [int(c) for c in concs_raw]

        variant_timeout = int(params.get("variant_timeout_sec") or state.conc_sweep_variant_timeout_sec or 1800)
        # An explicit ``None`` means "no budget gate" and must survive as None:
        # coercing it to 0 would instead read as "no time left" and skip.
        budget_raw = params.get("total_budget_sec", state.conc_sweep_total_budget_sec)
        total_budget = None if budget_raw is None else int(budget_raw)

        payload = await run_conc_sweep(
            state,
            session_dir,
            concs=concs,
            variant_timeout_sec=variant_timeout,
            total_budget_sec=total_budget,
        )
        # Map run_conc_sweep's skip envelope onto the SubAgentRunner contract:
        # a skip is not an executor failure, so surface as succeeded+was_skipped.
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
