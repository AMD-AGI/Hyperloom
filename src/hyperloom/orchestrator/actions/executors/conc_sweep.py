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

Also opens the sweep's SBD V6 timeline event. The event is opened here rather
than inside ``run_conc_sweep`` because the event id is a property of the
dispatch -- the phase and macro cycle the sweep was enqueued in -- and because
the sweep's own entry point is called directly by tests and scripts that have
no session bound and want no event.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any

from ...kernel.conc_sweep import run_conc_sweep
from ...state.shared_state import SharedState


log = logging.getLogger(__name__)

_RECORDER_PRODUCER = "orchestrator"


class ConcSweepExecutor:
    """Run the coordinator-owned concurrency sweep action."""

    async def __call__(self, ctx) -> dict[str, Any]:
        """Bind the session, run the sweep, and close its event either way.

        Args:
            ctx: Action context; ``ctx.extra['session_dir']`` is required.

        Returns:
            A result dict with a ``status`` field (and error metadata on
            failure, such as a missing ``session_dir``).
        """
        from hyperloom.inference_optimizer.session.session_binding import bound_session_or_none, session_scope

        # Only a context that names its session binds one: writing a session's
        # timeline into whatever directory the process happens to be in is
        # worse than not recording.
        named = (getattr(ctx, "extra", None) or {}).get("session_dir")
        with ExitStack() as stack:
            with suppress(Exception):
                session = Path(named).resolve() if named else None
                if session is not None and bound_session_or_none() != session:
                    stack.enter_context(session_scope(session))
            return await self._run(ctx)

    async def _run(self, ctx) -> dict[str, Any]:
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

        recorder = self._make_recorder(ctx, state)
        try:
            payload = await run_conc_sweep(
                state,
                session_dir,
                concs=concs,
                variant_timeout_sec=variant_timeout,
                total_budget_sec=total_budget,
                recorder=recorder,
            )
        except BaseException as exc:
            if recorder is not None:
                recorder.finish_crashed(exc)
            raise
        # Map run_conc_sweep's skip envelope onto the SubAgentRunner contract:
        # a skip is not an executor failure, so surface as succeeded+was_skipped.
        if payload.get("status") == "skipped":
            payload = dict(payload)
            payload["status"] = "succeeded"
            payload["was_skipped"] = True
        return payload

    def _make_recorder(self, ctx, state: SharedState) -> Any:
        """Build the recorder for this sweep, or ``None`` to record nothing.

        Args:
            ctx: The action context carrying the dispatched task.
            state (SharedState): Session state, read for the phase and macro
                cycle the event id is built from.

        Returns:
            Any: A ``ConcSweepEventRecorder``, or ``None`` when there is no
                session to record into or the event id could not be built.
        """
        from hyperloom.inference_optimizer.breakdown.recorder.conc_sweep_event import (
            conc_sweep_event_id,
            make_conc_sweep_recorder,
        )
        from hyperloom.inference_optimizer.breakdown.recorder.event_sink import make_sink
        from hyperloom.inference_optimizer.session.session_binding import session_is_bound

        params = ctx.task.params or {}
        try:
            if not session_is_bound():
                log.warning(
                    "conc_sweep timeline: no session bound; this sweep's whole event will be "
                    "missing from the breakdown. The coordinator binds at startup, so this "
                    "means either that never happened or the context did not name a session"
                )
                return None
            event = conc_sweep_event_id(
                phase=str(getattr(state, "phase", "") or "unphased"),
                macro_cycle=int(getattr(state, "macro_cycle", 0) or 0),
            )
            sink = make_sink(event, producer=_RECORDER_PRODUCER)
        except Exception:  # noqa: BLE001 — observability cannot change sweep behavior
            log.warning(
                "conc_sweep timeline: could not resolve an event to record into; this "
                "sweep's whole event will be missing from the breakdown",
                exc_info=True,
            )
            return None
        return make_conc_sweep_recorder(
            sink,
            task_id=str(getattr(ctx.task, "task_id", "") or ""),
            task_kind=str(getattr(ctx.task, "kind", "") or ""),
            reason=str(params.get("reason") or ""),
            params=params,
        )


conc_sweep_executor = ConcSweepExecutor()


__all__ = [
    "ConcSweepExecutor",
    "conc_sweep_executor",
]
