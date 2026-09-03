# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SWEEP phase handler: auto-enqueue of the concurrency-sweep task."""

from __future__ import annotations
import logging as _logging
from typing import Any
from ..state.task_registry import Task
from .base import PhaseHandler

log = _logging.getLogger(__name__)


#: Slack added to a conc_sweep lease on top of the task's own budget, covering
#: server teardown and the report flush that happen after the last variant. The
#: lease must outlive the work or the TTL watchdog fails a task that is still
#: making progress.
_CONC_SWEEP_LEASE_GRACE_SEC = 600


def _conc_sweep_lease_ttl_sec(clamped_budget: int | None) -> int:
    """Return the execution lease for a conc_sweep task, from its real budget.

    The lease has to bound the task that actually runs, which is the *clamped*
    budget — deriving it from the configured value reclaimed still-valid sweeps
    whenever the clamp produced something larger (a non-positive configured
    budget on a long session) or left the budget unbounded.

    Args:
        clamped_budget: The budget the task was enqueued with; ``None`` when the
            sweep runs without a budget gate.

    Returns:
        The lease TTL in seconds, or ``0`` for an unbounded sweep. ``0`` is the
        registry's "no lease" encoding (``reclaim_expired_running`` skips it):
        an unbounded sweep has no deadline to expire against, so opting out
        beats reclaiming it at an arbitrary one.
    """
    if clamped_budget is None or clamped_budget <= 0:
        return 0
    return int(clamped_budget) + _CONC_SWEEP_LEASE_GRACE_SEC


class SweepPhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

    async def _on_enter_sweep(self, *, from_phase: str) -> None:
        """Auto-enqueue the ``conc_sweep`` task on SWEEP entry.

        It is the phase's only action: the baseline-vs-current concurrency
        curve, on the ladder this workload sweeps.

        Args:
            from_phase: The phase being left, used only for logging.
        """
        state = self.shared_state
        # Drain pending KEEP integrates so sweep measures full current_best.
        if getattr(state, "has_keep_pending_integrate", False):
            await self._drain_pending_keep_integrates()
        # Validate the stack for positive NEEDS_REVIEW kernels.
        await self._maybe_validate_positive_needs_review_stack()
        if not getattr(state, "conc_sweep_enabled", False):
            log.info(
                "SWEEP entry (from=%s): conc_sweep disabled; recording terminal skip.",
                from_phase or "<unknown>",
            )
            self._record_terminal_conc_sweep_skip(
                skip_reason="disabled",
                auto_conc_sweep_skipped="disabled",
            )
            return
        prev_conc = getattr(state, "last_conc_sweep_watermark", None)
        prev_conc = prev_conc if isinstance(prev_conc, dict) else {}
        prev_validated = prev_conc.get("cumulative_gain_validated_at_record")
        cur_validated = float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0)
        if prev_conc and isinstance(prev_validated, (int, float)) and cur_validated <= float(prev_validated) + 1e-6:
            log.info(
                "SWEEP entry (from=%s): skipping auto-conc-sweep — no validated gain since last "
                "conc_sweep (validated=%.4f%% unchanged since %s).",
                from_phase or "<unknown>",
                cur_validated,
                prev_conc.get("ts") or "(unknown)",
            )
            self._record_terminal_conc_sweep_skip(
                skip_reason="no_validated_gain_since_last_conc_sweep",
                auto_conc_sweep_skipped="no_validated_gain_since_last_conc_sweep",
                auto_conc_sweep_skipped_validated_gain=cur_validated,
            )
            return
        denied = self._time_budget_denial_for_action("conc_sweep")
        if denied is not None:
            log.info(
                "SWEEP entry (from=%s): conc_sweep cannot fit the session budget (%s); recording terminal skip.",
                from_phase or "<unknown>",
                denied,
            )
            self._record_session_budget_conc_sweep_skip(denied=denied)
            return
        try:
            task = await self._enqueue_internal_conc_sweep_task(
                reason="phase_entry",
            )
        except Exception as exc:  # noqa: BLE001 — a failed enqueue must still close the phase
            log.exception(
                "SWEEP entry hook: failed to enqueue auto-conc-sweep: %r",
                exc,
            )
            self._record_terminal_conc_sweep_skip(
                skip_reason="enqueue_failed",
                auto_conc_sweep_error=repr(exc)[:240],
            )
            return
        # The only None is the helper's own budget decline, which records its
        # terminal skip before returning.
        if task is None:
            return
        log.info(
            "SWEEP entry (from=%s): auto-enqueued conc_sweep task=%s (concs=%s total_budget_sec=%s)",
            from_phase or "<unknown>",
            task.task_id,
            task.params.get("concs"),
            task.params.get("total_budget_sec"),
        )
        self._record_phase_entry_evidence(
            auto_conc_sweep_enqueued=True,
            auto_conc_sweep_task_id=task.task_id,
            # Verbatim: None records "the workload picks", which is not the
            # same statement as an empty ladder.
            auto_conc_sweep_concs=task.params.get("concs"),
        )

    async def _enqueue_internal_conc_sweep_task(
        self,
        *,
        reason: str,
    ) -> Task | None:
        """Build + enqueue a Coordinator-internal ``conc_sweep`` task.

        Idempotency key + PolicyGate singleton ensure at most one per SWEEP.

        Args:
            reason: Tag used in the task's idempotency key and logging.

        Returns:
            The created (or existing) ``conc_sweep`` task, or ``None`` when the
            session clock leaves nothing to spend -- which records its own
            terminal skip. An enqueue failure raises to the phase hook, which
            records the error text a swallowed one would have lost.
        """
        state = self.shared_state
        configured_budget = int(state.conc_sweep_total_budget_sec or 0)
        # Clamp total_budget_sec to the remaining session wall-clock budget so
        # a long conc_sweep cannot outlive --max-hours.  A 120 s reserve is kept
        # for the CLOSE phase.  ``None`` is the wire value for "no budget gate"
        # (no wall-clock cap, or a non-positive configured budget); a clamp that
        # leaves no time declines the task instead, because a 0 would read
        # downstream as an unbounded budget and run the whole ladder.
        _CLOSE_RESERVE_SEC = 120
        _rem_fn = getattr(state, "remaining_minutes", None)
        session_rem = _rem_fn() if callable(_rem_fn) else None
        clamped_budget: int | None
        if session_rem is not None:
            session_rem_sec = int(max(0.0, session_rem * 60.0) - _CLOSE_RESERVE_SEC)
            if session_rem_sec <= 0:
                log.info(
                    "conc_sweep: session clock leaves %ds after the %ds CLOSE reserve; not enqueueing.",
                    session_rem_sec,
                    _CLOSE_RESERVE_SEC,
                )
                self._record_session_budget_conc_sweep_skip(
                    denied=f"remaining_after_close_reserve={session_rem_sec}s",
                )
                return None
            clamped_budget = min(configured_budget, session_rem_sec) if configured_budget > 0 else session_rem_sec
        else:
            clamped_budget = configured_budget if configured_budget > 0 else None
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": str(reason),
            # None, not [], when the state carries no ladder: the executor reads
            # [] as a deliberate "no concs" and skips, while None lets it fall
            # back to the ladder for this workload.
            "concs": list(state.conc_sweep_concs) if state.conc_sweep_concs else None,
            "variant_timeout_sec": int(state.conc_sweep_variant_timeout_sec or 0),
            "total_budget_sec": clamped_budget,
        }
        task, was_existing = await self.tasks.create_or_return_existing(
            kind="conc_sweep",
            params=params,
            idempotency_key=f"internal-conc_sweep-{reason}{self._cycle_idem_suffix()}",
            lease_ttl_sec=_conc_sweep_lease_ttl_sec(clamped_budget),
        )
        if was_existing:
            log.info(
                "internal-conc_sweep task already exists (idempotent: task_id=%s, state=%s)",
                task.task_id,
                task.state,
            )
        else:
            log.info(
                "internal-conc_sweep task enqueued (task_id=%s reason=%s concs=%s total_budget_sec=%s)",
                task.task_id,
                reason,
                params["concs"],
                params["total_budget_sec"],
            )
        return task

    def _record_session_budget_conc_sweep_skip(self, *, denied: object) -> None:
        """Stamp last_conc_sweep skipped when the session clock refused conc_sweep.

        No-op when a conc_sweep result is already on the session: a later
        over-budget cancel must not erase a measurement the phase can close on.
        """
        last = getattr(self.shared_state, "last_conc_sweep", None) or {}
        if str(last.get("status") or "").strip():
            return
        self._record_terminal_conc_sweep_skip(
            skip_reason="session_time_budget",
            auto_conc_sweep_skipped="session_time_budget",
            auto_conc_sweep_denied=str(denied),
        )

    def _record_terminal_conc_sweep_skip(
        self,
        *,
        skip_reason: str,
        **evidence: Any,
    ) -> None:
        """Record an auto-conc-sweep skip as terminal so SWEEP can close cleanly."""
        self._record_phase_entry_evidence(**evidence)
        self.shared_state.record_conc_sweep(
            {
                "status": "skipped",
                "skip_reason": skip_reason,
                "was_skipped": True,
            }
        )
        self.shared_state.save(self.session_dir)
