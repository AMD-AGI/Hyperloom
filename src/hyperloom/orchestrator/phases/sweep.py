# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SWEEP phase handler: auto-enqueue of the concurrency-sweep task."""

from __future__ import annotations
import logging as _logging
import sqlite3
from typing import TYPE_CHECKING, Any
from ..kernel import kernel_decisions
from ..state.task_registry import TERMINAL_STATES, Task, TaskSpec
from ..state.orchestration_memory import CYCLE_DIRECTIVE_REQUEST, build_cycle_memory
from ..collaborator import CoordinatorCollaborator
from . import machine_state as _phase_state

if TYPE_CHECKING:
    from .machine import Transition

log = _logging.getLogger(__name__)


#: Slack added to a conc_sweep lease on top of the task's own budget, covering
#: server teardown and the report flush that happen after the last variant, so
#: the lease and the phase's grant to the sweep outlive the work.
_CONC_SWEEP_LEASE_GRACE_SEC = 600

#: The ``from_phase`` and reason of the entry a resumed session replays, as KERNEL's resume re-entry names it.
_RESUME = "resume"


def _conc_sweep_lease_ttl_sec(clamped_budget: int | None) -> int:
    """Return the execution lease for a conc_sweep task, from its real budget."""
    if clamped_budget is None or clamped_budget <= 0:
        return 0
    return int(clamped_budget) + _CONC_SWEEP_LEASE_GRACE_SEC


class SweepPhase(CoordinatorCollaborator):
    """SWEEP phase handler: drives concurrency sweep and roofline analysis."""

    #: Set once this process has checked a resumed SWEEP for a sweep that left no outcome.
    _resume_checked: bool = False

    async def pump(self) -> None:
        """Ask Orchestration, once per macro-cycle, how the next cycle should open."""
        state = self.shared_state
        if (
            "orchestration" not in self.backends
            or self.orch_prompt.is_user_supplied
            or state.orchestration_memory.get("for_cycle") == state.macro_cycle
        ):
            return
        feasible, _ = _phase_state.cycle_reloop_decision(state)
        if not feasible:
            return
        replies: list[Any] = []

        async def _directive_turn() -> None:
            replies.append(await self._coord.reactor_pass("orchestration", request=CYCLE_DIRECTIVE_REQUEST))

        await self._coord.await_within_session_bound(_directive_turn, stage="reactor:orchestration")
        state.orchestration_memory = build_cycle_memory(replies[0] if replies else None, cycle=state.macro_cycle)
        log.info(
            "cycle %d handoff: directive=%r parse_error=%r",
            state.macro_cycle,
            state.orchestration_memory["next_cycle_directive"][:80],
            state.orchestration_memory["parse_error"],
        )

    async def on_enter_sweep(self, tr: "Transition") -> None:
        """Start the ``conc_sweep``, and hand the cycle off at once when SWEEP settles on entry."""
        await self._start_conc_sweep(tr)
        # A settled SWEEP is left on the next advance, before this phase's pump runs again.
        if self.shared_state.last_conc_sweep.get("status"):
            await self.pump()

    async def _start_conc_sweep(self, tr: "Transition") -> None:
        """Auto-enqueue the ``conc_sweep`` task on SWEEP entry."""
        from_phase = tr.from_phase
        state = self.shared_state
        # A stack attempt an earlier entry or leg left behind may still have its
        # members on the tree, so settle it before the drain below applies
        # anything on top; the recovery halts the session if it cannot.
        await self._coord.phase_kernel_stack.recover_interrupted_stack_validation()
        # Drain pending KEEP integrates so sweep measures full current_best.
        if kernel_decisions.has_keep_pending_integrate(state):
            await self._coord.phase_kernel_stack.drain_pending_keep_integrates()
        # Validate the stack for positive NEEDS_REVIEW kernels.
        await self._coord.phase_kernel_stack.maybe_validate_positive_needs_review_stack()
        if not state.conc_sweep_enabled:
            log.info(
                "SWEEP entry (from=%s): conc_sweep disabled; recording terminal skip.",
                from_phase or "<unknown>",
            )
            self._record_terminal_conc_sweep_skip(
                skip_reason="disabled",
                auto_conc_sweep_skipped="disabled",
            )
            return
        prev_conc = state.last_conc_sweep_watermark
        prev_conc = prev_conc if isinstance(prev_conc, dict) else {}
        prev_validated = prev_conc.get("cumulative_gain_validated_at_record")
        cur_validated = float(state.cumulative_gain_validated or 0.0)
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
        denied = self._coord.dispatcher.time_budget_denial_for_action("conc_sweep")
        if denied is not None:
            log.info(
                "SWEEP entry (from=%s): conc_sweep cannot fit the session budget (%s); recording terminal skip.",
                from_phase or "<unknown>",
                denied,
            )
            self.record_session_budget_conc_sweep_skip(denied=denied)
            return
        try:
            task = await self._enqueue_internal_conc_sweep_task(
                reason="phase_entry",
            )
        except (sqlite3.Error, TypeError, ValueError) as exc:
            log.exception(
                "SWEEP entry hook: failed to enqueue auto-conc-sweep: %r",
                exc,
            )
            self._record_terminal_conc_sweep(
                "failed",
                "enqueue_failed",
                auto_conc_sweep_error=repr(exc)[:240],
            )
            return
        # The only None is the helper's own budget decline, which records its terminal skip before returning.
        if task is None:
            return
        log.info(
            "SWEEP entry (from=%s): auto-enqueued conc_sweep task=%s (concs=%s total_budget_sec=%s)",
            from_phase or "<unknown>",
            task.task_id,
            task.params.get("concs"),
            task.params.get("total_budget_sec"),
        )
        self._coord.phase_machine.record_phase_entry_evidence(
            auto_conc_sweep_enqueued=True,
            auto_conc_sweep_task_id=task.task_id,
            # Verbatim: None records "the workload picks", which is not the same statement as an empty ladder.
            auto_conc_sweep_concs=task.params.get("concs"),
        )

    async def _enqueue_internal_conc_sweep_task(
        self,
        *,
        reason: str,
    ) -> Task | None:
        """Build + enqueue a Coordinator-internal ``conc_sweep`` task."""
        state = self.shared_state
        configured_budget = int(state.conc_sweep_total_budget_sec)
        # Clamp total_budget_sec to the remaining session wall-clock budget so a long conc_sweep cannot outlive
        # --max-hours.
        _CLOSE_RESERVE_SEC = 120
        session_rem = state.remaining_minutes()
        clamped_budget: int | None
        if session_rem is not None:
            session_rem_sec = int(max(0.0, session_rem * 60.0) - _CLOSE_RESERVE_SEC)
            clamped_budget = min(configured_budget, session_rem_sec) if configured_budget > 0 else session_rem_sec
        else:
            clamped_budget = configured_budget if configured_budget > 0 else None
        admission_sec = 60.0 * self._coord.dispatcher.admission_cost_minutes("conc_sweep")
        if clamped_budget is not None and clamped_budget < admission_sec:
            log.info(
                "conc_sweep: a %ds grant cannot fit the %.0fs one pair costs; not enqueueing.",
                clamped_budget,
                admission_sec,
            )
            self._record_terminal_conc_sweep_skip(
                skip_reason="grant_cannot_fit_one_pair",
                auto_conc_sweep_skipped="grant_cannot_fit_one_pair",
                auto_conc_sweep_denied=f"granted={clamped_budget}s one_pair={admission_sec:.0f}s",
            )
            return None
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": str(reason),
            # None, not [], when the state carries no ladder: the executor reads [] as a deliberate "no concs" and
            # skips, while None lets it fall back to the ladder for this workload.
            "concs": list(state.conc_sweep_concs) if state.conc_sweep_concs else None,
            "total_budget_sec": clamped_budget,
        }
        task, was_existing = await self._live_conc_sweep_row(reason=reason, params=params, budget=clamped_budget)
        self._grant_conc_sweep(task, was_existing=was_existing, budget=clamped_budget)
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

    async def _live_conc_sweep_row(
        self,
        *,
        reason: str,
        params: dict[str, Any],
        budget: int | None,
    ) -> tuple[Task, bool]:
        """Return the cycle's live ``conc_sweep`` row, enqueueing one when the keyed row is absent or settled.

        A resumed SWEEP re-enters a cycle whose earlier sweep may already have settled without its outcome being
        recorded (``recover`` cancels it out of band), so only a live row is reused.
        """
        lanes, _ = self._coord.dispatcher.registry_lanes_ttl("conc_sweep")
        base_key = f"internal-conc_sweep-{reason}{self._coord.dispatcher.cycle_idem_suffix()}"
        attempt = 0
        while True:
            task, was_existing = await self.tasks.create_or_return_existing(
                TaskSpec(
                    kind="conc_sweep",
                    params=params,
                    requires_lanes=lanes,
                    idempotency_key=base_key if attempt == 0 else f"{base_key}-r{attempt}",
                    lease_ttl_sec=_conc_sweep_lease_ttl_sec(budget),
                    dispatch_class="coordinator",
                )
            )
            if not (was_existing and task.state in TERMINAL_STATES):
                return task, was_existing
            attempt += 1

    def _grant_conc_sweep(self, task: Task, *, was_existing: bool, budget: int | None) -> None:
        """Grant the phase's budget exits' deferral to the sweep it is waiting on."""
        state = self.shared_state
        now_unix = _phase_state._now_unix(state)
        if not was_existing and budget is not None and budget > 0:
            # The phase granted this budget, so its own budget exits must not cut the sweep short of it: they defer
            # until the grant (and the lease grace that bounds the task anyway) runs out.
            state.conc_sweep_granted_until_unix = now_unix + _conc_sweep_lease_ttl_sec(budget)
        elif (
            task.state == "queued"
            and int(task.lease_ttl_sec or 0) > 0
            and float(state.conc_sweep_granted_until_unix or 0.0) <= now_unix
        ):
            # A sweep that never started is still owed its budget: a resume after a long gap finds its grant spent
            # by the time the session was down, and the phase would otherwise leave SWEEP before dispatching it.
            state.conc_sweep_granted_until_unix = now_unix + int(task.lease_ttl_sec)

    async def resume_unsettled_sweep(self) -> None:
        """Re-fire the SWEEP entry hook, once per resumed process, when the phase's conc_sweep left no outcome.

        SWEEP's only work is the ``conc_sweep`` its entry hook enqueues, and only that task's own completion records
        ``last_conc_sweep``. A sweep cancelled out of band (``hyperloom recover`` cancels every task of a vanished
        owner scope) records nothing, and a resume never re-enters the current phase, so the resumed leg would wait
        in SWEEP for a result no task will write, until the grant the lost sweep held runs out. The entry hook is
        idempotent: a live sweep row is reused, and a settled one is replaced by a fresh dispatch that restarts the
        sweep (it keeps no per-point progress to resume from).

        Acts on the process's first phase decision only, before it is taken, so the decision sees the sweep and its
        grant. A session that reaches SWEEP later in the process enters it through the transition's own hook, and
        re-firing that hook under a running sweep would apply pending KEEPs underneath its measurement.
        """
        if self._resume_checked:
            return
        self._resume_checked = True
        state = self.shared_state
        if not self._coord.writeback.resumed_from.get("is_resume"):
            return
        if (state.phase or "").strip().upper() != _phase_state.PHASE_SWEEP:
            return
        last = state.last_conc_sweep
        if isinstance(last, dict) and str(last.get("status") or "").strip():
            return
        log.info("resume: SWEEP has no recorded conc_sweep outcome; re-entering SWEEP to dispatch it.")
        from .machine import Transition

        await self.on_enter_sweep(
            Transition(
                from_phase=_RESUME,
                to_phase=_phase_state.PHASE_SWEEP,
                reason=_RESUME,
                evidence={},
            )
        )

    def record_session_budget_conc_sweep_skip(self, *, denied: object) -> None:
        """Stamp last_conc_sweep skipped when the session clock refused conc_sweep."""
        last = self.shared_state.last_conc_sweep or {}
        if str(last.get("status") or "").strip():
            return
        self._record_terminal_conc_sweep(
            "skipped",
            "session_time_budget",
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
        self._record_terminal_conc_sweep("skipped", skip_reason, **evidence)

    def _record_terminal_conc_sweep(
        self,
        status: str,
        reason: str,
        **evidence: Any,
    ) -> None:
        """Record a terminal conc-sweep outcome and persist state."""
        self._coord.phase_machine.record_phase_entry_evidence(**evidence)
        record: dict[str, Any] = {"status": status, "skip_reason": reason}
        if status == "skipped":
            record["was_skipped"] = True
        self.shared_state.record_conc_sweep(record)
        self.shared_state.save(self.session_dir)
