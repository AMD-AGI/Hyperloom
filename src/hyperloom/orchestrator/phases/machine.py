# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase state-machine handler: initialisation, exit-condition scan/transition, and the per-phase entry dispatcher
(``_on_phase_entered``).
"""

from __future__ import annotations
import logging as _logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from . import machine_state as _phase_state
from ..bus.message_bus import Message
from ..prompts import write_prompt_snapshot as _write_prompt_snapshot
from ..collaborator import CoordinatorCollaborator
from hyperloom.plugs import progress as _progress

log = _logging.getLogger(__name__)

# SWEEP budget exits end the run for lack of time; every other CLOSE transition reason is itself a stop reason.
_BUDGET_EXIT_STOP_REASONS = {
    "sweep_budget_exhausted": "time_exhausted",
}

# The FRAMEWORK_AGENT exit reason for a plateau or an LLM hint; budget and terminal exits never wait.
_LEVERAGE_EXIT_REASON = "optimize_no_more_leverage"


def is_leverage_exit(next_phase: tuple[str, str, dict[str, Any]]) -> bool:
    """Whether a FRAMEWORK_AGENT transition is the plateau or LLM-hint exit, the one operator work may hold."""
    _target, reason, _evidence = next_phase
    return reason == _LEVERAGE_EXIT_REASON


@dataclass(frozen=True)
class Transition:
    from_phase: str
    to_phase: str
    reason: str
    evidence: dict


class MachinePhase(CoordinatorCollaborator):
    """Phase transition machine: validates, records, and dispatches phase transitions."""

    def __init__(self, coordinator) -> None:
        super().__init__(coordinator)
        self._on_enter: dict[str, Any] = {}
        self._on_exit: dict[str, Any] = {}
        self.pump_table: dict[str, Any] = {}

    def build_dispatch_tables(self) -> None:
        """Build on_enter, on_exit, and pump dispatch tables from phase owners. Called once after all collaborators are available."""
        c = self._coord
        self._on_exit = {
            _phase_state.PHASE_KERNEL_AGENT: c.phase_kernel.close_kernel_timeline,
            _phase_state.PHASE_FRAMEWORK_AGENT: c.phase_framework.close_framework_timeline,
        }
        self._on_enter = {
            _phase_state.PHASE_FRAMEWORK_AGENT: c.phase_framework.on_enter_framework,
            _phase_state.PHASE_KERNEL_AGENT: c.phase_kernel.on_enter_kernel,
            _phase_state.PHASE_SWEEP: c.phase_sweep.on_enter_sweep,
            _phase_state.PHASE_CLOSE: c.phase_close.on_enter_close,
        }
        self.pump_table = {
            _phase_state.PHASE_FRAMEWORK_AGENT: c.phase_framework.pump,
            _phase_state.PHASE_SWEEP: c.phase_sweep.pump,
        }

    def ensure_phase_initialised(self, budget_pct: dict[str, float] | None) -> None:
        """Set ``phase`` + persist ``phase_budget_pct`` once per session (idempotent).

        When *budget_pct* is given, or when state has no budget yet (fresh session),
        the budget is redistributed and written; on resume without explicit overrides the
        prior budget is kept.
        """
        state = self.shared_state
        if budget_pct or not state.phase_budget_pct:
            state.phase_budget_pct = _phase_state.redistribute_budget_pct(
                _phase_state.normalize_budget_pct(budget_pct),
                optimize_enabled=self.optimize_enabled(),
                kernel_enabled=self.kernel_enabled(),
                sweep_enabled=state.conc_sweep_enabled,
            )
        current = (state.phase or "").strip().upper()
        # Only an unset phase means fresh; an unknown one would otherwise re-run PRELUDE over the earlier build's
        # baseline and KEPT stack.
        if current and current not in _phase_state.PHASE_NAMES:
            raise RuntimeError(
                f"session was recorded at phase {current!r}, which this build's phase machine "
                f"does not have (known: {', '.join(_phase_state.PHASE_NAMES)}). "
                f"Resume it with the version that wrote it, or start a new session."
            )
        if current == _phase_state.PHASE_CLOSE:
            self._reopen_a_session_that_was_left_closed()
        elif not current:
            # Fresh start; pre-phase-machine resume state is treated as fresh.
            _phase_state.record_phase_transition(
                state,
                to_phase=_phase_state.PHASE_PRELUDE,
                reason="phase_entered",
                evidence={
                    "trigger": "fresh_session",
                    "predicate_inputs": _phase_state.initial_workflow_predicate_inputs(
                        state,
                        current_phase="",
                        flags=_phase_state.PhaseFlags(
                            kernel_enabled=self.kernel_enabled(),
                            optimize_enabled=self.optimize_enabled(),
                            enablement_enabled=self._coord.enablement_lane.enablement_admitted(),
                        ),
                    ),
                },
            )
        try:
            state.save(self.session_dir)
        except OSError:
            log.exception("Coordinator: save after phase init failed")

    def _reopen_a_session_that_was_left_closed(self) -> None:
        """Put a session persisted in CLOSE back at the phase machine's entrance."""
        state = self.shared_state
        log.info(
            "Coordinator: session resumed in CLOSE, a phase with no way out; "
            "reopening at PRELUDE so the new budget can be spent on the work "
            "the earlier leg stopped short of."
        )
        _phase_state.record_phase_transition(
            state,
            to_phase=_phase_state.PHASE_PRELUDE,
            reason="phase_entered",
            evidence={
                "trigger": "resumed_from_close",
                "predicate_inputs": _phase_state.initial_workflow_predicate_inputs(
                    state,
                    current_phase=_phase_state.PHASE_CLOSE,
                    flags=_phase_state.PhaseFlags(
                        kernel_enabled=self.kernel_enabled(),
                        optimize_enabled=self.optimize_enabled(),
                        enablement_enabled=self._coord.enablement_lane.enablement_admitted(),
                    ),
                ),
            },
        )
        # Locked True by the CLOSE sequencer and read by the end-of-run safety nets as "the sequencer already wrote
        # the breakdown".
        state.close_sequence_done = False

    def ensure_recipe_kb_t0_anchored(self) -> None:
        """Defensive T0 anchor for SDK callers constructed without cli plumbing. Skips when recipe_kb is None or recipe_kb_session_id set."""
        client = self.recipe_kb
        if client is None or not getattr(client, "enabled", True):
            return
        state = self.shared_state
        if (state.recipe_kb_session_id or "").strip():
            # cli already T0'd or resume picked up the sid.
            return
        # Derive workload / hw from SharedState.
        workload = state.model_name or "unknown_model"
        hw = state.gpu_type or "unknown_gpu"
        extra_attrs = {
            "marathon_dispatch_id": state.session_id or "",
            "framework_name": state.framework or "",
            "model_class": state.model_class or "",
            "claw_session_id": state.claw_session_id or "",
            "sandbox_user_id": state.sandbox_user_id or "",
            # boot_origin is a dev-debug label, NOT written to KB.
            "boot_origin": "coordinator_fallback",
        }
        try:
            from ..knowledge.recipe_kb_t0 import T0AnchorRequest, run_t0_anchor

            run_t0_anchor(
                client,
                state,
                T0AnchorRequest(workload=workload, hw=hw, extra_attrs=extra_attrs),
                session_dir=self.session_dir,
            )
        except Exception:  # isolation boundary: the SDK-path T0 anchor, which leaves warm start empty
            log.exception(
                "Coordinator T0 fallback: run_t0_anchor raised (workload=%s, hw=%s); warm_start stays empty",
                workload,
                hw,
            )

    def kernel_enabled(self) -> bool:
        """Whether kernel optimization is enabled for this run."""
        return bool(self.shared_state.kernel_enabled)

    def optimize_enabled(self) -> bool:
        """Whether the optimisation phase is enabled for this run."""
        return bool(self.shared_state.framework_agent_phase_enabled)

    def set_terminal_stop_reason(self, state: Any, reason: str) -> None:
        """Derive the stop reason for a CLOSE transition. The first reason recorded wins.

        Must run before CLOSE writes the session breakdown: the collector derives the stop reason from ``state.json``.
        """
        if not state.stop_reason:
            state.set_stop_reason(_BUDGET_EXIT_STOP_REASONS.get(reason, reason), strict=True)

    async def _next_phase(
        self, *, optimize_enabled: bool, enablement_in_flight: bool, kernel_work_in_flight: bool
    ) -> tuple[str, str, dict[str, Any]] | None:
        """The transition the predicates ask for, held while the operator-defined specialist's work is pending."""
        state = self.shared_state
        next_phase = _phase_state.compute_next_phase(
            state,
            enablement_in_flight=enablement_in_flight,
            kernel_work_in_flight=kernel_work_in_flight,
            flags=_phase_state.PhaseFlags(
                kernel_enabled=self.kernel_enabled(),
                optimize_enabled=optimize_enabled,
                enablement_enabled=self._coord.enablement_lane.enablement_admitted(),
            ),
        )
        if next_phase is None or str(state.phase or "").upper() != _phase_state.PHASE_FRAMEWORK_AGENT:
            return next_phase
        if not is_leverage_exit(next_phase) or not await self._coord.phase_framework.operator_work_pending():
            return next_phase
        log.info("phase_machine: holding %s exit (reason=%s) for operator work", state.phase, next_phase[1])
        return None

    def _settle_escalate_hint(self, move: Transition) -> None:
        """Consume or discard the pending escalate hint, according to what drove this transition."""
        state = self.shared_state
        if "hint" in move.evidence:
            state.consume_pending_escalate_hint()
        elif state.pending_escalate_hint:
            # A hint speaks for the phase it was emitted in; carried across a transition it did not drive, it would
            # act on a phase it was never about.
            discarded_hint = state.discard_pending_escalate_hint()
            log.info(
                "phase_machine: discarded stale pending_escalate_hint=%r on unrelated transition %s -> %s (reason=%s)",
                discarded_hint,
                move.from_phase,
                move.to_phase,
                move.reason,
            )

    def _carry_cycle_facts_across(self, move: Transition) -> None:
        """Persist the cycle-level facts the entered phase, or a later resume, has to see."""
        state = self.shared_state
        # A cyclic config-arm plateau winds the cycle down with ``switch_bottleneck``: record the plateaued bottleneck
        # so the next cycle steers specialists off it.
        if move.evidence.get("switch_bottleneck"):
            state.mark_bottleneck_switch(
                prev_bottleneck=state.current_top_bottleneck(),
            )
            log.info(
                "plateau → bottleneck switch flagged (off %r)",
                state.last_cycle_bottleneck,
            )
        # Persist the no-gain streak on a cyclic-mode terminal close so a subsequent resume sees the convergence state.
        if (
            not move.evidence.get("loopback")
            and move.to_phase == _phase_state.PHASE_CLOSE
            and "no_gain_cycle_streak_effective" in move.evidence
        ):
            state.no_gain_cycle_streak = int(move.evidence.get("no_gain_cycle_streak_effective", 0) or 0)

    def _open_looped_back_cycle(self, move: Transition, prior_cycle: int) -> None:
        """Open the macro-cycle a loopback transition has just entered."""
        state = self.shared_state
        state.open_macro_cycle(no_gain_cycle_streak=int(move.evidence.get("no_gain_cycle_streak_effective") or 0))
        self._coord.phase_macro_cycle.record_cycle_strategy_for_current_cycle()
        log.info(
            "Coordinator: macro-cycle reloop %d -> %d (no_gain_streak=%d, gain_anchor=%.4f)",
            prior_cycle,
            state.macro_cycle,
            state.no_gain_cycle_streak,
            state.gain_at_cycle_start,
        )

    async def _announce_transition(self, move: Transition) -> None:
        """Publish the phase boundary on the bus for the roles that follow it."""
        try:
            await self.bus.append_and_seq(
                Message.new(
                    "coordinator",
                    "*",
                    "event",
                    {
                        "kind": "phase_transition",
                        "from_phase": move.from_phase,
                        "to_phase": move.to_phase,
                        "reason": move.reason,
                        "evidence": move.evidence,
                    },
                )
            )
        except sqlite3.Error:
            log.exception("Coordinator: phase_transition event bus write failed")

    async def _fire_phase_entry(self, move: Transition) -> None:
        """Run the entered phase's side effects, recording rather than raising a hook failure."""
        # Phase-entry side effects are additive; hook failures are logged only. They run on the transition itself,
        # which is what callers advancing into CLOSE rely on to see the sequencer's settlement once it returns.
        try:
            await self._on_phase_entered(
                from_phase=move.from_phase,
                to_phase=move.to_phase,
                reason=move.reason,
                evidence=move.evidence,
            )
        except Exception as exc:  # isolation boundary: the entered phase's hook, kept apart from the transition
            log.exception("Coordinator: _on_phase_entered hook failed")
            # This hook is also what closes the left phase's event, so a raise here is the case where that event never
            # got its exit evidence.
            self._coord.record_exception(stage="phase_entered", exc=exc)

    async def _commit_transition(self, move: Transition) -> None:
        """Record the phase move and run everything the boundary owes, in order."""
        state = self.shared_state
        self._settle_escalate_hint(move)
        if move.to_phase == _phase_state.PHASE_CLOSE:
            self.set_terminal_stop_reason(state, move.reason)
        self._carry_cycle_facts_across(move)
        is_loopback = bool(move.evidence.get("loopback"))
        prior_cycle = state.macro_cycle
        _phase_state.record_phase_transition(
            state,
            to_phase=move.to_phase,
            reason=move.reason,
            evidence=move.evidence,
        )
        if is_loopback:
            self._open_looped_back_cycle(move, prior_cycle)
        # Mirror the phase boundary into the operator-facing lifecycle log using the ENTER status (a point-in-time
        # marker, not a START/END interval).
        _phase_state.record_lifecycle_event(
            state,
            _phase_state.LifecycleMark(
                step=move.to_phase,
                status=_phase_state.LIFECYCLE_STATUS_ENTER,
                phase=move.to_phase,
                detail=f"reason={move.reason}" if move.reason else "",
            ),
        )
        # The in-memory phase has already moved, so a failed save is raised only after the entry has run.
        save_error: OSError | None = None
        try:
            state.save(self.session_dir)
        except OSError as exc:
            save_error = exc
            _progress.warn(
                "state_save_failed", f"state.json could not be written at the {move.to_phase} boundary: {exc}"
            )
        _progress.phase_changed(move.from_phase, move.to_phase, state.macro_cycle)
        log.info(
            "Coordinator.phase: %s → %s (reason=%s)",
            move.from_phase or "<unset>",
            move.to_phase,
            move.reason,
        )
        await self._announce_transition(move)
        await self._fire_phase_entry(move)
        if is_loopback:
            await self._coord.phase_macro_cycle.run_cycle_soft_restart(
                prior_cycle=prior_cycle,
                new_cycle=state.macro_cycle,
            )
        if save_error is not None:
            raise save_error

    async def advance_phase_if_needed(self) -> None:
        """Scan exit conditions and transition phase at most once per tick."""
        state = self.shared_state
        # The deadline path records CLOSE before it dispatches the report, so the machine has no transition left to
        # make; the sequencer it stopped short of still has to run.
        if state.closing_phase and state.phase == _phase_state.PHASE_CLOSE and not state.close_sequence_done:
            # The closing transition supersedes any barrier this machine was holding, and the closing report is
            # dispatched by the pump.
            self._coord.dispatcher.admission_frozen = False
            await self._coord.phase_close.ensure_close_sequence(reason="time_exhausted")
            return
        kernel_facts = await self._coord.phase_kernel.exit_facts()
        in_enablement = str(state.phase or "").upper() == _phase_state.PHASE_ENABLEMENT
        await self._release_holds_no_task_will_settle()
        next_phase = await self._next_phase(
            optimize_enabled=self.optimize_enabled(),
            enablement_in_flight=in_enablement and await self._coord.enablement_lane.enablement_in_flight(),
            kernel_work_in_flight=kernel_facts.agent_in_flight,
        )
        await self._coord.phase_internal.maybe_enqueue_trajectory_reviewer()
        dispatcher = self._coord.dispatcher
        # Admission stays frozen for as long as a transition is pending.
        dispatcher.admission_frozen = next_phase is not None and next_phase[0] != (state.phase or "").upper()
        if not dispatcher.admission_frozen:
            return
        target, reason, evidence = next_phase
        prior = state.phase
        if not await self._drain_for_transition(prior=prior, target=target, reason=reason, evidence=evidence):
            return
        dispatcher.admission_frozen = False
        await self._commit_transition(
            Transition(
                from_phase=prior or "",
                to_phase=target,
                reason=reason,
                evidence=evidence if isinstance(evidence, dict) else {},
            )
        )

    async def _release_holds_no_task_will_settle(self) -> None:
        """Settle what the current phase's exit is waiting on when the task that would have settled it is gone.

        A task cancelled outside the run (``hyperloom recover``) or ended before a restart booked it never reaches
        the writeback that settles the phase's hold, so the hold would run to its allowance.
        """
        phase = str(self.shared_state.phase or "").upper()
        if phase in (_phase_state.PHASE_PRELUDE, _phase_state.PHASE_ENABLEMENT):
            # Both phases hold their exit for this marker.
            await self._coord.phase_kernel.release_finished_roofline_gate()
        # SWEEP waits for its conc_sweep's outcome; checked on a resumed process's first decision only.
        await self._coord.phase_sweep.resume_unsettled_sweep()

    async def _drain_for_transition(self, *, prior: str, target: str, reason: str, evidence: Any) -> bool:
        """Clear the way to ``target`` and return whether the transition may commit on this tick.

        The next phase starts on quiet GPUs: nothing new is admitted, queued work the next phase does not admit is
        dropped, and the transition waits until every running action has finished and is booked in this phase.
        Running work is cancelled only when the transition is terminal (``evidence["terminal"]``): the run is
        already over, so there is no measurement left worth settling. Otherwise a phase budget only stops new work,
        and cancelling the last action in flight would throw away the measurement it was about to settle; session
        exhaustion and shutdown still cancel it through the dispatcher.
        """
        dispatcher = self._coord.dispatcher
        barrier_reason = f"phase_transition:{str(prior or '').strip().upper()}->{target}"
        cancelled = await self.tasks.cancel_queued(
            allowed_kinds=_phase_state.PHASE_ALLOWED_ACTIONS.get(target, frozenset()),
            reason=barrier_reason,
        )
        terminal = isinstance(evidence, dict) and bool(evidence.get("terminal"))
        stopped = await dispatcher.cancel_inflight_actions(reason=barrier_reason) if terminal else []
        if cancelled or stopped:
            log.info(
                "Coordinator.phase: %s (reason=%s) cancelled %d queued and stopped %d running task(s)",
                barrier_reason,
                reason,
                len(cancelled),
                len(stopped),
            )
            await self.bus.record_observation(
                "coordinator",
                "observation",
                {
                    "kind": "tasks_cancelled_on_phase_transition",
                    "prior_phase": str(prior or ""),
                    "target_phase": target,
                    "reason": reason,
                    "cancelled_task_ids": cancelled,
                    "stopped_task_ids": stopped,
                },
            )
        running = await self.tasks.running()
        if running or dispatcher.has_unbooked_completions():
            log.info(
                "phase_machine: holding %s (reason=%s) until %d running task(s) finish and are booked",
                barrier_reason,
                reason,
                len(running),
            )
            return False
        return True

    async def _on_phase_entered(
        self,
        *,
        from_phase: str,
        to_phase: str,
        reason: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> None:
        """Fire per-phase entry side effects (pure dispatcher; hooks catch + log internally). CLOSE runs the 7-step sequencer (sets close_sequence_done)."""
        self._reseed_orch_prompt_for_phase(to_phase)

        tr = Transition(
            from_phase=from_phase or "",
            to_phase=(to_phase or "").upper(),
            reason=str(reason or ""),
            evidence=evidence if isinstance(evidence, dict) else {},
        )

        self.fire_phase_exit(tr)

        entry_hook = self._on_enter.get(tr.to_phase)
        if entry_hook:
            await entry_hook(tr)

    def fire_phase_exit(self, tr: Transition) -> None:
        """Settle the internal timeline owned by the phase being left.

        Reached on its own by the terminal CLOSE path, which records the transition without an entry dispatch.
        """
        if not self._on_exit:
            self.build_dispatch_tables()
        exit_hook = self._on_exit.get(tr.from_phase.upper())
        if exit_hook:
            exit_hook(tr)

    def _cycle_directive(self) -> str:
        """The directive Orchestration wrote at the previous cycle's handoff, or empty."""
        memory = self.shared_state.orchestration_memory
        if memory.get("for_cycle") == self.shared_state.macro_cycle - 1:
            return str(memory.get("next_cycle_directive") or "")
        return ""

    def _reseed_orch_prompt_for_phase(self, to_phase: str) -> bool:
        """Re-scope the orchestration system prompt to the phase being entered."""
        phase = (to_phase or "").strip().upper()
        orch_prompt = self.orch_prompt
        if not phase or orch_prompt.is_user_supplied:
            return False
        rebuild = orch_prompt.rebuild
        if rebuild is None:
            return False
        state = self.shared_state
        cycle = int(state.macro_cycle or 0)
        log_rows = list(state.cycle_strategy_log or [])
        prior_cycles = [r for r in log_rows if isinstance(r, dict) and int(r.get("cycle", -1) or -1) != cycle]
        focus_plan = self._coord.phase_macro_cycle.plan_cycle_focus()
        focus_plan["prior_cycles"] = prior_cycles[-5:]
        scoped = rebuild(
            macro_cycle=state.macro_cycle,
            cycle_directive=self._cycle_directive(),
            cycle_strategy=focus_plan,
            phase=phase,
        )
        orch_prompt.set("orchestration", scoped)
        _write_prompt_snapshot(
            self.session_dir, "orchestration", scoped, phase=phase, macro_cycle=int(state.macro_cycle or 0)
        )
        log.info("orchestration prompt re-scoped for phase=%s", phase)
        return True

    def record_phase_entry_evidence(self, **kvs: Any) -> None:
        """Merge ``kvs`` into the latest phase_history row's evidence dict (no-op when empty)."""
        history = self.shared_state.phase_history or []
        if not history:
            return
        row = history[-1]
        if not isinstance(row, dict):
            return
        evidence = row.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
            row["evidence"] = evidence
        for k, v in kvs.items():
            evidence[k] = v
        self.shared_state.save(self.session_dir)
