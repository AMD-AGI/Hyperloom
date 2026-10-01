# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase state-machine handler: initialisation, exit-condition scan/transition, and the per-phase entry dispatcher
(``_on_phase_entered``).
"""

from __future__ import annotations
import logging as _logging
from dataclasses import dataclass
from typing import Any

from . import machine_state as _phase_state
from ..bus.message_bus import Message
from ..prompts import write_prompt_snapshot as _write_prompt_snapshot
from ..state.shared_state import ESCALATE_HINT_SKIP_TO_CLOSE
from ..collaborator import CoordinatorCollaborator

log = _logging.getLogger(__name__)

# SWEEP budget exits end the run for lack of time; every other CLOSE transition reason is itself a stop reason.
_BUDGET_EXIT_STOP_REASONS = {
    "sweep_budget_exhausted": "time_exhausted",
    "sweep_budget_cap": "time_exhausted",
}


@dataclass(frozen=True)
class Transition:
    from_phase: str
    to_phase: str
    reason: str
    evidence: dict
    loopback: bool


class MachinePhase(CoordinatorCollaborator):
    """Phase transition machine: validates, records, and dispatches phase transitions."""

    def __init__(self, coordinator) -> None:
        super().__init__(coordinator)
        self._on_enter: dict[str, Any] = {}
        self._on_exit: dict[str, Any] = {}
        self._pump_table: dict[str, Any] = {}

    def _build_dispatch_tables(self) -> None:
        """Build on_enter, on_exit, and pump dispatch tables from phase owners. Called once after all collaborators are available."""
        c = self._coord
        self._on_exit = {
            _phase_state.PHASE_KERNEL_AGENT: c.phase_kernel._close_kernel_timeline,
            _phase_state.PHASE_FRAMEWORK_AGENT: c.phase_framework._close_framework_timeline,
        }
        self._on_enter = {
            _phase_state.PHASE_FRAMEWORK_AGENT: c.phase_framework._on_enter_framework,
            _phase_state.PHASE_KERNEL_AGENT: c.phase_kernel._on_enter_kernel,
            _phase_state.PHASE_SWEEP: c.phase_sweep._on_enter_sweep,
            _phase_state.PHASE_CLOSE: c.phase_close._on_enter_close,
        }
        self._pump_table = {
            _phase_state.PHASE_FRAMEWORK_AGENT: c.phase_framework.pump,
            _phase_state.PHASE_SWEEP: c.phase_sweep.pump,
        }

    def _ensure_phase_initialised(self, budget_pct: dict[str, float] | None) -> None:
        """Set ``phase`` + persist ``phase_budget_pct`` once per session (idempotent).

        When *budget_pct* is given, or when state has no budget yet (fresh session),
        the budget is redistributed and written; on resume without explicit overrides the
        prior budget is kept.
        """
        state = self.shared_state
        if budget_pct or not state.phase_budget_pct:
            state.phase_budget_pct = _phase_state.redistribute_budget_pct(
                _phase_state.normalize_budget_pct(budget_pct),
                optimize_enabled=self._optimize_enabled(),
                kernel_enabled=self._kernel_enabled(),
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
                        kernel_enabled=self._kernel_enabled(),
                        optimize_enabled=self._optimize_enabled(),
                        enablement_enabled=self._coord.enablement_lane._enablement_admitted(),
                    ),
                },
            )
        try:
            state.save(self.session_dir)
        except Exception:
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
                    kernel_enabled=self._kernel_enabled(),
                    optimize_enabled=self._optimize_enabled(),
                    enablement_enabled=self._coord.enablement_lane._enablement_admitted(),
                ),
            },
        )
        # Locked True by the CLOSE sequencer and read by the end-of-run safety nets as "the sequencer already wrote
        # the breakdown".
        state.close_sequence_done = False

    def _ensure_recipe_kb_t0_anchored(self) -> None:
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
            from ..knowledge.recipe_kb_t0 import run_t0_anchor

            run_t0_anchor(
                client,
                state,
                workload=workload,
                hw=hw,
                extra_attrs=extra_attrs,
                session_dir=self.session_dir,
                save_state=True,
            )
        except Exception:
            log.exception(
                "Coordinator T0 fallback: run_t0_anchor raised (workload=%s, hw=%s); warm_start stays empty",
                workload,
                hw,
            )

    def _kernel_enabled(self) -> bool:
        """Whether kernel optimization is enabled for this run."""
        return bool(self.shared_state.kernel_enabled)

    def _optimize_enabled(self) -> bool:
        """Whether the optimisation phase is enabled for this run."""
        return bool(self.shared_state.framework_agent_phase_enabled)

    async def _advance_phase_if_needed(self) -> None:
        """Scan exit conditions and transition phase at most once per tick."""
        state = self.shared_state
        kernel_facts = await self._coord.phase_kernel.exit_facts()
        optimize_enabled = self._optimize_enabled()
        # Only asked inside the phase: the query renews the open round's lease.
        in_enablement = str(state.phase or "").upper() == _phase_state.PHASE_ENABLEMENT
        enablement_in_flight = in_enablement and await self._coord.enablement_lane._enablement_in_flight()
        next_phase = _phase_state.compute_next_phase(
            state,
            kernel_enabled=self._kernel_enabled(),
            optimize_enabled=optimize_enabled,
            enablement_enabled=self._coord.enablement_lane._enablement_admitted(),
            enablement_in_flight=enablement_in_flight,
            kernel_work_in_flight=kernel_facts.agent_in_flight,
        )
        await self._coord.phase_internal._maybe_enqueue_trajectory_reviewer()
        if next_phase is None:
            return
        target, reason, evidence = next_phase
        if target == (state.phase or "").upper():
            return  # already there
        prior = state.phase
        barrier_reason = f"phase_transition:{str(prior or '').strip().upper()}->{target}"
        # The next phase starts on quiet GPUs: every running action is stopped, and the transition waits until the
        # registry confirms none is left running. Queued work the next phase does not admit is dropped here too.
        cancelled = await self.tasks.cancel_queued(
            allowed_kinds=_phase_state.PHASE_ALLOWED_ACTIONS.get(target, frozenset()),
            reason=barrier_reason,
        )
        stopped = await self._coord.dispatcher.cancel_inflight_actions(reason=barrier_reason)
        if cancelled or stopped:
            log.info(
                "Coordinator.phase: %s cancelled %d queued and stopped %d running task(s)",
                barrier_reason,
                len(cancelled),
                len(stopped),
            )
            await self._coord.writeback._record_observation(
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
        if running:
            log.info("phase_machine: holding %s until %d running task(s) stop", barrier_reason, len(running))
            return
        # Consume escalate hint after a hint-driven transition.
        if isinstance(evidence, dict) and (evidence.get("evidence") == "llm_escalation" or "hint" in evidence):
            state.consume_pending_escalate_hint()
        elif (
            str(prior or "").strip().upper() == _phase_state.PHASE_SWEEP
            and str(state.pending_escalate_hint or "").strip() == ESCALATE_HINT_SKIP_TO_CLOSE
        ):
            # SWEEP already had an honest closeout, so skip_to_close was suppressed in _global_terminal.
            state.consume_pending_escalate_hint()
        elif state.pending_escalate_hint and target != _phase_state.PHASE_FRAMEWORK_AGENT:
            # FRAMEWORK_AGENT exit consumes ``skip_to_kernel``; a transition to any other phase leaves the hint
            # unclaimable. A transition *into* FRAMEWORK_AGENT is the opposite case: discarding there would drop
            # the hint on the doorstep of the rules that read it.
            discarded_hint = state.discard_pending_escalate_hint()
            log.info(
                "phase_machine: discarded stale pending_escalate_hint=%r on unrelated transition %s -> %s (reason=%s)",
                discarded_hint,
                prior,
                target,
                reason,
            )
        # Terminal transition (target=CLOSE): set stop_reason once from the transition reason.
        if target == _phase_state.PHASE_CLOSE and not state.stop_reason:
            state.set_stop_reason(_BUDGET_EXIT_STOP_REASONS.get(reason, reason), strict=True)
        # A cyclic config-arm plateau winds the cycle down with ``switch_bottleneck``: record the plateaued bottleneck
        # so the next cycle steers specialists off it.
        if isinstance(evidence, dict) and evidence.get("switch_bottleneck"):
            state.mark_bottleneck_switch(
                prev_bottleneck=state.current_top_bottleneck(),
            )
            log.info(
                "plateau → bottleneck switch flagged (off %r)",
                state.last_cycle_bottleneck,
            )
        is_loopback = bool(isinstance(evidence, dict) and evidence.get("loopback"))
        # Persist the no-gain streak on a cyclic-mode terminal close so a subsequent resume sees the convergence state.
        if (
            not is_loopback
            and target == _phase_state.PHASE_CLOSE
            and isinstance(evidence, dict)
            and "no_gain_cycle_streak_effective" in evidence
        ):
            state.no_gain_cycle_streak = int(evidence.get("no_gain_cycle_streak_effective", 0) or 0)
        prior_cycle = state.macro_cycle
        _phase_state.record_phase_transition(
            state,
            to_phase=target,
            reason=reason,
            evidence=evidence,
        )
        if is_loopback:
            state.open_macro_cycle(no_gain_cycle_streak=int(evidence.get("no_gain_cycle_streak_effective") or 0))
            self._coord.phase_macro_cycle._record_cycle_strategy_for_current_cycle()
            log.info(
                "Coordinator: macro-cycle reloop %d -> %d (no_gain_streak=%d, gain_anchor=%.4f)",
                prior_cycle,
                state.macro_cycle,
                state.no_gain_cycle_streak,
                state.gain_at_cycle_start,
            )
        # Mirror the phase boundary into the operator-facing lifecycle log using the ENTER status (a point-in-time
        # marker, not a START/END interval).
        _phase_state.record_lifecycle_event(
            state,
            step=target,
            status=_phase_state.LIFECYCLE_STATUS_ENTER,
            phase=target,
            detail=f"reason={reason}" if reason else "",
        )
        state.save(self.session_dir)
        log.info(
            "Coordinator.phase: %s → %s (reason=%s)",
            prior or "<unset>",
            target,
            reason,
        )
        try:
            await self.bus.append_and_seq(
                Message.new(
                    "coordinator",
                    "*",
                    "event",
                    {
                        "kind": "phase_transition",
                        "from_phase": prior or "",
                        "to_phase": target,
                        "reason": reason,
                        "evidence": evidence,
                    },
                )
            )
        except Exception:
            log.exception("Coordinator: phase_transition event bus write failed")
        # Phase-entry side effects are additive; hook failures are logged only. They run on the transition itself,
        # which is what callers advancing into CLOSE rely on to see the sequencer's settlement once it returns.
        try:
            await self._on_phase_entered(
                from_phase=prior or "",
                to_phase=target,
                reason=reason or "",
                evidence=evidence if isinstance(evidence, dict) else None,
            )
        except Exception as exc:
            log.exception("Coordinator: _on_phase_entered hook failed")
            # This hook is also what closes the left phase's event, so a raise here is the case where that event never
            # got its exit evidence.
            self._coord.record_exception(stage="phase_entered", exc=exc)
        if is_loopback:
            await self._coord.phase_macro_cycle._run_cycle_soft_restart(
                prior_cycle=prior_cycle,
                new_cycle=state.macro_cycle,
            )

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
            loopback=bool(isinstance(evidence, dict) and evidence.get("loopback")),
        )

        if not self._on_exit:
            self._build_dispatch_tables()

        exit_hook = self._on_exit.get((from_phase or "").upper())
        if exit_hook:
            exit_hook(tr)

        entry_hook = self._on_enter.get(tr.to_phase)
        if entry_hook:
            await entry_hook(tr)

    def _cycle_directive(self) -> str:
        """The directive Orchestration wrote at the previous cycle's handoff, or empty."""
        memory = self.shared_state.orchestration_memory
        if memory.get("for_cycle") == self.shared_state.macro_cycle - 1:
            return str(memory.get("next_cycle_directive") or "")
        return ""

    def _reseed_orch_prompt_for_phase(self, to_phase: str) -> bool:
        """Re-scope the orchestration system prompt to the phase being entered."""
        phase = (to_phase or "").strip().upper()
        orch_prompt = self._coord.orch_prompt
        if not phase or orch_prompt.is_user_supplied:
            return False
        rebuild = orch_prompt.rebuild
        if rebuild is None:
            return False
        state = self.shared_state
        cycle = int(state.macro_cycle or 0)
        log_rows = list(state.cycle_strategy_log or [])
        prior_cycles = [r for r in log_rows if isinstance(r, dict) and int(r.get("cycle", -1) or -1) != cycle]
        focus_plan = self._coord.phase_macro_cycle._plan_cycle_focus()
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

    def _record_phase_entry_evidence(self, **kvs: Any) -> None:
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
