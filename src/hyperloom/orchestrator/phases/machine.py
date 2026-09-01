# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase state-machine handler: initialisation, exit-condition scan/transition,
and the per-phase entry dispatcher (``_on_phase_entered``)."""

from __future__ import annotations
import logging as _logging
from typing import Any
from . import geak_rebench as _geak_rebench
from . import machine_state as _phase_state
from ..bus.message_bus import Message
from ..prompts import write_prompt_snapshot as _write_prompt_snapshot
from .base import PhaseHandler

log = _logging.getLogger(__name__)


class MachinePhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

    def _ensure_phase_initialised(self) -> None:
        """Set ``phase`` + persist ``phase_budget_pct`` once per session (idempotent).

        Raises:
            RuntimeError: When the session was recorded at a phase this build's
                machine does not have.
        """
        state = self.shared_state
        # Redistribute disabled phases' budget shares to the enabled work phases.
        # Done here (not in Coordinator.__init__) because the enablement flags on
        # shared_state are only authoritative after load_or_init. Idempotent, so
        # the per-tick refresh below is a no-op once applied.
        self._phase_budget_pct = _phase_state.redistribute_budget_pct(
            self._phase_budget_pct,
            optimize_enabled=self._optimize_enabled(),
            kernel_enabled=self._kernel_enabled(),
        )
        # Persist the phase budget so CLI flags land in state.json for resume parity.
        if not state.phase_budget_pct:
            state.phase_budget_pct = dict(self._phase_budget_pct)
        current = (state.phase or "").strip().upper()
        # Only an unset phase means fresh; an unknown one would otherwise
        # re-run PRELUDE over the earlier build's baseline and KEPT stack.
        if current and current not in _phase_state.PHASE_NAMES:
            raise RuntimeError(
                f"session was recorded at phase {current!r}, which this build's phase machine "
                f"does not have (known: {', '.join(_phase_state.PHASE_NAMES)}). "
                f"Resume it with the version that wrote it, or start a new session."
            )
        if current == _phase_state.PHASE_CLOSE:
            self._reopen_a_session_that_was_left_closed()
            current = _phase_state.PHASE_PRELUDE
        if current in _phase_state.PHASE_NAMES:
            # Already initialised; keep the CLI-side budget override authoritative.
            state.phase_budget_pct = dict(self._phase_budget_pct)
            try:
                state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: save after phase budget refresh failed")
            return
        # Fresh start; pre-phase-machine resume state is treated as fresh.
        state.record_phase_transition(
            to_phase=_phase_state.PHASE_PRELUDE,
            reason="phase_entered",
            evidence={"trigger": "fresh_session"},
        )
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("Coordinator: save after phase init failed")

    def _reopen_a_session_that_was_left_closed(self) -> None:
        """Put a session persisted in CLOSE back at the phase machine's entrance.

        CLOSE is terminal -- the machine has no transition out of it -- so a
        resumed session that loads it stays there for the whole leg. The run loop
        does not stop on CLOSE either, so what such a leg actually does is tick
        in a phase that admits only ``report``, ``session_breakdown`` and
        ``recover``: it spends its new clock on none of the work it was resumed
        for.

        Reopened at PRELUDE rather than at the phase CLOSE was entered from,
        because PRELUDE is the phase that works out where a session belongs. It
        exits on its first evaluation when the anchor the run needs is already
        measured, and it measures one when it is not. A session stopped for a
        cold anchor is the second case, and re-measuring is the whole reason its
        stop was worth resuming from.

        A session that still cannot fund the work is not kept open by this: the
        PRELUDE exits price the new clock and route it back to CLOSE, this time
        against the budget it actually has.

        Reached only from the constructor, so a session cannot reopen itself
        mid-run -- within one leg, CLOSE is entered long after this has run.
        """
        state = self.shared_state
        log.info(
            "Coordinator: session resumed in CLOSE, a phase with no way out; "
            "reopening at PRELUDE so the new budget can be spent on the work "
            "the earlier leg stopped short of."
        )
        state.record_phase_transition(
            to_phase=_phase_state.PHASE_PRELUDE,
            reason="phase_entered",
            evidence={"trigger": "resumed_from_close"},
        )
        # Locked True by the CLOSE sequencer and read by the end-of-run safety
        # nets as "the sequencer already wrote the breakdown". Carried into a leg
        # that then never reaches CLOSE, it suppresses the write that would have
        # stood in for it, and the leg produces no breakdown at all.
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
        workload = getattr(state, "model_name", "") or "unknown_model"
        hw = getattr(state, "gpu_type", "") or "unknown_gpu"
        extra_attrs = {
            "marathon_dispatch_id": getattr(state, "session_id", "") or "",
            "framework_name": getattr(state, "framework", "") or "",
            "model_class": getattr(state, "model_class", "") or "",
            "claw_session_id": getattr(state, "claw_session_id", "") or "",
            "sandbox_user_id": getattr(state, "sandbox_user_id", "") or "",
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
        except Exception:  # noqa: BLE001 — defensive; helper is itself best-effort
            log.exception(
                "Coordinator T0 fallback: run_t0_anchor raised (workload=%s, hw=%s); warm_start stays empty",
                workload,
                hw,
            )

    def _kernel_enabled(self) -> bool:
        """Whether kernel optimization is enabled for this run."""
        return bool(self.shared_state.kernel_enabled)

    def _optimize_enabled(self) -> bool:
        """Whether the optimisation phase is enabled for this run.

        Returns:
            ``True`` unless ``--no-framework-agent`` disabled it, collapsing
            the chain to KERNEL/SWEEP.
        """
        return bool(self.shared_state.framework_agent_phase_enabled)

    async def _inflight_kernel_task_ids(self) -> tuple[str, ...]:
        """Return the ids of queued/running tasks doing KERNEL-lane work.

        The task registry, not the kernel ledger, is what tells the idle guard
        whether the phase is genuinely busy: a kernel build or benchmark can
        occupy half an hour without touching a single ledger field while ticks
        keep advancing every few seconds. Queued counts as in flight alongside
        running, because a task waiting on a resource lane is work the phase is
        committed to, not dead air.

        The kind filter is ``KERNEL_LANE_TASK_KINDS``, not the phase's proposable
        actions: a Coordinator-owned lane is dispatched without ever being
        proposable, and its task is every bit as much work in flight. Reusing the
        proposable set here once made a running kernel_opt invisible to the idle
        guard the moment that action stopped being model-requestable.

        Returns:
            tuple[str, ...]: Sorted ids of in-flight kernel-lane tasks.
        """
        kinds = _phase_state.KERNEL_LANE_TASK_KINDS
        tasks = list(await self.tasks.queued()) + list(await self.tasks.running())
        return tuple(
            sorted(str(task.task_id) for task in tasks if str(getattr(task, "kind", "") or "").strip() in kinds)
        )

    async def _track_kernel_idle_streak(self) -> None:
        """Advance or reset the KERNEL idle-streak counters for this tick.

        ``exit_normal_kernel`` winds KERNEL down to SWEEP once the streak proves
        the phase has stopped moving. The streak is measured against an
        observable progress fingerprint rather than ``kernel_work_pending``: that
        predicate answers "does the ledger still list something unresolved",
        which stays true forever once an attempt can no longer be advanced, and
        it was resetting the counter on every one of 1130 consecutive idle ticks.

        Three outcomes per tick:

        * the fingerprint changed — something moved, so the streak restarts;
        * kernel-lane work is in flight — the phase is legitimately busy, so the
          streak is frozen AND its clock rebased, or a long build would silently
          accumulate idle time and trip the guard mid-build;
        * otherwise — nothing happened and nothing is running, so the streak
          grows.

        The tick counter can advance more than once per coordinator tick (the
        phase machine is scanned several times per tick); that imprecision is
        harmless because the wall-clock floor, not the tick count, is what
        actually decides when the guard may fire.
        """
        state = self.shared_state
        if str(getattr(state, "phase", "") or "").upper() != _phase_state.PHASE_KERNEL_AGENT:
            state.kernel_idle_ticks = 0
            state.kernel_progress_fingerprint = ""
            state.kernel_idle_since_unix = 0.0
            return
        import time as _time

        now = _time.time()
        inflight = await self._inflight_kernel_task_ids()
        fingerprint = _phase_state.compute_kernel_progress_fingerprint(
            state,
            inflight_task_ids=inflight,
        )
        if fingerprint != str(getattr(state, "kernel_progress_fingerprint", "") or ""):
            state.kernel_progress_fingerprint = fingerprint
            state.kernel_idle_ticks = 0
            state.kernel_idle_since_unix = now
            return
        if inflight or _phase_state.kernel_inline_step_running(state, now_unix=now):
            state.kernel_idle_since_unix = now
            return
        # Only reachable after a tick that opened the streak above, so
        # ``kernel_idle_since_unix`` is already stamped whenever the counter is
        # non-zero — the pairing the guard's wall-clock floor relies on.
        state.kernel_idle_ticks = int(getattr(state, "kernel_idle_ticks", 0) or 0) + 1

    async def _advance_phase_if_needed(self) -> None:
        """Scan exit conditions and transition phase at most once per tick.

        Priority order (Inv-8.2): global terminal > exit_terminal > exit_normal, per phase_state.compute_next_phase.
        """
        state = self.shared_state
        await self._track_kernel_idle_streak()
        optimize_enabled = self._optimize_enabled()
        next_phase = _phase_state.compute_next_phase(
            state,
            kernel_enabled=self._kernel_enabled(),
            budget_pct=self._phase_budget_pct,
            optimize_enabled=optimize_enabled,
        )
        if str(state.phase or "").upper() == _phase_state.PHASE_FRAMEWORK_AGENT:
            await self._maybe_enqueue_explore_research_scout()
            await self._maybe_force_stalled_domain_specialist()
        await self._maybe_enqueue_trajectory_reviewer()
        if next_phase is None:
            return
        target, reason, evidence = next_phase
        if target == (state.phase or "").upper():
            return  # already there
        prior = state.phase
        # Consume escalate hint after a hint-driven transition.
        if isinstance(evidence, dict) and (evidence.get("evidence") == "llm_escalation" or "hint" in evidence):
            state.consume_pending_escalate_hint()
        elif (
            str(prior or "").strip().upper() == _phase_state.PHASE_SWEEP
            and str(getattr(state, "pending_escalate_hint", "") or "").strip()
            == _phase_state.ESCALATE_HINT_SKIP_TO_CLOSE
        ):
            # SWEEP already had an honest closeout, so skip_to_close was
            # suppressed in _global_terminal. Consume it (not discard) here:
            # the hint's intended outcome -- reaching CLOSE -- did happen, just
            # via the more honest reason, so the next phase must not re-evaluate
            # it as if it were still unclaimed.
            state.consume_pending_escalate_hint()
        elif state.pending_escalate_hint and target != _phase_state.PHASE_FRAMEWORK_AGENT:
            # ``exit_normal_optimize`` is the hint's only consumer, so a
            # transition away from that phase leaves it unclaimable; keeping it
            # would let an unrelated phase re-evaluate it. A transition *into*
            # that phase is the opposite case: discarding there would drop the
            # hint on the doorstep of the one rule that reads it.
            discarded_hint = state.discard_pending_escalate_hint()
            log.info(
                "phase_machine: discarded stale pending_escalate_hint=%r on unrelated transition %s -> %s (reason=%s)",
                discarded_hint,
                prior,
                target,
                reason,
            )
        # Terminal transition (target=CLOSE): mirror the stop_reason onto state.
        if (
            target == _phase_state.PHASE_CLOSE
            and isinstance(evidence, dict)
            and evidence.get("terminal")
            and reason
            and _phase_state.is_valid_stop_reason(reason)
            and not state.stop_reason
        ):
            state.set_stop_reason(reason)
        # A cyclic config-arm plateau winds the cycle down with ``switch_bottleneck``:
        # record the plateaued bottleneck so the next cycle steers specialists off it.
        if isinstance(evidence, dict) and evidence.get("switch_bottleneck"):
            try:
                state.mark_bottleneck_switch(
                    prev_bottleneck=state.current_top_bottleneck(),
                )
                log.info(
                    "plateau → bottleneck switch flagged (off %r)",
                    state.last_cycle_bottleneck,
                )
            except Exception:  # noqa: BLE001 — advisory bookkeeping is best-effort
                log.exception("mark_bottleneck_switch failed")
        is_loopback = bool(isinstance(evidence, dict) and evidence.get("loopback"))
        if is_loopback:
            prior_cycle = int(getattr(state, "macro_cycle", 0) or 0)
            self._apply_macro_cycle_reloop(evidence)
            await self._run_cycle_soft_restart(
                prior_cycle=prior_cycle,
                new_cycle=int(getattr(state, "macro_cycle", 0) or 0),
            )
        # Also persist the no-gain streak on a cyclic-mode terminal close so
        # a subsequent resume sees the convergence state.
        elif (
            target == _phase_state.PHASE_CLOSE
            and isinstance(evidence, dict)
            and "no_gain_cycle_streak_effective" in evidence
        ):
            state.no_gain_cycle_streak = int(evidence.get("no_gain_cycle_streak_effective", 0) or 0)
        allowed_kinds = _phase_state.PHASE_ALLOWED_ACTIONS.get(target, frozenset())
        target_phase = str(target or "").strip().upper()
        cancelled = await self.tasks.cancel_queued_not_allowed(
            allowed_kinds=allowed_kinds,
            reason=f"phase_transition:{str(prior or '').strip().upper()}->{target}",
            spare_queued=lambda _task_id, kind, params: _geak_rebench.spare_geak_rebench_on_phase_transition(
                target_phase=target_phase,
                kind=kind,
                params=params,
            ),
        )
        if cancelled:
            log.info(
                "Coordinator.phase: cancelled %d queued task(s) incompatible with %s",
                len(cancelled),
                target,
            )
        state.record_phase_transition(
            to_phase=target,
            reason=reason,
            evidence=evidence,
        )
        # Mirror the phase boundary into the operator-facing lifecycle log using
        # the ENTER status (a point-in-time marker, not a START/END interval).
        # Best-effort; never rolls back the transition.
        try:
            state.record_lifecycle_event(
                step=target,
                status=_phase_state.LIFECYCLE_STATUS_ENTER,
                phase=target,
                detail=f"reason={reason}" if reason else "",
            )
        except Exception:  # noqa: BLE001 — defensive
            log.debug("Coordinator: lifecycle phase emit failed", exc_info=True)
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("Coordinator: save after phase transition failed")
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
        except Exception:  # noqa: BLE001 — defensive
            log.exception("Coordinator: phase_transition event bus write failed")
        # Phase-entry side effects are additive; hook failures are logged only.
        try:
            await self._on_phase_entered(from_phase=prior or "", to_phase=target)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("Coordinator: _on_phase_entered hook failed")

    async def _on_phase_entered(self, *, from_phase: str, to_phase: str) -> None:
        """Fire per-phase entry side effects (pure dispatcher; hooks catch + log internally). CLOSE runs the 7-step sequencer (sets close_sequence_done).

        Args:
            from_phase: The phase being left.
            to_phase: The phase being entered; selects which per-phase entry
                hook fires.
        """
        # Orchestration checkpoint at the phase seam.
        try:
            await self._maybe_checkpoint_orchestration(
                tick=int(getattr(self.shared_state, "tick", 0) or 0),
                phase_changed=True,
            )
        except Exception:  # noqa: BLE001
            log.exception("Coordinator: phase-boundary checkpoint failed")
        # Cache-safe here only because the checkpoint above already re-seeded
        # the conversation, so the cached prefix is rebuilt regardless.
        try:
            self._reseed_orch_prompt_for_phase(to_phase)
        except Exception:  # noqa: BLE001 — prompt scoping is best-effort
            log.exception("Coordinator: phase-boundary prompt reseed failed")

        target = (to_phase or "").upper()
        if target == _phase_state.PHASE_FRAMEWORK_AGENT:
            await self._on_enter_framework(from_phase=from_phase)
        elif target == _phase_state.PHASE_KERNEL_AGENT:
            await self._on_enter_kernel(from_phase=from_phase)
        elif target == _phase_state.PHASE_SWEEP:
            await self._on_enter_sweep(from_phase=from_phase)
        elif target == _phase_state.PHASE_CLOSE:
            await self._on_enter_close(from_phase=from_phase)

    def _reseed_orch_prompt_for_phase(self, to_phase: str) -> bool:
        """Re-scope the orchestration system prompt to the phase being entered.

        Carries the current macro-cycle and cycle directive over unchanged; only
        the phase scope moves. Snapshots the installed scope so the artefacts
        record what each phase actually ran under. Skips a user-supplied
        ``--orch-prompt``.

        Args:
            to_phase: The phase being entered.

        Returns:
            ``True`` when the override was rebuilt, else ``False``.
        """
        phase = (to_phase or "").strip().upper()
        if not phase or getattr(self, "_orch_prompt_is_user_supplied", False):
            return False
        rebuild = getattr(self, "_rebuild_orch_prompt", None)
        overrides = getattr(self, "system_prompt_overrides", None)
        if rebuild is None or not isinstance(overrides, dict):
            return False
        state = self.shared_state
        scoped = rebuild(
            macro_cycle=state.macro_cycle,
            cycle_directive=str(state.orchestration_memory.get("next_cycle_directive", "") or ""),
            phase=phase,
        )
        overrides["orchestration"] = scoped
        _write_prompt_snapshot(self.session_dir, "orchestration", scoped, phase=phase)
        log.info("orchestration prompt re-scoped for phase=%s", phase)
        return True

    def _record_phase_entry_evidence(self, **kvs: Any) -> None:
        """Merge ``kvs`` into the latest phase_history row's evidence dict (no-op when empty).

        Args:
            **kvs: Arbitrary key/value pairs merged into the latest
                phase_history row's evidence dict.
        """
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
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception(
                "phase entry evidence: SharedState.save failed for kvs=%r",
                kvs,
            )
