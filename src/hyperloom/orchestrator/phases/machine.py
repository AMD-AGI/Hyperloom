# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Phase state-machine handler: initialisation, exit-condition scan/transition,
and the per-phase entry dispatcher (``_on_phase_entered``)."""

from __future__ import annotations
import logging as _logging
from typing import Any
from . import machine_state as _phase_state
from ..bus.message_bus import Message
from .base import PhaseHandler

log = _logging.getLogger(__name__)


class MachinePhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

    # phase state machine
    def _ensure_phase_initialised(self) -> None:
        """Set ``phase`` + persist ``phase_budget_pct`` once per session (idempotent)."""
        state = self.shared_state
        # Phase budget normalised + persisted so CLI flags land in state.json for resume parity.
        if not state.phase_budget_pct:
            state.phase_budget_pct = dict(self._phase_budget_pct)
        current = (state.phase or "").strip().upper()
        if current in _phase_state.PHASE_NAMES:
            # Already initialised; keep CLI-side budget override authoritative.
            state.phase_budget_pct = dict(self._phase_budget_pct)
            try:
                state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: save after phase budget refresh failed")
            return
        # Fresh start; pre-phase-machine resume state is treated as fresh (cross-version unsupported).
        state.record_phase_transition(
            to_phase=_phase_state.PHASE_PRELUDE,
            reason="phase_entered",
            evidence={"trigger": "fresh_session"},
        )
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("Coordinator: save after phase init failed")

    def _ensure_cortex_t0_anchored(self) -> None:
        """Defensive T0 anchor for SDK callers constructed without cli plumbing. Skips when cortex_kb is None or cortex_session_id set."""
        client = self.cortex_kb
        if client is None or not getattr(client, "enabled", True):
            return
        state = self.shared_state
        if (state.cortex_session_id or "").strip():
            # cli already T0'd or resume picked up the sid; gate up here to skip the import.
            return
        # Derive workload / hw from SharedState.
        workload = getattr(state, "model_name", "") or "unknown_model"
        hw = getattr(state, "gpu_type", "") or "unknown_gpu"
        # marathon_dispatch_id mirrors the cli path: the hyperloom-internal manifest session id.
        extra_attrs = {
            "marathon_dispatch_id": getattr(state, "session_id", "") or "",
            "framework_name": getattr(state, "framework", "") or "",
            "model_class": getattr(state, "model_class", "") or "",
            "claw_session_id": getattr(state, "claw_session_id", "") or "",
            "sandbox_user_id": getattr(state, "sandbox_user_id", "") or "",
            # boot_origin is a dev-debug label, NOT written to KB; distinguishes SDK-fallback from cli path.
            "boot_origin": "coordinator_fallback",
        }
        try:
            # Reuse the held dispatcher so T0 anchors the SAME local store KEEP/REVERT/CLOSE target.
            from ..knowledge.cortex_t0 import run_t0_anchor

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
        """Whether the kernel_agent role is registered and enabled.

        Returns:
            ``True`` if the kernel_agent role exists and the persisted
            ``kernel_enabled`` flag is set.
        """
        # Mirror persisted kernel_enabled flag; --no-kernel removes the kernel_agent role.
        return "kernel_agent" in self.role_registry and bool(self.shared_state.kernel_enabled)

    def _explore_enabled(self) -> bool:
        """Whether the EXPLORE phase is enabled for this run.

        Returns:
            ``True`` unless ``--no-explore`` disabled it (collapsing to
            KERNEL/SWEEP).
        """
        # Mirror persisted explore_enabled flag; --no-explore collapses to KERNEL/SWEEP. EXPLORE is a phase, not a role.
        return bool(self.shared_state.explore_enabled)

    async def _advance_phase_if_needed(self) -> None:
        """Scan exit conditions and transition phase at most once per tick.

        Priority order (Inv-8.2): abort > exit_terminal > exit_normal, per phase_state.compute_next_phase.
        """
        state = self.shared_state
        max_hours_arg: float | None = None
        mm = float(getattr(state, "max_minutes", 0) or 0.0)
        if mm > 0:
            max_hours_arg = mm / 60.0
        next_phase = _phase_state.compute_next_phase(
            state,
            kernel_enabled=self._kernel_enabled(),
            budget_pct=self._phase_budget_pct,
            framework_agent_phase_enabled=bool(state.framework_agent_phase_enabled),
            explore_enabled=self._explore_enabled(),
            max_hours=max_hours_arg,
        )
        if str(state.phase or "").upper() == "EXPLORE":
            await self._maybe_enqueue_explore_research_scout()
            await self._maybe_force_stalled_domain_specialist()
        await self._maybe_enqueue_trajectory_reviewer()
        # (default OFF) FRAMEWORK config-exploration lane: before actually
        # leaving FRAMEWORK_AGENT, run explore-style config-grid rounds so
        # FRAMEWORK gains the EXPLORE config-search capability. Placed after the
        # trajectory reviewer so holding the phase never skips it. No-op unless
        # framework_config_exploration_enabled is set (default flow unchanged).
        if self._framework_config_lane_should_engage(next_phase):
            if await self._maybe_hold_for_framework_config_lane():
                return
        if next_phase is None:
            return
        target, reason, evidence = next_phase
        if target == (state.phase or "").upper():
            return  # already there
        prior = state.phase
        # Consume escalate hint after a hint-driven transition so the next tick re-evaluates fresh.
        if isinstance(evidence, dict) and (evidence.get("evidence") == "llm_escalation" or "hint" in evidence):
            state.consume_pending_escalate_hint()
        # Terminal transition (target=CLOSE): mirror vocab stop_reason onto state via ENUM-validated writer.
        if (
            target == _phase_state.PHASE_CLOSE
            and isinstance(evidence, dict)
            and evidence.get("terminal")
            and reason
            and _phase_state.is_valid_stop_reason(reason)
            and not state.stop_reason
        ):
            state.set_stop_reason(reason)
        # A SWEEP→EXPLORE loopback opens a new macro-cycle. Bump the cycle
        # counter + persist the no-gain streak BEFORE recording the
        # transition so the new EXPLORE phase rows carry the new cycle number.
        # A cyclic EXPLORE plateau winds the cycle down with
        # ``switch_bottleneck`` — record the bottleneck we plateaued on so the
        # next macro-cycle's orchestration prompt redirects specialists off it.
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
        state.record_phase_transition(
            to_phase=target,
            reason=reason,
            evidence=evidence,
        )
        # Mirror the phase boundary into the operator-facing
        # lifecycle log so a launcher poll surfaces "entered <phase>" in
        # chat (with the human-friendly label) alongside the step-level
        # events. Uses the ENTER status (not START): a phase boundary is a
        # point-in-time marker, not a paired START/END interval, so it must
        # not read as "still running" forever. Best-effort; must never roll
        # back the transition.
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
        # Phase-entry side effects are additive; hook failures are logged but never roll back the transition.
        try:
            await self._on_phase_entered(from_phase=prior or "", to_phase=target)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("Coordinator: _on_phase_entered hook failed")

    async def _on_phase_entered(self, *, from_phase: str, to_phase: str) -> None:
        """Fire per-phase entry side effects (pure dispatcher; hooks catch + log internally). CLOSE runs the 5-step sequencer (sets close_sequence_done).

        Args:
            from_phase: The phase being left.
            to_phase: The phase being entered; selects which per-phase entry
                hook fires.
        """
        # Orchestration checkpoint at the phase seam; runs before per-phase side effects.
        try:
            await self._maybe_checkpoint_orchestration(
                tick=int(getattr(self.shared_state, "tick", 0) or 0),
                phase_changed=True,
            )
        except Exception:  # noqa: BLE001
            log.exception("Coordinator: phase-boundary checkpoint failed")

        target = (to_phase or "").upper()
        if target == _phase_state.PHASE_FRAMEWORK_AGENT:
            await self._on_enter_framework(from_phase=from_phase)
        elif target == _phase_state.PHASE_EXPLORE:
            await self._on_enter_explore(from_phase=from_phase)
        elif target == _phase_state.PHASE_KERNEL_AGENT:
            await self._on_enter_kernel(from_phase=from_phase)
        elif target == _phase_state.PHASE_SWEEP:
            await self._on_enter_sweep(from_phase=from_phase)
        elif target == _phase_state.PHASE_CLOSE:
            await self._on_enter_close(from_phase=from_phase)

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
