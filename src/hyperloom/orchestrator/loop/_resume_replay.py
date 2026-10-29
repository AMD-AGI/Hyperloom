# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rebuilding in-memory state from the event log when a session resumes."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from hyperloom.orchestrator.lever import patch_owner_phase

from ..state.shared_state import SharedState
from ._integrate_promotion import _integrate_keep_variant
from .proposals import PendingProposal

log = logging.getLogger(__name__)


def _replayed_explore_variant(result: Mapping[str, Any]) -> dict[str, Any] | None:
    best = result.get("best_variant")
    if not isinstance(best, dict) or not best.get("name"):
        return None
    return dict(best)


def _replayed_integrate_variant(result: Mapping[str, Any], tput: float) -> dict[str, Any] | None:
    if not str(result.get("specialist_task_id") or ""):
        return None
    # Ownership is normalised from the recorded evidence; a KEEP whose owner cannot be recovered is not publishable.
    source_phase = patch_owner_phase(result)
    variant = _integrate_keep_variant(result, None, result.get("bench_result") or result, tput, source_phase)
    if not source_phase:
        variant["recipe_publishable"] = False
    return variant


class ResumeReplayMixin:
    """Replaying a resumed session's log."""

    # ------------------------------------------------------------------
    # Resume / replay (folded in from the former ResumeCollaborator).
    # Three semantic boundaries live below: the live-promote / replay path
    # (``_replay_keep_from_result``), the resume-reconcile path
    # (``_resume_consistency_pass`` + its recover helpers), and the
    # current_best lift path (``current_best_launch_config`` /
    # ``build_env_spec``).
    # ------------------------------------------------------------------
    def detect_resume_state(self) -> None:
        """Synchronously inspect persistence to determine if this is a resume (non-blocking)."""
        self._resumed_from = {
            "is_resume": self.bus.count_sync() > 0 or SharedState.state_path(self.session_dir).exists(),
            "rebuilt": False,  # set by replay_for_resume()
        }

    async def replay_for_resume(self) -> None:
        """Walk the event log to reconstruct ``CoordinatorState.pending_proposals``. Idempotent; a proposal is undecided when no review_verdict targets it."""
        proposal_msgs = await self.bus.tail(topic="proposal", n=10_000)
        verdicts = await self.bus.tail(topic="review_verdict", n=10_000)

        decided_ids = {v.payload["target_proposal_msg_id"] for v in verdicts if v.payload.get("target_proposal_msg_id")}
        self.state.pending_proposals.clear()
        for p in proposal_msgs:
            if p.msg_id in decided_ids:
                continue
            payload = p.payload or {}
            self.state.pending_proposals[p.msg_id] = PendingProposal(
                proposal_msg_id=p.msg_id,
                from_agent=p.from_agent,
                action_name=str(payload.get("action_name", "")),
                predicted_gain_pct=float(payload.get("predicted_gain_pct", 0.0)),
                payload=dict(payload),
            )

        self._resumed_from["rebuilt"] = True

    def _publish_recovered_keeps(self, report: dict[str, Any]) -> None:
        state = self.shared_state
        resume_state_durable = True
        try:
            state.save(self.session_dir)
        except OSError:
            resume_state_durable = False
            log.exception("Coordinator: pre-outbox resume save failed")
            report["warnings"].append({"kind": "resume_pre_outbox_save_failed"})
        pending_before = len(state.kb_stage_outbox or [])
        if not (pending_before and resume_state_durable):
            return
        self._drain_agent_keep_outbox()
        pending_after = len(state.kb_stage_outbox or [])
        if pending_after:
            report["warnings"].append({"kind": "kb_stage_outbox_incomplete", "pending": pending_after})
        else:
            report["fixes"].append({"kind": "reconciled_kb_stage_outbox", "count": pending_before})

    def _replay_keep_from_result(self, kind: str, result: dict[str, Any]) -> bool:
        """Replay a recorded KEEP delegated-result into current_best/stack.

        Reconstructs the winning-variant dict from a persisted ``delegated_result``
        and routes it through :meth:`lift_to_current_best`, which dedupes by
        ``(action, variant_name)`` — so replay is idempotent. Used by both the
        pending-integrate (Gap C) and orphaned-KEEP (Gap B) resume recovery
        paths. Returns ``True`` only when a new stack entry was appended.

        Args:
            kind: The originating action kind (``integrate_patch`` / ``explore``
                / ``framework``).
            result: The recorded delegated result payload for that KEEP.

        Returns:
            ``True`` when the replay appended a new stack entry, else ``False``.
        """
        if not isinstance(result, dict):
            return False
        tput = result.get("output_throughput")
        if not (isinstance(tput, (int, float)) and float(tput) > 0):
            return False
        if kind == "explore":
            bv = _replayed_explore_variant(result)
        elif kind == "integrate_patch":
            bv = _replayed_integrate_variant(result, float(tput))
        else:
            return False
        if bv is None:
            return False
        before = len(self.shared_state.optimization_stack or [])
        if not self.lift_to_current_best(
            kind,
            float(tput),
            bv,
            gap_canonical_id=str(result.get("gap_canonical_id") or ""),
        ):
            return False
        return len(self.shared_state.optimization_stack or []) > before

    @property
    def resumed_from(self) -> dict[str, Any]:
        """Read-only snapshot of resume detection (set by ``__init__``).

        Returns:
            A copy of the resume-detection dict so callers cannot mutate
            internal state.
        """
        return dict(self._resumed_from)

    # Bounded test interface
    async def replay_resume_if_needed(self) -> None:
        """Rebuild in-memory state once for a resumed session (replay log + abandon orphan dispatches)."""
        if not (self._resumed_from["is_resume"] and not self._resumed_from["rebuilt"]):
            return
        await self.replay_for_resume()
        await self._resume_consistency_pass()
        # Re-fire the KERNEL delegation hook when resuming parked in KERNEL_AGENT
        # (phase-entry side effects are bound to transitions, not resume). Runs
        # after the consistency pass so current_best/stack are already rebuilt.
        await self._resume_reenter_kernel_if_needed()
