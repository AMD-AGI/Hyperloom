# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rebuilding in-memory state from the session's records when a session resumes."""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from hyperloom.orchestrator.lever import patch_owner_phase

from ..state.shared_state import SharedState
from ._integrate_promotion import _integrate_keep_variant

log = logging.getLogger(__name__)


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
            "rebuilt": False,  # set by replay_resume_if_needed()
        }

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

    def _replay_keep_from_result(self, result: dict[str, Any]) -> bool:
        """Replay an integrate KEEP its task row recorded into current_best/stack.

        Reconstructs the winning-variant dict from the recorded result and routes
        it through :meth:`lift_to_current_best`, which dedupes by
        ``(action, variant_name)`` — so replay is idempotent.

        Args:
            result: The result the ``integrate_patch`` task recorded for that KEEP.

        Returns:
            ``True`` when the replay appended a new stack entry, else ``False``.
        """
        tput = result.get("output_throughput")
        if not (isinstance(tput, (int, float)) and float(tput) > 0):
            return False
        bv = _replayed_integrate_variant(result, float(tput))
        if bv is None:
            return False
        before = len(self.shared_state.optimization_stack or [])
        if not self.lift_to_current_best(
            "integrate_patch",
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
        """Rebuild in-memory state once for a resumed session (abandon orphan dispatches, re-queue stranded work)."""
        if not (self._resumed_from["is_resume"] and not self._resumed_from["rebuilt"]):
            return
        self._resumed_from["rebuilt"] = True
        await self._resume_consistency_pass()
        # The stopped leg's rows read running until a reconcile pass fails them; re-entry would reuse one as live.
        await self._coord.reconciler.run(time.time())
        # Re-fire the KERNEL delegation hook when resuming parked in KERNEL_AGENT
        # (phase-entry side effects are bound to transitions, not resume). Runs
        # after the consistency pass so current_best/stack are already rebuilt.
        await self._resume_reenter_kernel_if_needed()
