# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rebuilding in-memory state from the event log when a session resumes."""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from hyperloom.orchestrator.lever import patch_lever_kind, patch_owner_phase

from ..state.shared_state import SharedState
from ._integrate_promotion import _patch_recipe_delta
from ._keep_patch_column import _source_layer_handles
from .proposals import PendingProposal

log = logging.getLogger(__name__)


def _replayed_explore_variant(result: Mapping[str, Any]) -> dict[str, Any] | None:
    best = result.get("best_variant")
    if not isinstance(best, dict) or not best.get("name"):
        return None
    return dict(best)


def _replay_origin_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    domain = str(result.get("domain") or result.get("source_domain") or "").strip()
    provenance = str(result.get("provenance") or "").strip()
    if domain and not provenance.startswith("specialist:"):
        provenance = f"specialist:{domain}"
    fields: dict[str, Any] = {"provenance": provenance or "integrate_patch", "scope": "source_patch"}
    lever_kind = patch_lever_kind(result)
    if lever_kind:
        fields["lever_kind"] = lever_kind
    source_phase = patch_owner_phase(result)
    if source_phase:
        fields["source_phase"] = source_phase
    else:
        fields["recipe_publishable"] = False
    if domain:
        fields["domain"] = domain
    gap_layer = str(result.get("gap_layer") or "").strip()
    if gap_layer:
        fields["gap_layer"] = gap_layer
    if result.get("framework_agent_authoring"):
        fields["framework_agent_authoring"] = True
    if result.get("enablement") or result.get("enablement_landing"):
        fields["recipe_publishable"] = False
    return fields


def _replayed_integrate_variant(result: Mapping[str, Any], tput: float) -> dict[str, Any] | None:
    from hyperloom.common.perf_metric import constraint_fields, graded_axes_of

    specialist_task_id = str(result.get("specialist_task_id") or "")
    if not specialist_task_id:
        return None
    bench = result.get("bench_result") or result
    return {
        "name": specialist_task_id,
        "candidate_extra_server_args": str(result.get("extra_server_args_applied") or ""),
        "candidate_extra_envs": dict(result.get("extra_envs_applied") or {}),
        "recipe_delta": _patch_recipe_delta(result, result),
        "extra_envs": dict(result.get("extra_envs_applied") or {}),
        "tput": float(tput),
        **graded_axes_of(bench),
        # ``graded_axes_of`` carries the throughput axes only; the latency and
        # power budgets grade on these and fail closed without them.
        **constraint_fields(bench),
        "workspace": result.get("workspace"),
        **_source_layer_handles(result),
        **_replay_origin_fields(result),
    }


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
    def detect_resume_state(self) -> dict[str, Any]:
        """Synchronously inspect persistence to determine if this is a resume (non-blocking).

        Returns:
            A dict with ``is_resume``, ``event_count``, ``state_json_present``
            and ``rebuilt`` (the last set later by :meth:`replay_for_resume`).
        """
        event_count = self.bus.count_sync()
        state_path = SharedState.state_path(self.session_dir)
        self._resumed_from = {
            "is_resume": event_count > 0 or state_path.exists(),
            "event_count": event_count,
            "state_json_present": state_path.exists(),
            "rebuilt": False,  # set by replay_for_resume()
        }
        return self._resumed_from

    async def replay_for_resume(self) -> dict[str, Any]:
        """Walk the event log to reconstruct ``CoordinatorState.pending_proposals``. Idempotent; a proposal is undecided when no review_verdict targets it.

        Returns:
            A dict summarising the replay: ``is_resume``, ``event_count``,
            ``state_json_present``, ``pending_restored`` (count rebuilt) and
            ``verdicts_seen``.
        """
        proposal_msgs = await self.bus.tail(topic="proposal", n=10_000)
        verdicts = await self.bus.tail(topic="review_verdict", n=10_000)

        decided_ids: set[str] = set()
        verdict_by_target: dict[str, str] = {}
        for v in verdicts:
            target = v.payload.get("target_proposal_msg_id")
            if not target:
                continue
            # Verdicts with a verdict_map but no summary are treated as needs_review.
            summary = v.payload.get("verdict") or ""
            if not summary and isinstance(v.payload.get("verdict_map"), dict):
                summary = "needs_review"
            verdict_by_target[target] = summary
            decided_ids.add(target)

        rebuilt = 0
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
            rebuilt += 1

        self._resumed_from["rebuilt"] = True
        self._resumed_from["pending_restored"] = rebuilt
        return {
            "is_resume": self._resumed_from["is_resume"],
            "event_count": self._resumed_from["event_count"],
            "state_json_present": self._resumed_from["state_json_present"],
            "pending_restored": rebuilt,
            "verdicts_seen": len(verdicts),
        }

    def _publish_recovered_keeps(self, report: dict[str, Any]) -> None:
        state = self.shared_state
        resume_state_durable = True
        try:
            state.save(self.session_dir)
        except Exception:
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
        # The stopped leg's rows still read running until a reconcile pass fails them; the re-entry below would reuse
        # one as live, and the first tick's reconcile would then leave KERNEL with nothing in flight.
        await self._coord.reconciler.run(time.time())
        # Re-fire the KERNEL delegation hook when resuming parked in KERNEL_AGENT
        # (phase-entry side effects are bound to transitions, not resume). Runs
        # after the consistency pass so current_best/stack are already rebuilt.
        await self._resume_reenter_kernel_if_needed()
