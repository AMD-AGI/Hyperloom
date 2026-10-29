# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Promoting an explore result: its winners, the lift they earn, the revalidation they reopen."""

from __future__ import annotations

import logging
from typing import Any, Mapping, NamedTuple

from hyperloom.inference_optimizer.session.optimization_journal import Verdict

from ..actions.executors.explore import STACK_REVALIDATE_SOURCE, is_stack_revalidation
from ._promote_outcome import _PromoteOutcome
from ..state.task_registry import Task

log = logging.getLogger(__name__)


class _ExploreLift(NamedTuple):
    promoted: bool
    last_lifted: dict[str, Any] | None


def _explore_audit_extras(result: Mapping[str, Any]) -> dict[str, Any]:
    winners = result.get("winners") or []
    best_winner = result.get("best_variant")
    return {
        "round_id": str(result.get("round_id") or ""),
        "winners_count": (len(winners) if isinstance(winners, list) else 0),
        "losers_count": len(result.get("losers") or []),
        "skipped_dup_count": len(result.get("skipped_dup") or []),
        "best_variant_name": (best_winner.get("name") if isinstance(best_winner, dict) else None),
        "best_gain_pct_vs_base": result.get("best_gain_pct"),
        "output_throughput": result.get("output_throughput"),
        "explore_grid_exhausted": bool(result.get("explore_grid_exhausted")),
    }


# Default resume-drift floor (%): a re-measured current_best below this fraction of its recorded tput is flagged as
# drift.
_DEFAULT_RESUME_DRIFT_FLOOR_PCT: float = 95.0


class ExplorePromotionMixin:
    """Promoting an explore sweep's outcome."""

    async def _promote_explore(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote an explore result: ledger increment, winners, current_best lift, resume revalidation."""
        # The Coordinator is single-writer for explore_search.accepted,
        # current_best and optimization_stack.
        update = result.get("explore_search_update")
        if isinstance(update, dict):
            self.shared_state.apply_explore_search_update(update)
        self._record_framework_lever_attributions(result)
        winners = result.get("winners") or []
        # A stack revalidation confirms the existing stack and current_best
        # rather than adding a variant, so it never promotes: the accept and
        # lift path below is skipped, or a rebench would append a duplicate
        # stack entry and re-lift current_best.
        is_revalidation_task = task is not None and is_stack_revalidation(task.params)
        if is_revalidation_task:
            stale = await self._warn_on_stale_revalidation(task)
            # A GEAK revalidation (2b) rules on its own rebench and publishes the
            # verdict itself; a native one reconciles the watermark here.
            if bool((task.params or {}).get("geak_fallback")):
                outcome.verdict = await self._rule_on_geak_rebench(result, task=task)
                outcome.changed = True
                return
            await self._reconcile_revalidated_watermark(result, stale=stale)
        lift = _ExploreLift(promoted=False, last_lifted=None)
        if isinstance(winners, list) and winners and not is_revalidation_task:
            lift = self._accept_explore_winners(result, task, outcome)
        self.shared_state.note_explore_outcome(promoted=lift.promoted)
        # A round with no measured variant is not a data point for the plateau window.
        if not is_revalidation_task and (winners or result.get("losers")):
            self.shared_state.gain_gated_action_count += 1
        if lift.last_lifted is not None:
            # The last successful lift owns every axis of the validated measurement.
            await self._validate_and_watermark(
                float(lift.last_lifted["tput"]),
                lift.last_lifted,
                source="explore_keep",
                measurement_basis="e2e_decision_round",
                watermark_reason="explore_keep_watermark",
            )
        if lift.promoted:
            outcome.audit_decision = "promoted"
            outcome.verdict = Verdict.ADOPTED
        elif winners and not is_revalidation_task:
            outcome.audit_decision = "no_promote"
            outcome.verdict = Verdict.REFUSED
        else:
            outcome.audit_decision = "discarded"
            outcome.verdict = Verdict.RECORDED if is_revalidation_task else Verdict.REVERTED
        outcome.audit_extras = _explore_audit_extras(result)

    def _record_framework_lever_attributions(self, result: Mapping[str, Any]) -> None:
        for attribution in result.get("framework_lever_attributions") or []:
            if not isinstance(attribution, dict):
                continue
            self.shared_state.record_framework_lever_attribution(
                str(attribution.get("switch") or ""),
                gain_pct=attribution.get("gain_pct"),
            )

    async def _warn_on_stale_revalidation(self, task: "Task") -> bool:
        measured_generation = (task.params or {}).get("recipe_generation")
        working_generation = int(self.shared_state.working_recipe_generation or 0)
        if measured_generation is None or int(measured_generation) == working_generation:
            return False
        log.warning(
            "stack revalidation %s measured recipe generation %s; current_best is now %s -- not validating",
            task.task_id,
            measured_generation,
            working_generation,
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "stale_stack_revalidation",
                "task_id": task.task_id,
                "measured_recipe_generation": int(measured_generation),
                "working_recipe_generation": working_generation,
                "geak_fallback": bool((task.params or {}).get("geak_fallback")),
            },
        )
        return True

    async def _reconcile_revalidated_watermark(self, result: dict, *, stale: bool) -> None:
        measured = result.get("output_throughput")
        if stale or not isinstance(measured, (int, float)) or measured <= 0:
            return
        if self.shared_state.baseline_tput <= 0:
            return
        self.validate(measured, result, source=STACK_REVALIDATE_SOURCE)
        recorded = self.shared_state.current_best.get("tput")
        if not isinstance(recorded, (int, float)) or recorded <= 0:
            return
        floor = _DEFAULT_RESUME_DRIFT_FLOOR_PCT
        if float(measured) >= float(recorded) * floor / 100.0:
            return
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "current_best_drift",
                "severity": "high",
                "measured_tput": float(measured),
                "recorded_tput": float(recorded),
                "floor_pct": floor,
            },
        )

    def _record_accepted_winners(self, result: Mapping[str, Any], winners: list[dict[str, Any]]) -> None:
        round_id = str(result.get("round_id") or "")
        for winner in winners:
            accepted = dict(winner)
            accepted.setdefault("accepted_at_round", round_id)
            accepted.setdefault("provenance", winner.get("provenance") or "llm_direct")
            self.shared_state.record_explore_accepted(accepted)
            # A specialist-provenance KEEP zeroes that domain's rounds_since_last_keep counter.
            provenance = str(accepted.get("provenance") or "")
            if provenance.startswith("specialist:"):
                self.shared_state.note_domain_keep(provenance.split(":", 1)[1].strip())

    def _accept_explore_winners(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> _ExploreLift:
        winners = [winner for winner in result.get("winners") or [] if isinstance(winner, dict)]
        self._record_accepted_winners(result, winners)
        gap_canonical_id = str((task.params or {}).get("gap_canonical_id") or "").strip() if task is not None else ""
        promoted = False
        last_lifted: dict[str, Any] | None = None
        for winner in winners:
            winner_tput = winner.get("tput")
            if not isinstance(winner_tput, (int, float)) or float(winner_tput) <= 0:
                continue
            entry = dict(winner)
            if task is not None:
                entry["task_id"] = str(task.task_id or "")
            if not self.lift_to_current_best("explore", float(winner_tput), entry, gap_canonical_id=gap_canonical_id):
                continue
            promoted = True
            last_lifted = entry
            if winner.get("fingerprint"):
                outcome.adopted_variants.add(str(winner["fingerprint"]))
        return _ExploreLift(promoted=promoted, last_lifted=last_lifted)
