# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase-independent internal task handler: research-scout, static-recon, and trajectory-reviewer auto-enqueue helpers used across multiple phases."""

from __future__ import annotations
import logging as _logging
import os
import re
from typing import Any

from hyperloom.common.env import env_bool

from ..state.task_registry import Task, TaskSpec
from ..collaborator import CoordinatorCollaborator
from . import machine_state as _phase_state

log = _logging.getLogger(__name__)


def _bridge_symptom(cand: dict[str, Any], predicate_file: str, why: str) -> str:
    """The operator-readable account of one un-bridged switch."""
    predicate_name = str(cand.get("predicate_name") or "").strip()
    parts = [
        f"Un-bridged switch in {predicate_file}" + (f" ({predicate_name})" if predicate_name else "") + f": {why}",
    ]
    consequence = str(cand.get("consequence") or "").strip()
    if consequence:
        parts.append(f"Consequence: {consequence}")
    bridge_sketch = str(cand.get("bridge_sketch") or "").strip()
    if bridge_sketch:
        parts.append(f"Bridge: {bridge_sketch}")
    return " ".join(parts)[:1200]


def _bridge_candidate_gap(cand: dict[str, Any], idx: int) -> dict[str, Any] | None:
    """The gap row a static-recon bridge candidate seeds, or None when it is not actionable."""
    predicate_file = str(cand.get("predicate_file") or "").strip()
    why = str(cand.get("why_disabled_here") or "").strip()
    # A candidate without a source anchor + explanation is not actionable.
    if not predicate_file or not why:
        return None
    raw_id = str(cand.get("id") or f"cand{idx}").strip() or f"cand{idx}"
    slug = "".join(c if (c.isalnum() or c in "._-") else "_" for c in raw_id)[:80]
    return {
        "canonical_id": f"gap.static_recon.{slug}",
        "symptom": _bridge_symptom(cand, predicate_file, why),
        "layer": "static_recon",
        "severity": "medium",
        "domain_hint": str(cand.get("domain_hint") or "freeform").strip() or "freeform",
        "source": "static_recon",
        "provenance": predicate_file,
    }


class InternalTasksPhase(CoordinatorCollaborator):
    """Handles internal task execution: report, session breakdown, and internal bookkeeping tasks."""

    def _research_scout_already_proven(self) -> list[dict[str, str]]:
        """Priors this session has already validated, so the scout does not re-propose them."""
        proven = list(self._coord.phase_prelude.warm_recipe_proven_items())
        search = self.shared_state.explore_search or {}
        accepted = search.get("accepted") if isinstance(search, dict) else []
        if not isinstance(accepted, list):
            return proven
        for variant in accepted:
            if not isinstance(variant, dict):
                continue
            name = str(variant.get("name") or variant.get("fingerprint") or "").strip()
            if not name:
                continue
            sources = variant.get("source_evidence") or variant.get("pr_evidence") or []
            source = str(sources[0]).strip() if isinstance(sources, list) and sources else ""
            proven.append({"name": name, "source": source})
        return proven

    def _research_scout_residual_questions(self) -> str:
        """The open questions the latest scout round left behind, as a note block."""
        rounds = self.shared_state.specialist_rounds or []
        if not isinstance(rounds, list):
            return ""
        for row in reversed(rounds):
            if not isinstance(row, dict) or row.get("domain") != "research_scout_specialist":
                continue
            questions = [str(item).strip() for item in (row.get("residual_questions") or []) if str(item).strip()]
            return "\n".join(f"- {question}" for question in questions)
        return ""

    def _research_scout_params(self, *, reason: str, round_id: int) -> dict[str, Any]:
        """The specialist params for one research-scout round."""
        seen = sorted(
            {str(item).strip() for item in (self.shared_state.research_scout_seen_pr_ids or []) if str(item).strip()}
        )
        params: dict[str, Any] = {
            "domain": "research_scout_specialist",
            "source_phase": str(self.shared_state.phase or "PRELUDE").strip().upper(),
            "gap_canonical_id": f"gap.research_scout.round{int(round_id)}",
            "gap_symptom": (
                "Collect proven priors (reference launch scripts, model "
                "config.json architecture features, cross-framework / "
                "NVIDIA research) into prioritised research hints with "
                "sources; do not benchmark or patch."
            ),
            "gap_layer": "research",
            # Depth bounded by wall-clock budget, not turns (max_turns omitted).
            "source": "coordinator_internal",
            "reason": str(reason),
            "seen_pr_ids": seen,
            "mode": "research",
        }
        proven = self._research_scout_already_proven()
        if proven:
            params["already_proven"] = proven
        recipe_sites = [
            s.strip() for s in re.split(r"[,\s]+", os.environ.get("HYPERLOOM_RECIPE_SITES", "")) if s.strip()
        ]
        if recipe_sites:
            params["recipe_sites"] = recipe_sites
        notes = self._research_scout_residual_questions()
        if notes:
            params["notes"] = notes
        return params

    async def _enqueue_internal_research_scout_task(
        self,
        *,
        reason: str,
        round_id: int,
    ) -> "Task":
        """Enqueue a Coordinator-owned read-only research-scout specialist task; idempotency keyed by round."""
        params = self._research_scout_params(reason=reason, round_id=round_id)
        await self._coord.specialist_dispatch.warm_specialist_params(params)
        lanes, ttl = self._coord.gpu_lanes.framework_authoring_lanes_ttl(params, base_ttl_sec=1800)
        task, was_existing = await self.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params=params,
                idempotency_key=f"internal-research-scout-round{int(round_id)}",
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            )
        )
        if not was_existing:
            self.shared_state.research_scout_last_round = int(round_id)
            self.shared_state.save(self.session_dir)
            log.info(
                "research-scout dispatched: task_id=%s round=%d reason=%s",
                task.task_id,
                int(round_id),
                reason,
            )
        return task

    async def maybe_enqueue_prelude_research_scout(self) -> None:
        """Force-dispatch the PRELUDE research scout (not LLM-proposable); writes hints skeleton first."""
        try:
            from hyperloom.inference_optimizer.baseline_comparison import research_hints as _research_hints

            _research_hints.write_hints_skeleton(self.session_dir)
        except OSError:
            log.exception("research-scout: hints skeleton write failed")
        if not bool(self.shared_state.research_scout_enabled):
            return
        await self._enqueue_internal_research_scout_task(
            reason="prelude_initial",
            round_id=0,
        )

    async def maybe_enqueue_explore_research_scout(self) -> None:
        """Re-dispatch the scout every K config-arm rounds (append-only)."""
        state = self.shared_state
        if not bool(state.research_scout_enabled):
            return
        interval = max(1, int(state.research_scout_interval or 3))
        round_id = int((state.explore_search or {}).get("cursor") or 0)
        if round_id <= 0 or (round_id % interval) != 0:
            return
        if int(state.research_scout_last_round) == round_id:
            return
        await self._enqueue_internal_research_scout_task(
            reason="explore_periodic",
            round_id=round_id,
        )

    async def _enqueue_internal_static_recon_task(
        self,
        *,
        reason: str,
        slug: str,
    ) -> "Task":
        """Enqueue the Coordinator-owned read-only static-recon specialist task; ``slug`` scopes the idempotency key so a later pass is not deduplicated against the first."""
        state = self.shared_state
        idempotency_key = f"internal-static-recon-{slug}"
        params: dict[str, Any] = {
            "domain": "static_recon_specialist",
            "source_phase": str(state.phase or "PRELUDE").strip().upper(),
            "gap_canonical_id": f"gap.static_recon.{slug}",
            "gap_symptom": (
                "Grep the framework source for un-bridged capability switches "
                "(predicates that silently disable a faster path for this "
                "model/GPU/precision) and emit bridge candidates; do not "
                "benchmark or patch."
            ),
            "gap_layer": "static_recon",
            "source": "coordinator_internal",
            "reason": str(reason),
            "scope": "domain",
            "mode": "research",
        }
        # Seed the curated checklist + rendered prompt block.
        from ..knowledge import static_recon_checklist as _src_recon

        _entries = _src_recon.entries_for(
            gpu_type=str(state.gpu_type or ""),
            precision=_src_recon.workload_precision(state),
        )
        _entries = _src_recon.filter_entries_for_model(_entries, dict(state.model_info or {}))
        _rendered = _src_recon.render_checklist_for_prompt(_entries)
        if _rendered:
            params["static_recon_checklist"] = _rendered
        await self._coord.specialist_dispatch.warm_specialist_params(params)
        lanes, ttl = self._coord.gpu_lanes.framework_authoring_lanes_ttl(params, base_ttl_sec=1800)
        task, was_existing = await self.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params=params,
                idempotency_key=idempotency_key,
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            )
        )
        if not was_existing:
            log.info(
                "static-recon dispatched: task_id=%s reason=%s",
                task.task_id,
                reason,
            )
        return task

    async def maybe_enqueue_prelude_static_recon(self) -> None:
        """Force-dispatch the PRELUDE static-recon specialist (not LLM-proposable)."""
        if not bool(self.shared_state.static_recon_enabled):
            return
        await self._enqueue_internal_static_recon_task(
            reason="prelude_initial",
            slug="prelude",
        )

    async def maybe_enqueue_config_plateau_static_recon(self) -> None:
        """Re-run static recon once per macro-cycle when the config arm goes dry.

        The PRELUDE pass greps the source before any measurement exists. Once
        the grids stop paying, the profile says which subsystem to look at, so a
        second pass reads the predicates that matter rather than all of them.
        """
        state = self.shared_state
        if not bool(state.static_recon_enabled):
            return
        if str(state.phase or "").strip().upper() != _phase_state.PHASE_FRAMEWORK_AGENT:
            return
        _, evidence = _phase_state.per_lever_dryness(state)
        if not bool(evidence.get("config_arm_plateaued")):
            return
        await self._enqueue_internal_static_recon_task(
            reason="config_arm_plateau",
            slug=f"cycle{int(state.macro_cycle or 0)}",
        )

    async def maybe_enqueue_trajectory_reviewer(self) -> None:
        """On a plateau, dispatch a Coordinator-owned readonly specialist seeded with the deterministic trajectory digest to propose fresh directions."""
        if not env_bool("INFERENCE_OPTIMIZER_TRAJECTORY_LLM_REVIEW", default=True):
            return
        state = self.shared_state
        plateau_active = bool(self._coord.conversation.plateau_advisory_block())
        if not plateau_active:
            return
        cycle = int(state.macro_cycle or 0)
        digest = self._coord.conversation.trajectory_digest()
        direction, _pct = self._coord.conversation.dominant_roofline_direction()
        from hyperloom.inference_optimizer.roofline_snapshot import BOTTLENECK_DOMAIN_HINTS

        hint = BOTTLENECK_DOMAIN_HINTS.get(direction)
        domain = hint[0] if hint else "serving_specialist"
        params: dict[str, Any] = {
            "domain": domain,
            "source_phase": str(state.phase or "INTERNAL").strip().upper(),
            "gap_canonical_id": f"gap.trajectory_review.cycle{cycle}",
            "gap_symptom": (
                "The search has plateaued. Review the optimization trajectory "
                "below and propose fresh, non-redundant directions (avoid the "
                "exhausted ones).\n" + (digest or "(no digest)")
            ),
            "gap_layer": "research",
            # Depth bounded by wall-clock budget, not turns.
            "source": "coordinator_internal",
            "reason": "plateau_trajectory_review",
            "mode": "research",
        }
        if digest:
            params["gap_evidence"] = {"trajectory_review": digest}
        await self._coord.specialist_dispatch.warm_specialist_params(params)
        lanes, ttl = self._coord.gpu_lanes.framework_authoring_lanes_ttl(params, base_ttl_sec=1800)
        task, was_existing = await self.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params=params,
                idempotency_key=f"internal-trajectory-review-cycle{cycle}",
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            )
        )
        if not was_existing:
            log.info(
                "trajectory-review dispatched: task_id=%s cycle=%d domain=%s",
                task.task_id,
                cycle,
                domain,
            )

    def consume_static_recon(self, done_payload: dict[str, Any]) -> None:
        """Seed static-recon bridge candidates into gaps[] (idempotent)."""
        block = done_payload.get("recon")
        if not isinstance(block, dict):
            return
        candidates = block.get("bridge_candidates")
        if not isinstance(candidates, list):
            return
        seeded = 0
        for idx, cand in enumerate(candidates):
            gap = _bridge_candidate_gap(cand, idx) if isinstance(cand, dict) else None
            if gap is None:
                continue
            self.shared_state.upsert_gap(gap)
            seeded += 1
        if seeded:
            try:
                self.shared_state.save(self.session_dir)
            except OSError:
                log.exception("static-recon: SharedState.save after seeding failed")
        log.info("static-recon consumed: bridge_candidates_seeded=%d", seeded)
