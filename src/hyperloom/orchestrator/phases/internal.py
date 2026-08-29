# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase-independent internal task handler: research-scout, static-recon, and
trajectory-reviewer auto-enqueue helpers used across multiple phases."""

from __future__ import annotations
import logging as _logging
import os
import re
from typing import Any
from ..state.task_registry import Task
from .base import PhaseHandler

log = _logging.getLogger(__name__)


class InternalTasksPhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

    async def _enqueue_internal_research_scout_task(
        self,
        *,
        reason: str,
        round_id: int,
    ) -> "Task | None":
        """Enqueue a Coordinator-owned read-only research-scout specialist task; idempotency keyed by round, returns None on existing/failure (fail-soft).

        Args:
            reason: Tag distinguishing the enqueue site, recorded on the task.
            round_id: The config-arm round (or 0 for PRELUDE) scoping the
                idempotency key.

        Returns:
            The created (or existing) specialist :class:`Task`, or ``None`` when
            the scout is disabled or enqueue fails.
        """
        if not bool(getattr(self.shared_state, "research_scout_enabled", True)):
            return None
        idempotency_key = f"internal-research-scout-round{int(round_id)}"
        seen = sorted(
            {
                str(item).strip()
                for item in (getattr(self.shared_state, "research_scout_seen_pr_ids", []) or [])
                if str(item).strip()
            }
        )
        params: dict[str, Any] = {
            "domain": "research_scout_specialist",
            "source_phase": str(getattr(self.shared_state, "phase", "") or "PRELUDE").strip().upper(),
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
        proven = list(self._warm_recipe_proven_items())
        search = getattr(self.shared_state, "explore_search", None) or {}
        accepted = search.get("accepted") if isinstance(search, dict) else []
        if isinstance(accepted, list):
            for variant in accepted:
                if not isinstance(variant, dict):
                    continue
                name = str(variant.get("name") or variant.get("fingerprint") or "").strip()
                if name:
                    sources = variant.get("source_evidence") or variant.get("pr_evidence") or []
                    source = str(sources[0]).strip() if isinstance(sources, list) and sources else ""
                    proven.append({"name": name, "source": source})
        if proven:
            params["already_proven"] = proven
        recipe_sites = [
            s.strip() for s in re.split(r"[,\s]+", os.environ.get("HYPERLOOM_RECIPE_SITES", "")) if s.strip()
        ]
        if recipe_sites:
            params["recipe_sites"] = recipe_sites
        rounds = getattr(self.shared_state, "specialist_rounds", None) or []
        if isinstance(rounds, list):
            for row in reversed(rounds):
                if not isinstance(row, dict) or row.get("domain") != "research_scout_specialist":
                    continue
                questions = [str(item).strip() for item in (row.get("residual_questions") or []) if str(item).strip()]
                if questions:
                    params["notes"] = "\n".join(f"- {question}" for question in questions)
                break
        await self._warm_specialist_params(params)
        try:
            task, was_existing = await self.tasks.create_or_return_existing(
                kind="specialist",
                params=params,
                idempotency_key=idempotency_key,
                requires_lanes=["research_lane"],
                side_effects=["writes_results"],
                lease_ttl_sec=1800,
            )
        except Exception:  # noqa: BLE001 — TaskRegistry edge cases
            log.exception(
                "research-scout: enqueue failed (round=%d)",
                int(round_id),
            )
            return None
        if not was_existing:
            self.shared_state.bump_research_scout_runs()
            self.shared_state.research_scout_last_round = int(round_id)
            self.shared_state.save(self.session_dir)
            log.info(
                "research-scout dispatched: task_id=%s round=%d reason=%s runs=%d",
                task.task_id,
                int(round_id),
                reason,
                self.shared_state.research_scout_runs,
            )
        return task

    async def _maybe_enqueue_prelude_research_scout(self) -> None:
        """Force-dispatch the PRELUDE research scout (not LLM-proposable); writes hints skeleton first."""
        try:
            from ..knowledge import research_hints as _research_hints

            _research_hints.write_hints_skeleton(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: hints skeleton write failed")
        if not bool(getattr(self.shared_state, "research_scout_enabled", True)):
            return
        try:
            await self._enqueue_internal_research_scout_task(
                reason="prelude_initial",
                round_id=0,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: PRELUDE dispatch failed")

    async def _maybe_enqueue_explore_research_scout(self) -> None:
        """Re-dispatch the scout every K config-arm rounds (append-only)."""
        state = self.shared_state
        if not bool(getattr(state, "research_scout_enabled", True)):
            return
        interval = max(1, int(getattr(state, "research_scout_interval", 3) or 3))
        round_id = int((state.explore_search or {}).get("cursor") or 0)
        if round_id <= 0 or (round_id % interval) != 0:
            return
        if int(getattr(state, "research_scout_last_round", -1)) == round_id:
            return
        try:
            await self._enqueue_internal_research_scout_task(
                reason="explore_periodic",
                round_id=round_id,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: re-dispatch failed")

    async def _enqueue_internal_static_recon_task(
        self,
        *,
        reason: str,
    ) -> "Task | None":
        """Enqueue the Coordinator-owned read-only static-recon specialist task.

        Seeds a source-code reconnaissance mandate: the specialist greps the
        framework source for un-bridged capability switches and produces bridge
        candidates only (read-only). Idempotency keyed to the session (PRELUDE
        one-shot). Returns None when disabled or enqueue fails.

        Args:
            reason: Tag distinguishing the enqueue site, recorded on the task.

        Returns:
            The created (or existing) specialist :class:`Task`, or ``None`` when
            static-recon is disabled or enqueue fails.
        """
        state = self.shared_state
        if not bool(getattr(state, "static_recon_enabled", True)):
            return None
        idempotency_key = "internal-static-recon-prelude"
        params: dict[str, Any] = {
            "domain": "static_recon_specialist",
            "source_phase": str(getattr(state, "phase", "") or "PRELUDE").strip().upper(),
            "gap_canonical_id": "gap.static_recon.prelude",
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
            "lane": "cpu",
        }
        # Seed the curated checklist + rendered prompt block; best-effort.
        try:
            from ..knowledge import static_recon_checklist as _src_recon

            _entries = _src_recon.entries_for(
                model_class=str(getattr(state, "model_class", "") or ""),
                gpu_type=str(getattr(state, "gpu_type", "") or ""),
                precision=str(getattr(state, "precision", "") or ""),
            )
            _entries = _src_recon.filter_entries_for_model(_entries, dict(getattr(state, "model_info", None) or {}))
            _rendered = _src_recon.render_checklist_for_prompt(_entries)
            if _rendered:
                params["static_recon_checklist"] = _rendered
            _dicts = _src_recon.checklist_as_dicts(_entries)
            if _dicts:
                params["static_recon_checklist_entries"] = _dicts
        except Exception:  # noqa: BLE001 — advisory; never block dispatch
            log.exception("static-recon: checklist seeding failed")
        await self._warm_specialist_params(params)
        try:
            task, was_existing = await self.tasks.create_or_return_existing(
                kind="specialist",
                params=params,
                idempotency_key=idempotency_key,
                requires_lanes=["research_lane"],
                side_effects=["writes_results"],
                lease_ttl_sec=1800,
            )
        except Exception:  # noqa: BLE001 — TaskRegistry edge cases
            log.exception("static-recon: enqueue failed")
            return None
        if not was_existing:
            try:
                state.static_recon_runs = int(getattr(state, "static_recon_runs", 0) or 0) + 1
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — defensive bookkeeping
                log.exception("static-recon: bookkeeping save failed")
            log.info(
                "static-recon dispatched: task_id=%s reason=%s",
                task.task_id,
                reason,
            )
        return task

    async def _maybe_enqueue_prelude_static_recon(self) -> None:
        """Force-dispatch the PRELUDE static-recon specialist (not LLM-proposable)."""
        if not bool(getattr(self.shared_state, "static_recon_enabled", True)):
            return
        try:
            await self._enqueue_internal_static_recon_task(
                reason="prelude_initial",
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("static-recon: PRELUDE dispatch failed")

    async def _maybe_enqueue_trajectory_reviewer(self) -> None:
        """On a plateau, dispatch a Coordinator-owned readonly specialist seeded
        with the deterministic trajectory digest to propose fresh directions.

        Gated by ``INFERENCE_OPTIMIZER_TRAJECTORY_LLM_REVIEW`` (default on; set
        to a falsy value to opt out); idempotent per macro-cycle. The specialist
        targets the dominant bottleneck's domain so its proposals flow through
        the standard specialist → explore pipeline. Fail-soft.
        """
        if os.getenv(
            "INFERENCE_OPTIMIZER_TRAJECTORY_LLM_REVIEW",
            "1",
        ).strip().lower() not in ("1", "true", "on", "yes"):
            return
        state = self.shared_state
        try:
            plateau_active = bool(self._plateau_advisory_block())
        except Exception:  # noqa: BLE001 — defensive
            plateau_active = False
        if not plateau_active:
            return
        cycle = int(getattr(state, "macro_cycle", 0) or 0)
        try:
            from ..knowledge import trajectory_reviewer as _trajectory_reviewer

            digest = _trajectory_reviewer.build_trajectory_digest(
                self.session_dir,
                state,
            )
        except Exception:  # noqa: BLE001 — defensive
            digest = ""
        direction, _pct = self._dominant_roofline_direction()
        from ..kernel.roofline_snapshot import BOTTLENECK_DOMAIN_HINTS

        hint = BOTTLENECK_DOMAIN_HINTS.get(direction)
        domain = hint[0] if hint else "serving_specialist"
        params: dict[str, Any] = {
            "domain": domain,
            "source_phase": str(getattr(state, "phase", "") or "INTERNAL").strip().upper(),
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
        await self._warm_specialist_params(params)
        try:
            task, was_existing = await self.tasks.create_or_return_existing(
                kind="specialist",
                params=params,
                idempotency_key=f"internal-trajectory-review-cycle{cycle}",
                requires_lanes=["research_lane"],
                side_effects=["writes_results"],
                lease_ttl_sec=1800,
            )
        except Exception:  # noqa: BLE001 — TaskRegistry edge cases
            log.exception("trajectory-review: enqueue failed (cycle=%d)", cycle)
            return
        if not was_existing:
            log.info(
                "trajectory-review dispatched: task_id=%s cycle=%d domain=%s",
                task.task_id,
                cycle,
                domain,
            )

    def _consume_static_recon(self, done_payload: dict[str, Any]) -> None:
        """Seed static-recon bridge candidates into gaps[] (idempotent, fail-soft).

        Reads the specialist's ``recon`` block, validates each
        ``bridge_candidate``, and upserts one gap per candidate so the config-arm
        freeform specialist later dispatches against it with a precise mandate
        (predicate location + consequence + bridge sketch). Read-only producer:
        no patch is applied here; the normal KEEP gate still governs landing.

        Args:
            done_payload: The completed static-recon task payload; its ``recon``
                block carries ``bridge_candidates``.
        """
        block = done_payload.get("recon")
        if not isinstance(block, dict):
            return
        candidates = block.get("bridge_candidates")
        if not isinstance(candidates, list):
            return
        seeded = 0
        for idx, cand in enumerate(candidates):
            if not isinstance(cand, dict):
                continue
            predicate_file = str(cand.get("predicate_file") or "").strip()
            why = str(cand.get("why_disabled_here") or "").strip()
            # A candidate without a source anchor + explanation is not actionable.
            if not predicate_file or not why:
                continue
            raw_id = str(cand.get("id") or f"cand{idx}").strip() or f"cand{idx}"
            slug = "".join(c if (c.isalnum() or c in "._-") else "_" for c in raw_id)[:80]
            cid = f"gap.static_recon.{slug}"
            predicate_name = str(cand.get("predicate_name") or "").strip()
            consequence = str(cand.get("consequence") or "").strip()
            bridge_sketch = str(cand.get("bridge_sketch") or "").strip()
            domain_hint = str(cand.get("domain_hint") or "freeform").strip() or "freeform"
            symptom_parts = [
                f"Un-bridged switch in {predicate_file}"
                + (f" ({predicate_name})" if predicate_name else "")
                + f": {why}",
            ]
            if consequence:
                symptom_parts.append(f"Consequence: {consequence}")
            if bridge_sketch:
                symptom_parts.append(f"Bridge: {bridge_sketch}")
            symptom = " ".join(symptom_parts)[:1200]
            try:
                self.shared_state.upsert_gap(
                    {
                        "canonical_id": cid,
                        "symptom": symptom,
                        "layer": "static_recon",
                        "severity": "medium",
                        "domain_hint": domain_hint,
                        "source": "static_recon",
                        "provenance": predicate_file,
                    }
                )
                seeded += 1
            except Exception:  # noqa: BLE001 — defensive
                log.exception("static-recon: upsert_gap failed for %s", cid)
        if seeded:
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — defensive
                log.exception("static-recon: SharedState.save after seeding failed")
        log.info("static-recon consumed: bridge_candidates_seeded=%d", seeded)
