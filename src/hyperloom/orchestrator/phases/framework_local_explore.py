# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK_AGENT local-explore specialists: their gap, rewrite evidence, working memory and claim."""

from __future__ import annotations

import logging as _logging
from pathlib import Path
from collections.abc import Mapping
from typing import Any
from hyperloom_common.coerce import first_str, to_list, to_mapping
from ..state.task_registry import TaskSpec
from hyperloom.common.framework_arm import LOCAL_EXPLORE_CANDIDATE_PREFIX
from .framework_records import _recorder

log = _logging.getLogger(__name__)


# Specialist attempts a local-exploration candidate gets before the phase moves on.
_LOCAL_EXPLORE_MAX_ATTEMPTS: int = 3


def _settled_local_explore_rounds(state: Any) -> int:
    """How many local-exploration rounds the phase progress already records."""
    return sum(
        1
        for p in to_list(state.framework_agent_phase_progress)
        if first_str(to_mapping(p).get("candidate_id")).startswith(LOCAL_EXPLORE_CANDIDATE_PREFIX)
    )


def _local_explore_gap(gap_cid: str, gap: str, domain: str) -> dict[str, Any]:
    """The gap row a local-exploration round is filed under."""
    return {
        "canonical_id": gap_cid,
        "symptom": gap or "Author a throughput patch from live source + profiling evidence",
        "layer": "framework",
        "severity": "medium",
        "domain_hint": domain,
        "source": "coordinator_internal",
    }


class _LocalExploreMixin:
    """Dispatches the local-explore specialist that searches the framework for the current gap."""

    # Max tried-candidate rows fed into the ranker/discovery working memory.
    _FRAMEWORK_TRIED_MEMORY_CAP: int = 12

    def _render_rewrite_evidence_for_prompt(self) -> str:
        """Render the measured host-side rewrite evidence as prompt lines."""
        path = str(self.shared_state.last_framework_rewrite_evidence or "").strip()
        if not path:
            return ""
        try:
            import json as _json

            from ..actions.executors.framework_change._framework_rewrite_evidence import summarize_for_prompt

            document = _json.loads(Path(path).read_text(encoding="utf-8"))
            return summarize_for_prompt(document)
        except (OSError, ValueError):
            log.warning("FRAMEWORK: rewrite-evidence render failed path=%s", path, exc_info=True)
            return ""

    def _rewrite_evidence_absence_note(self) -> str:
        """Explain an empty evidence block instead of implying there is nothing to find."""
        read_the_source = (
            "Locate the candidates by reading the source: find the denoising / "
            "rollout loop and ask, for each call inside it, whether the result "
            "can change across iterations."
        )
        status = str(self.shared_state.last_framework_rewrite_evidence_status or "").strip()
        if status == "no_candidates":
            return (
                "The host-side probe ran and found no rewrite candidates. Treat "
                "that as a measured negative for the patterns it covers "
                "(host round-trips, host syncs, device residency, collective "
                "fusion, memoization, loop hoisting) and look elsewhere. " + read_the_source
            )
        if status and status != "ok":
            return (
                "No host-side rewrite evidence is available because the probe did "
                f"not deliver any: {status}. This is a broken instrument, NOT a "
                "measured negative -- do not conclude the loop is clean. " + read_the_source
            )
        if str(self.shared_state.last_framework_rewrite_evidence or "").strip():
            # Reached only when a document is on record but rendering it produced nothing, so the evidence exists and
            # this prompt cannot show it.
            return (
                "Host-side rewrite evidence was collected but could not be "
                "rendered here, so its absence below means nothing. " + read_the_source
            )
        return "No host-side rewrite evidence has been collected yet for this session. " + read_the_source

    def _compose_framework_local_explore_gap(self) -> tuple[str, list[str]]:
        """Compose the ``(gap, keywords)`` steering the local-exploration arm."""
        state = self.shared_state
        from ..actions.executors.framework_change._framework_gap_composer import compose_gap

        return compose_gap(
            framework=str(state.framework or ""),
            gpu_type=str(state.gpu_type or ""),
            model_class=str(state.model_class or ""),
            precision=str(state.precision or ""),
            rewrite_evidence_path=state.last_framework_rewrite_evidence or None,
        )

    def _record_local_explore_proposal(
        self, cand_id: str, candidate: Mapping[str, Any], params: Mapping[str, Any], *, gap_cid: str, reason: str
    ) -> None:
        """Record a local-exploration round as a proposal the orchestration itself produced."""
        recorder = _recorder(self)
        if recorder is None:
            return
        from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
            ARM_SOURCE,
            PRODUCER_ORCHESTRATION,
            STEP_PROPOSED,
        )

        recorder.record_proposal(
            cand_id,
            arm=ARM_SOURCE,
            producer=PRODUCER_ORCHESTRATION,
            title=first_str(candidate.get("title")) or cand_id,
            reasoning=str(params["gap_symptom"]),
            gap_canonical_id=gap_cid,
            route="author_via_specialist",
        )
        recorder.record_proposal_step(cand_id, step=STEP_PROPOSED, reason=reason)

    async def _claim_local_explore_specialist(self, params: dict[str, Any], cand_id: str) -> tuple[Any, bool] | None:
        """Claim the specialist row for a local-exploration round, keying past rows an earlier attempt left failed."""
        # The registry de-duplicates by key and hands back whatever row it finds, so a candidate whose specialist
        # failed keeps resolving to that failure: the phase re-selects the candidate every tick, logs a dispatch, and
        # nothing runs.
        lanes, ttl = self._coord.gpu_lanes.framework_authoring_lanes_ttl(params, base_ttl_sec=3600)
        base_idem = f"framework_agent_local_explore:{cand_id}{self._coord.dispatcher.cycle_idem_suffix()}"
        for attempt in range(_LOCAL_EXPLORE_MAX_ATTEMPTS):
            idem = base_idem if attempt == 0 else f"{base_idem}:r{attempt}"
            spec_task, existing = await self.tasks.create_or_return_existing(
                TaskSpec(
                    kind="specialist",
                    params=params,
                    idempotency_key=idem,
                    requires_lanes=lanes,
                    lease_ttl_sec=ttl,
                    dispatch_class="coordinator",
                )
            )
            if not (existing and first_str(getattr(spec_task, "state", "")) == "failed"):
                return spec_task, existing
            log.info(
                "FRAMEWORK local-explore: %s already failed under %s; retrying candidate %s",
                getattr(spec_task, "task_id", "?"),
                idem,
                cand_id,
            )
        log.warning(
            "FRAMEWORK local-explore: candidate %s exhausted %d specialist "
            "attempt(s); leaving it to the phase to select another",
            cand_id,
            _LOCAL_EXPLORE_MAX_ATTEMPTS,
        )
        return None

    async def _enqueue_framework_agent_local_explore_specialist(
        self,
        candidate: dict[str, Any],
        *,
        reason: str = "",
    ) -> str:
        """Dispatch a candidate-free authoring specialist (no upstream PR lead)."""
        state = self.shared_state
        # A local-exploration round has no upstream lead to key on, so its id counts the rounds already settled.
        settled = _settled_local_explore_rounds(state)
        cand_id = self._framework_candidate_key(candidate) or f"{LOCAL_EXPLORE_CANDIDATE_PREFIX}{settled}"
        gap = first_str(candidate.get("gap_description")).strip()
        gap_cid = first_str(candidate.get("gap_canonical_id")).strip() or f"gap.framework.local_explore.{cand_id}"
        framework = first_str(candidate.get("framework"), state.framework).strip().lower()
        domain = self._authoring_specialist_domain()
        notes = ""
        if domain == "framework_rewrite_specialist":
            notes = self._render_rewrite_evidence_for_prompt() or self._rewrite_evidence_absence_note()
        state.upsert_gap(_local_explore_gap(gap_cid, gap, domain))
        memory = self._build_framework_working_memory()
        prior_attempts = [
            t for t in to_list(memory.get("tried_and_why")) if first_str(to_mapping(t).get("ref")).strip()
        ]
        params: dict[str, Any] = {
            "domain": domain,
            "source_phase": "FRAMEWORK_AGENT",
            "gap_canonical_id": gap_cid,
            # No ``lever_kind``: this arm names a gap, not a lever, and its specialist returns either.
            "gap_symptom": (gap or "Author a framework source patch from live source + profile evidence"),
            "gap_layer": "framework",
            "framework": framework,
            "task_kind": "framework_local_explore",
            "prior_attempts": prior_attempts,
            "notes": notes,
            # Same provenance markers as the PR-authoring track so the autosubmit -> integrate_patch ->
            # authored-outcome bridge applies.
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": cand_id,
            "framework_batch_id": "",
            "framework_audit": {},
            "framework_local_explore": True,
            "source": "coordinator_internal",
        }
        self._record_local_explore_proposal(cand_id, candidate, params, gap_cid=gap_cid, reason=reason)
        await self._coord.specialist_dispatch.warm_specialist_params(params)
        claimed = await self._claim_local_explore_specialist(params, cand_id)
        if claimed is None:
            return ""
        spec_task, _spec_existing = claimed
        if _spec_existing and await self._settle_finished_authoring_specialist(
            spec_task, cand_id=cand_id, batch_id=str(candidate.get("batch_id") or ""), label="local-explore"
        ):
            return ""
        spec_tid = self._map_authoring_specialist(spec_task, cand_id=cand_id, label="local-explore")
        log.info(
            "FRAMEWORK: dispatched local-exploration specialist candidate=%s gap=%s reason=%s",
            cand_id,
            gap_cid,
            reason or "resident",
        )
        return spec_tid

    def _build_framework_working_memory(self) -> dict[str, Any]:
        """Summarise the most recent tried candidates from the progress ledger (deterministic, zero-LLM)."""
        state = self.shared_state
        progress = state.framework_agent_phase_progress or []
        rows = [p for p in progress if isinstance(p, dict) and self._framework_candidate_key(p)]
        tried: list[dict[str, Any]] = []
        for row in rows[-self._FRAMEWORK_TRIED_MEMORY_CAP :]:
            status = str(row.get("status") or "").strip()
            gain = row.get("gain_pct")
            why = str(row.get("rationale") or row.get("reason") or "").strip()
            tried.append(
                {
                    "ref": self._framework_candidate_key(row),
                    "status": status,
                    "gain_pct": (float(gain) if isinstance(gain, (int, float)) else None),
                    "why": why[:200],
                }
            )
        return {
            "tried_and_why": tried,
        }
