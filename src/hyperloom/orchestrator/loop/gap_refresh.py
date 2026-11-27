# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gap refresh: the Coordinator collaborator that files research hints and explore outcomes on ``SharedState.gaps``."""

from __future__ import annotations

import logging as _logging
from hashlib import sha1
from typing import TYPE_CHECKING, Any

from ..collaborator import CoordinatorCollaborator
from ..state.failure_evidence import UNMEASURED_OUTCOMES, failure_from_variant_outcome
from ..state.failure_packets import write_failure

if TYPE_CHECKING:
    from ..state.task_registry import Task

log = _logging.getLogger(__name__)

__all__ = ["GapRefreshCollaborator"]


# Artifact references copied from a per-variant outcome onto its gap attempt.
_GAP_ATTEMPT_ARTIFACT_KEYS: tuple[str, ...] = (
    "failure_id",
    "fingerprint",
    "stage",
    "workspace",
    "server_log_path",
)


class GapRefreshCollaborator(CoordinatorCollaborator):
    """Seeds a gap per research hint and records each explore round's per-variant outcomes."""

    def seed_gaps_from_research_hints(self) -> None:
        """Inject research hints as advisory gaps[] seeds (idempotent)."""
        from hyperloom.inference_optimizer.baseline_comparison import research_hints as _research_hints

        hints = _research_hints.load_hints(self.session_dir)
        for hint in hints:
            what = str(hint.get("what") or "").strip()
            source = str(hint.get("source") or "").strip()
            if not what or not source:
                continue
            tags = hint.get("domain_tags") or []
            key = f"{what.lower()}::{source.lower()}"
            cid = f"gap.research_hint.{sha1(key.encode()).hexdigest()[:16]}"
            self.shared_state.upsert_gap(
                {
                    "canonical_id": cid,
                    "symptom": what,
                    "layer": "research_hint",
                    "severity": "medium",
                    "domain_hint": str(tags[0]) if tags else "",
                    "source": "research_scout",
                    "provenance": str(hint.get("source") or ""),
                }
            )

    def _framework_authoring_domain(self) -> str:
        """Return the authoring domain matching this session's framework kind.

        Returns:
            str: ``"framework_rewrite_specialist"`` for a scriptable framework,
            else ``"serving_specialist"``.
        """
        from ..specialists.domains import authoring_domain_for_framework

        return authoring_domain_for_framework(getattr(self.shared_state, "framework", ""))

    def record_explore_round_gaps(
        self,
        *,
        task: "Task | None",
        result: dict[str, Any],
        workload_id: str,
    ) -> None:
        """Append per-variant KEEP/REVERT outcomes to the matching gap (or the anchor gap as fallback).

        Args:
            task: The explore task whose params carry the gap canonical id;
                ``None`` is a no-op.
            result: The explore result; its ``per_variant_outcomes`` drive the
                appended gap attempts.
            workload_id: Canonical id of the anchor gap used when the task names no gap.
        """
        if task is None:
            return
        per_variant = result.get("per_variant_outcomes")
        if not isinstance(per_variant, list) or not per_variant:
            return
        params = dict(task.params or {})
        canonical = str(params.get("gap_canonical_id") or "").strip() or workload_id
        state = self.shared_state
        existing = state.find_gap(canonical)
        if existing is None:
            state.upsert_gap(
                {
                    "canonical_id": canonical,
                    "symptom": "explore round outcomes",
                    "layer": "framework",
                    "severity": "medium",
                    "domain_hint": self._framework_authoring_domain(),
                    "source": "attempts",
                }
            )
        for outcome in per_variant:
            if not isinstance(outcome, dict):
                continue
            attempt: dict[str, Any] = {
                "action": "explore",
                "variant_name": str(outcome.get("variant_name") or ""),
                "outcome": str(outcome.get("outcome") or "").upper(),
                "gain_pct": outcome.get("gain_pct"),
                "reason": str(outcome.get("reason") or ""),
                "error_class": str(outcome.get("error_class") or ""),
            }
            for key in _GAP_ATTEMPT_ARTIFACT_KEYS:
                value = outcome.get(key)
                if value:
                    attempt[key] = str(value)
            state.append_gap_attempt(canonical, attempt)

    def record_explore_variant_failures(
        self,
        *,
        task: "Task | None",
        result: dict[str, Any],
    ) -> None:
        """Record each unmeasured ``per_variant_outcomes`` row as failure evidence + ``last_action_failures``.

        A crashed variant does not fail the round, so the round-level recorder
        never sees it.

        Args:
            task: The completed explore task; ``None`` is a no-op.
            result: The explore result dict carrying ``per_variant_outcomes``.
        """
        if task is None:
            return
        per_variant = result.get("per_variant_outcomes")
        if not isinstance(per_variant, list):
            return
        task_id = str(task.task_id or "")
        round_id = str(result.get("round_id") or "")
        for vo in per_variant:
            if not isinstance(vo, dict):
                continue
            if str(vo.get("outcome") or "").upper() not in UNMEASURED_OUTCOMES:
                continue
            fe = failure_from_variant_outcome(task_id=task_id, round_id=round_id, vo=vo)
            write_failure(self.session_dir, fe)
            self.shared_state.record_action_failure(
                action="explore",
                task_id=task_id,
                result={
                    "variant_name": str(vo.get("variant_name") or ""),
                    "error_class": str(vo.get("error_class") or ""),
                    "error": str(vo.get("reason") or ""),
                    "workspace": vo.get("workspace"),
                    "stderr_log_path": vo.get("server_log_path"),
                    "failure_id": fe.get("failure_id"),
                },
            )
