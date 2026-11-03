# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gap refresh: the Coordinator collaborator that derives ``SharedState.gaps`` from baselines, attempts and hints."""

from __future__ import annotations

import logging as _logging
from datetime import datetime, timezone
from hashlib import sha1
from typing import TYPE_CHECKING, Any

from hyperloom_common.coerce import first_text

from ..collaborator import CoordinatorCollaborator
from ..state.failure_evidence import UNMEASURED_OUTCOMES, failure_from_variant_outcome

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
    """Gap-signal extraction from baselines, attempt history, and research hints."""

    async def refresh_gaps(self, *, reason: str, workload_id: str) -> None:
        """Refresh :attr:`SharedState.gaps` from observable signals. Additive upsert deduped by canonical_id.

        Args:
            reason: Tag describing the refresh trigger, used only in logging.
            workload_id: The workload's canonical Recipe id that the derived gap rows are keyed on.
        """
        state = self.shared_state
        for entry in self._extract_gaps_from_baseline(workload_id):
            state.upsert_gap(entry)
        for entry in self._extract_gaps_from_attempts(workload_id):
            state.upsert_gap(entry)
        log.debug(
            "gaps refresh (reason=%s): %d gaps after merge",
            reason,
            len(state.gaps),
        )

    def _extract_gaps_from_baseline(self, workload_id: str) -> list[dict[str, Any]]:
        """Derive initial gap rows from the baseline snapshot (throughput_below_target, baseline_unstable); key the rows on ``workload_id`` (matching ``recipe_kb_t0.run_t0_anchor``) so traverse rows align.

        Returns:
            A list of gap row dicts derived from the baseline; empty when no
            baseline throughput is recorded.
        """
        state = self.shared_state
        gaps: list[dict[str, Any]] = []
        if state.baseline_tput <= 0:
            return gaps
        target_gap = self._coord.target_gap_pct()
        if target_gap > 0.0:
            severity = "high" if target_gap >= 10.0 else "medium" if target_gap >= 3.0 else "low"
            gaps.append(
                {
                    "canonical_id": f"{workload_id}#throughput_below_target",
                    "symptom": (f"current_best is {target_gap:.1f}% short of the run objective target"),
                    "layer": "framework",
                    "severity": severity,
                    "domain_hint": self._framework_authoring_domain(),
                    "source": "baseline",
                }
            )
        if state.baseline_failure_streak > 0:
            gaps.append(
                {
                    "canonical_id": f"{workload_id}#baseline_unstable",
                    "symptom": (f"baseline crashed {state.baseline_failure_streak} consecutive time(s)"),
                    "layer": "system",
                    "severity": ("high" if state.baseline_failure_streak >= 2 else "medium"),
                    "domain_hint": "system_specialist",
                    "source": "baseline",
                }
            )
        return gaps

    def _extract_gaps_from_attempts(self, workload_id: str) -> list[dict[str, Any]]:
        """Derive gaps from rolling failures + winners history (recurring (action, error_class[, variant]) + explore plateau).

        Returns:
            A list of gap row dicts derived from recurring action failures and
            an explore-plateau signal.
        """
        state = self.shared_state
        gaps: list[dict[str, Any]] = []

        # Already capped by ``record_action_failure``; read the whole log.
        seen_failures: dict[str, dict[str, Any]] = {}
        for row in state.last_action_failures or []:
            if not isinstance(row, dict):
                continue
            action = first_text(row.get("action")) or "unknown"
            err = first_text(row.get("error_class")) or "unknown_error"
            variant = first_text(row.get("variant_name"))
            layer, domain = self._gap_layer_for_action(action, str(getattr(self.shared_state, "framework", "") or ""))
            excerpt = str(row.get("error_excerpt") or "")
            detail = next((ln.strip() for ln in excerpt.splitlines() if ln.strip()), err)[:200]
            # Variant discriminator keeps distinct crash causes in distinct gaps.
            if variant:
                key = f"{action}::{err}::{variant}"
                symptom = f"{action}/{variant} fails: {detail}"
                cid_variant = f":{variant}"
            else:
                key = f"{action}::{err}"
                symptom = f"{action} repeatedly fails with {detail}"
                cid_variant = ""
            attempt = {
                "action": action,
                "variant_name": variant,
                "outcome": "REVERT",
                "error_class": err,
                "ts": str(row.get("ts") or datetime.now(timezone.utc).isoformat()),
            }
            if key in seen_failures:
                seen_failures[key]["attempts"].append(attempt)
            else:
                seen_failures[key] = {
                    "canonical_id": f"{workload_id}#fail:{action}:{err}{cid_variant}",
                    "symptom": symptom,
                    "layer": layer,
                    "severity": "medium",
                    "domain_hint": domain,
                    "source": "attempts",
                    "attempts": [attempt],
                }
        gaps.extend(seen_failures.values())
        gaps.extend(self._explore_plateau_gaps(workload_id))
        return gaps

    def _explore_plateau_gaps(self, workload_id: str) -> list[dict[str, Any]]:
        """The explore-plateau gap when grid rounds stopped promoting, else none."""
        state = self.shared_state
        no_promote = int(state.params_no_promote_streak or 0)
        # Every writer of explore_search stores a dict, and a loaded state rebuilds it as one.
        winners_hist = list(state.explore_search.get("winners_history") or [])
        recent_promotions = sum(
            1 for w in winners_hist[-5:] if isinstance(w, dict) and float(w.get("gain_pct") or 0.0) > 0.0
        )
        if no_promote < 3 or recent_promotions != 0:
            return []
        return [
            {
                "canonical_id": f"{workload_id}#explore_plateau",
                "symptom": (f"{no_promote} consecutive grid rounds without a new current_best"),
                "layer": "framework",
                "severity": "high" if no_promote >= 6 else "medium",
                "domain_hint": self._framework_authoring_domain(),
                "source": "attempts",
            }
        ]

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

    @staticmethod
    def _gap_layer_for_action(action: str, framework: str = "") -> tuple[str, str]:
        """Map an action name → (layer, domain_hint) for gap rows.

        Args:
            action: The action name to classify.
            framework: The session's framework, which decides the authoring
                domain for framework-layer rows. Defaults to the serving domain
                so a caller with no framework in hand keeps the old mapping.

        Returns:
            A ``(layer, domain_hint)`` tuple for the action.
        """
        from ..specialists.domains import authoring_domain_for_framework

        a = str(action or "").strip().lower()
        if a in {
            # ``kernel_opt`` names the lane, not a request kind: it is still what
            # a gap row calls kernel work, so it keeps classifying to the kernel layer.
            "kernel_opt",
            "kernel_agent",
            "integrate",
            "trace_analyze",
            "run_gemm_tuning",
            "profile",
            "roofline",
        }:
            return ("kernel_agent", "kernel_switch_specialist")
        if a in {"baseline"}:
            return ("system", "system_specialist")
        return ("framework", authoring_domain_for_framework(framework))

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
            self.shared_state.record_failure_evidence(fe)
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
