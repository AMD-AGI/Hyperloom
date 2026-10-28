# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gap ledger: upsert, dedup, and gap-based signal extraction."""

from __future__ import annotations

import logging as _logging
from datetime import datetime, timezone
from hashlib import sha1
from typing import TYPE_CHECKING, Any

from ..collaborator import CoordinatorCollaborator
from hyperloom.common.coerce import first_text
from hyperloom.common.timeutil import now_iso

from ._shared_state import gap_rows
from .failure_evidence import UNMEASURED_OUTCOMES, failure_from_variant_outcome

if TYPE_CHECKING:
    from .task_registry import Task

log = _logging.getLogger(__name__)

__all__ = ["GapsStateMixin", "GapRefreshCollaborator"]

# gap ledger caps; both enforced in upsert_gap.
_GAPS_MAX_ENTRIES = 50
_GAPS_ATTEMPTS_HISTORY = 20

# Free-text gap fields an incoming row may restate; ``severity`` defaults to "medium" rather than "".
_GAP_TEXT_FIELDS: tuple[str, ...] = ("symptom", "layer", "severity", "domain_hint", "source", "provenance")


def _merge_gap_attempts(existing: list[Any], incoming: list[Any]) -> list[Any]:
    """Merge ``incoming`` attempts into ``existing`` once each, newest-last, capped at the history limit.

    ``refresh_gaps`` re-derives every attempt from the whole failure log on each call, so an equal attempt (same
    failure: action, error class, variant and ts) is the same one, not a new one. Equal rows take the incoming
    position so a re-derived log keeps its newest-last order under the cap.
    """
    older = [a for a in existing if a not in incoming]
    merged: list[Any] = []
    for attempt in older + incoming:
        if attempt not in merged:
            merged.append(attempt)
    # Capped tail; callers supply newest-last lists (convention).
    return merged[-_GAPS_ATTEMPTS_HISTORY:]


class GapsStateMixin:
    """Gap ledger read/write helpers mixed into SharedState."""

    def find_gap(self, canonical_id: str) -> dict[str, Any] | None:
        """Return the gap entry matching ``canonical_id`` (or ``None``)."""
        if not canonical_id:
            return None
        cid = str(canonical_id)
        for gap in self.gaps:
            if isinstance(gap, dict) and str(gap.get("canonical_id") or "") == cid:
                return gap
        return None

    def upsert_gap(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Insert or update one gap row, keyed by ``canonical_id``. Coordinator-only writer. Returns the merged entry."""
        if not isinstance(entry, dict):
            return {}
        cid = str(entry.get("canonical_id") or "").strip()
        if not cid:
            return {}
        now = now_iso()
        existing = self.find_gap(cid)
        if existing is None:
            merged: dict[str, Any] = {
                "canonical_id": cid,
                # ``provenance`` is an optional origin reference (PR/blog URL).
                **{key: str(entry.get(key) or "") for key in _GAP_TEXT_FIELDS},
                "severity": str(entry.get("severity") or "medium"),
                "first_seen_ts": str(entry.get("first_seen_ts") or now),
                "last_updated_ts": now,
                "attempts": list(entry.get("attempts") or [])[-_GAPS_ATTEMPTS_HISTORY:],
            }
            self.gaps.append(merged)
        else:
            # Field-wise merge: incoming non-empty values win except ``first_seen_ts``.
            for key in _GAP_TEXT_FIELDS:
                incoming = entry.get(key)
                if incoming:
                    existing[key] = str(incoming)
            existing.setdefault("first_seen_ts", str(entry.get("first_seen_ts") or now))
            existing["last_updated_ts"] = now
            incoming_attempts = list(entry.get("attempts") or [])
            if incoming_attempts:
                existing["attempts"] = _merge_gap_attempts(list(existing.get("attempts") or []), incoming_attempts)
            merged = existing
        # Enforce global cap, trimming oldest after the upsert so the just-touched gap is retained.
        if len(self.gaps) > _GAPS_MAX_ENTRIES:
            others = sorted((g for g in self.gaps if g is not merged), key=gap_rows.recency_key)
            self.gaps = others[-(_GAPS_MAX_ENTRIES - 1) :] + [merged]
        return merged

    def append_gap_attempt(
        self,
        canonical_id: str,
        attempt: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Append one attempt row to an existing gap; returns the gap or ``None`` when unknown."""
        gap = self.find_gap(canonical_id)
        if gap is None:
            return None
        attempts = list(gap.get("attempts") or [])
        attempts.append(dict(attempt) | {"ts": str(attempt.get("ts") or now_iso())})
        if len(attempts) > _GAPS_ATTEMPTS_HISTORY:
            attempts = attempts[-_GAPS_ATTEMPTS_HISTORY:]
        gap["attempts"] = attempts
        gap["last_updated_ts"] = now_iso()
        return gap

    def retire_gap(self, canonical_id: str) -> bool:
        """Drop the gap matching ``canonical_id`` if present; returns whether a row was removed. Coordinator-only writer.

        ``upsert_gap`` only ever adds or ages rows out under the cap, so a signal-derived gap whose condition has
        cleared would otherwise keep its severity and keep being ranked. The deriving extractor retires it here.
        """
        cid = str(canonical_id or "")
        if not cid:
            return False
        kept = [g for g in self.gaps if not (isinstance(g, dict) and str(g.get("canonical_id") or "") == cid)]
        if len(kept) == len(self.gaps):
            return False
        self.gaps = kept
        return True


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
        profile_gaps = self._extract_gaps_from_profile(workload_id)
        for entry in profile_gaps:
            state.upsert_gap(entry)
        if not profile_gaps:
            # Coverage is complete (or no longer applicable): retire the prefill gap so a resolved, high-severity row
            # does not keep steering specialists. A no-op when the gap was never raised.
            state.retire_gap(f"{workload_id}#phase_unmeasured:prefill")
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

    def _extract_gaps_from_profile(self, workload_id: str) -> list[dict[str, Any]]:
        """Emit a gap when the last profile covered only one serving phase on a workload that exercises the other.

        A partial capture drops the missing phase out of the Amdahl denominator, so ranking against it steers the run
        to the phase that happened to be measured. Scoped to a prefill miss on an agentic (prefill-heavy) workload —
        the failure mode where the dominant phase goes unseen — so a synthetic decode-steady-state capture stays quiet.
        When a later profile captures prefill the extractor emits nothing and ``refresh_gaps`` retires the gap, so a
        resolved row does not linger at ``severity: high``.

        Returns:
            A single-row list carrying the prefill-unmeasured gap, or empty when coverage is complete or the workload
            is not agentic.
        """
        from hyperloom.common.perf_metric import is_agentx_mode

        state = self.shared_state
        coverage = getattr(state, "last_profile_phase_coverage", None) or {}
        if not coverage.get("partial") or str(coverage.get("missing") or "") != "prefill":
            return []
        if not is_agentx_mode(getattr(state, "benchmark_mode", "")):
            return []
        return [
            {
                "canonical_id": f"{workload_id}#phase_unmeasured:prefill",
                "symptom": (
                    "prefill is absent from the last profile split — on this agentic workload prefill dominates the "
                    "engine, so its component shares are unmeasured, not zero. Instrument the prefill path directly "
                    "(a targeted per-component eval) before ranking against decode-only data."
                ),
                "layer": "framework",
                "severity": "high",
                "domain_hint": self._framework_authoring_domain(),
                "source": "profile",
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
