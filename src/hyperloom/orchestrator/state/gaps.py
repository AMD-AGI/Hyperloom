# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gap ledger: upsert, dedup, and gap-based signal extraction."""

from __future__ import annotations

import logging as _logging
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any

from ..collaborator import CoordinatorCollaborator

log = _logging.getLogger(__name__)

__all__ = ["GapsStateMixin", "GapRefreshCollaborator"]


def _shared_state_module():
    """Import parent shared_state lazily to avoid a module-level cycle."""
    from . import shared_state

    return shared_state


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
        """Insert or update one gap row, keyed by ``canonical_id``. Coordinator-only writer (Inv-1 single-writer + CORE_STATE_FIELDS lock). Returns the merged entry."""
        if not isinstance(entry, dict):
            return {}
        cid = str(entry.get("canonical_id") or "").strip()
        if not cid:
            return {}
        ss = _shared_state_module()
        now = ss._now_iso()
        existing = self.find_gap(cid)
        if existing is None:
            merged: dict[str, Any] = {
                "canonical_id": cid,
                "symptom": str(entry.get("symptom") or ""),
                "layer": str(entry.get("layer") or ""),
                "severity": str(entry.get("severity") or "medium"),
                "domain_hint": str(entry.get("domain_hint") or ""),
                "source": str(entry.get("source") or ""),
                # Optional origin reference (PR/blog URL).
                "provenance": str(entry.get("provenance") or ""),
                "first_seen_ts": str(entry.get("first_seen_ts") or now),
                "last_updated_ts": now,
                "attempts": list(entry.get("attempts") or []),
            }
            if len(merged["attempts"]) > ss._GAPS_ATTEMPTS_HISTORY:
                merged["attempts"] = merged["attempts"][-ss._GAPS_ATTEMPTS_HISTORY:]
            self.gaps.append(merged)
        else:
            # Field-wise merge: incoming non-empty values win except ``first_seen_ts``.
            for key in ("symptom", "layer", "severity", "domain_hint", "source", "provenance"):
                incoming = entry.get(key)
                if incoming:
                    existing[key] = str(incoming)
            existing.setdefault("first_seen_ts", str(entry.get("first_seen_ts") or now))
            existing["last_updated_ts"] = now
            incoming_attempts = list(entry.get("attempts") or [])
            if incoming_attempts:
                merged_attempts = list(existing.get("attempts") or []) + incoming_attempts
                # Capped tail; callers supply newest-last lists (convention).
                if len(merged_attempts) > ss._GAPS_ATTEMPTS_HISTORY:
                    merged_attempts = merged_attempts[-ss._GAPS_ATTEMPTS_HISTORY:]
                existing["attempts"] = merged_attempts
            merged = existing
        # Enforce global cap, trimming oldest after the upsert so the just-touched gap is retained.
        if len(self.gaps) > ss._GAPS_MAX_ENTRIES:
            others = [g for g in self.gaps if g is not merged]

            def _sort_key(g: dict[str, Any]) -> str:
                """Sort key for gap trimming: newest-updated timestamp."""
                return str(g.get("last_updated_ts") or g.get("first_seen_ts") or "")

            others.sort(key=_sort_key)
            keep_count = ss._GAPS_MAX_ENTRIES - 1
            others = others[-keep_count:] if keep_count > 0 else []
            self.gaps = others + [merged]
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
        ss = _shared_state_module()
        attempts.append(dict(attempt) | {"ts": str(attempt.get("ts") or ss._now_iso())})
        if len(attempts) > ss._GAPS_ATTEMPTS_HISTORY:
            attempts = attempts[-ss._GAPS_ATTEMPTS_HISTORY:]
        gap["attempts"] = attempts
        gap["last_updated_ts"] = ss._now_iso()
        return gap


class GapRefreshCollaborator(CoordinatorCollaborator):
    """Gap-signal extraction from baselines, attempt history, and research hints."""

    async def _refresh_gaps(self, *, reason: str) -> None:
        """Refresh :attr:`SharedState.gaps` from observable signals. Additive upsert deduped by canonical_id; best-effort.

        Args:
            reason: Tag describing the refresh trigger, used only in logging.
        """
        state = self.shared_state
        try:
            for entry in self._extract_gaps_from_baseline():
                state.upsert_gap(entry)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("gaps refresh: baseline extraction failed")
        try:
            for entry in self._extract_gaps_from_attempts():
                state.upsert_gap(entry)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("gaps refresh: attempts extraction failed")

        plane = getattr(self, "knowledge_plane", None)
        if plane is not None and hasattr(plane, "recipe_kb_traverse_issues"):
            try:
                traverse = getattr(plane, "recipe_kb_traverse_issues")
                rows = traverse(
                    model_class=getattr(state, "model_class", "") or "",
                    gpu_type=getattr(state, "gpu_type", "") or "",
                )
                if isinstance(rows, list):
                    for entry in rows:
                        if isinstance(entry, dict):
                            entry = dict(entry)
                            entry.setdefault("source", "recipe_kb")
                            state.upsert_gap(entry)
            except Exception:  # noqa: BLE001 — defensive
                log.warning(
                    "gaps refresh: recipe_kb_traverse_issues failed (reason=%s)",
                    reason,
                    exc_info=True,
                )
        log.debug(
            "gaps refresh (reason=%s): %d gaps after merge",
            reason,
            len(state.gaps),
        )

    def _extract_gaps_from_baseline(self) -> list[dict[str, Any]]:
        """Derive initial gap rows from the baseline snapshot (throughput_below_target, baseline_unstable); reuse the workload canonical_id (``_workload_canonical_id``, matching ``recipe_kb_t0.run_t0_anchor``) so traverse rows align.

        Returns:
            A list of gap row dicts derived from the baseline; empty when no
            baseline throughput is recorded.
        """
        state = self.shared_state
        gaps: list[dict[str, Any]] = []
        if state.baseline_tput <= 0:
            return gaps
        anchor = self._workload_canonical_id()
        target_gap = float(getattr(state, "target_gap_pct", 0.0) or 0.0)
        if target_gap > 0.0:
            severity = "high" if target_gap >= 10.0 else "medium" if target_gap >= 3.0 else "low"
            gaps.append(
                {
                    "canonical_id": f"{anchor}#throughput_below_target",
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
                    "canonical_id": f"{anchor}#baseline_unstable",
                    "symptom": (f"baseline crashed {state.baseline_failure_streak} consecutive time(s)"),
                    "layer": "system",
                    "severity": ("high" if state.baseline_failure_streak >= 2 else "medium"),
                    "domain_hint": "system_specialist",
                    "source": "baseline",
                }
            )
        return gaps

    def _extract_gaps_from_attempts(self) -> list[dict[str, Any]]:
        """Derive gaps from rolling failures + winners history (recurring (action, error_class[, variant]) + explore plateau).

        Returns:
            A list of gap row dicts derived from recurring action failures and
            an explore-plateau signal.
        """
        state = self.shared_state
        anchor = self._workload_canonical_id()
        gaps: list[dict[str, Any]] = []

        # Already capped by ``record_action_failure``; read the whole log.
        seen_failures: dict[str, dict[str, Any]] = {}
        for row in state.last_action_failures or []:
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "").strip() or "unknown"
            err = str(row.get("error_class") or "").strip() or "unknown_error"
            variant = str(row.get("variant_name") or "").strip()
            # Variant discriminator keeps distinct crash causes in distinct gaps.
            key = f"{action}::{err}::{variant}" if variant else f"{action}::{err}"
            layer, domain = self._gap_layer_for_action(action, str(getattr(self.shared_state, "framework", "") or ""))
            excerpt = str(row.get("error_excerpt") or "")
            detail = next((ln.strip() for ln in excerpt.splitlines() if ln.strip()), err)[:200]
            symptom = f"{action}/{variant} fails: {detail}" if variant else f"{action} repeatedly fails with {detail}"
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
                cid_variant = f":{variant}" if variant else ""
                seen_failures[key] = {
                    "canonical_id": f"{anchor}#fail:{action}:{err}{cid_variant}",
                    "symptom": symptom,
                    "layer": layer,
                    "severity": "medium",
                    "domain_hint": domain,
                    "source": "attempts",
                    "attempts": [attempt],
                }
        gaps.extend(seen_failures.values())

        no_promote = int(state.params_no_promote_streak or 0)
        explore_search = state.explore_search or {}
        winners_hist = []
        if isinstance(explore_search, dict):
            winners_hist = list(explore_search.get("winners_history") or [])
        recent_promotions = sum(
            1 for w in winners_hist[-5:] if isinstance(w, dict) and float(w.get("gain_pct") or 0.0) > 0.0
        )
        if no_promote >= 3 and recent_promotions == 0:
            gaps.append(
                {
                    "canonical_id": f"{anchor}#explore_plateau",
                    "symptom": (f"{no_promote} consecutive grid rounds without a new current_best"),
                    "layer": "framework",
                    "severity": "high" if no_promote >= 6 else "medium",
                    "domain_hint": self._framework_authoring_domain(),
                    "source": "attempts",
                }
            )
        return gaps

    def _seed_gaps_from_research_hints(self) -> None:
        """Inject research hints as advisory gaps[] seeds (idempotent)."""
        from ..knowledge import research_hints as _research_hints

        hints = _research_hints.load_hints(self.session_dir)
        for hint in hints:
            what = str(hint.get("what") or "").strip()
            source = str(hint.get("source") or "").strip()
            if not what or not source:
                continue
            tags = hint.get("domain_tags") or []
            key = f"{what.lower()}::{source.lower()}"
            cid = f"gap.research_hint.{sha1(key.encode()).hexdigest()[:16]}"
            try:
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
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "research-scout: upsert_gap failed for %s",
                    cid,
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
            # ``kernel_opt`` names the lane, not a request kind: it is still in
            # KERNEL_LANE_TASK_KINDS and still what a gap row calls kernel work,
            # so it keeps classifying to the kernel layer.
            "kernel_opt",
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
