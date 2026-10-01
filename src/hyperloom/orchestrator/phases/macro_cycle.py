# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Macro-cycle engine: cycle-focus planning, reloop, soft-restart, and the cycle-start reprofile."""

from __future__ import annotations

import logging as _logging
from typing import Any

from hyperloom.common.env import env_bool

from ..collaborator import CoordinatorCollaborator
from . import machine_state as _phase_state
from ..loop.maintenance import run_lease_and_db_reclaim

log = _logging.getLogger(__name__)

__all__ = ["MacroCycleCollaborator"]


class MacroCycleCollaborator(CoordinatorCollaborator):
    """Macro-cycle planning, focus scoring, soft-restart, reprofile, and orchestration-memory helpers."""

    def __init__(self, coordinator) -> None:
        super().__init__(coordinator)
        # Medium-intensity soft restart at each macro-cycle boundary.
        self._soft_restart_enabled = not env_bool("INFERENCE_OPTIMIZER_DISABLE_CYCLE_SOFT_RESTART")

    def _negative_ledger_domain_counts(self, *, recent_cycles: int = 3) -> dict[str, int]:
        """Summarise recent negative explore-ledger pressure by specialist domain."""
        state = self.shared_state
        cur_cycle = int(getattr(state, "macro_cycle", 0) or 0)
        search = getattr(state, "explore_search", {}) or {}
        rows: list[Any] = []
        if isinstance(search, dict):
            tested = search.get("tested") or {}
            if isinstance(tested, dict):
                rows.extend(tested.values())
            rejected = search.get("rejected") or []
            if isinstance(rejected, list):
                rows.extend(rejected)
        counts: dict[str, int] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                cycle = int(row.get("cycle", cur_cycle) or 0)
            except (TypeError, ValueError):
                cycle = cur_cycle
            if cycle < max(0, cur_cycle - recent_cycles + 1):
                continue
            domain = str(
                row.get("domain")
                or row.get("specialist_domain")
                or row.get("source_domain")
                or row.get("provenance")
                or ""
            ).strip()
            if not domain:
                continue
            counts[domain] = counts.get(domain, 0) + 1
        return counts

    def _plan_cycle_focus(self) -> dict[str, Any]:
        """Pick an advisory specialist-domain focus for the current macro-cycle."""
        from hyperloom.inference_optimizer.roofline_snapshot import BOTTLENECK_DOMAIN_HINTS

        state = self.shared_state
        cycle = int(getattr(state, "macro_cycle", 0) or 0)
        domains = sorted({v[0] for v in BOTTLENECK_DOMAIN_HINTS.values()} | {"freeform_specialist"})
        scores: dict[str, float] = {d: 0.0 for d in domains}
        reasons: dict[str, list[str]] = {d: [] for d in domains}
        shift = getattr(state, "bottleneck_shift", {}) or {}
        to_domain = str(shift.get("to_domain") or "").strip()
        if to_domain:
            scores.setdefault(to_domain, 0.0)
            reasons.setdefault(to_domain, [])
            scores[to_domain] += 5.0
            reasons[to_domain].append(f"matches current bottleneck shift to {shift.get('to') or to_domain}")
        sat = getattr(state, "saturated_directions", {}) or {}
        if isinstance(sat, dict):
            for domain, row in sat.items():
                if not isinstance(row, dict):
                    continue
                d = str(domain or row.get("domain") or "").strip()
                if not d:
                    continue
                scores.setdefault(d, 0.0)
                reasons.setdefault(d, [])
                if bool(row.get("saturated")):
                    scores[d] -= 100.0
                    reasons[d].append(f"saturated at {row.get('within_pct')}% within roofline; deprioritized")
                else:
                    scores[d] += 1.0
                    reasons[d].append("not saturated in latest roofline snapshot")
        log_rows = list(getattr(state, "cycle_strategy_log", []) or [])
        tried = {str(r.get("focus") or "") for r in log_rows if isinstance(r, dict)}
        for row in log_rows:
            if not isinstance(row, dict):
                continue
            domain = str(row.get("focus") or "").strip()
            if not domain:
                continue
            scores.setdefault(domain, 0.0)
            reasons.setdefault(domain, [])
            gd = row.get("gain_delta")
            if isinstance(gd, (int, float)):
                scores[domain] += max(-2.0, min(3.0, float(gd)))
                reasons[domain].append(f"historical cycle gain_delta={float(gd):+.2f}%")
        for domain in domains:
            if domain not in tried:
                scores[domain] += 1.5
                reasons[domain].append("exploration bonus: not yet used as cycle focus")
        negative_counts = self._negative_ledger_domain_counts()
        for domain, count in negative_counts.items():
            scores.setdefault(domain, 0.0)
            reasons.setdefault(domain, [])
            penalty = min(4.0, 0.5 * float(count))
            scores[domain] -= penalty
            reasons[domain].append(f"recent negative ledger count={count} penalty={penalty:.1f}")
        focus = max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0] if scores else "freeform_specialist"
        rationale_bits = reasons.get(focus) or ["fallback focus; no stronger cycle-level evidence"]
        return {
            "cycle": cycle,
            "focus": focus,
            "score": round(float(scores.get(focus, 0.0)), 3),
            "rationale": "; ".join(rationale_bits[:4]),
            "bottleneck_at_start": str(shift.get("to") or self.shared_state.current_top_bottleneck() or ""),
            "saturated_at_start": sorted(
                str(k)
                for k, v in (sat.items() if isinstance(sat, dict) else [])
                if isinstance(v, dict) and bool(v.get("saturated"))
            ),
            "gain_at_start": float(getattr(state, "gain_at_cycle_start", 0.0) or 0.0),
            "gain_delta": None,
        }

    def _record_cycle_strategy_for_current_cycle(self) -> None:
        """Append/update the advisory cycle-strategy row for the current cycle."""
        state = self.shared_state
        planned = self._plan_cycle_focus()
        log_rows = [r for r in (getattr(state, "cycle_strategy_log", []) or []) if isinstance(r, dict)]
        cycle = int(planned.get("cycle", 0) or 0)
        replaced = False
        for idx, row in enumerate(log_rows):
            if int(row.get("cycle", -1) or -1) == cycle:
                merged = dict(row)
                merged.update(planned)
                log_rows[idx] = merged
                replaced = True
                break
        if not replaced:
            log_rows.append(planned)
        state.cycle_strategy_log = log_rows[-50:]

    async def _run_cycle_soft_restart(
        self,
        *,
        prior_cycle: int,
        new_cycle: int,
    ) -> dict[str, Any] | None:
        """Medium-intensity soft restart at a macro-cycle boundary.

        Recycles transient/per-cycle resources (fresh leases, pruned DB, cleared
        caches, re-scoped system prompt) without losing accumulated optimization state;
        ``current_best`` / ``optimization_stack`` / ``explore_search`` are
        preserved. Idempotent.

        Args:
            prior_cycle: The macro-cycle number that just finished.
            new_cycle: The macro-cycle number being entered.

        Returns:
            A summary dict of the restart steps performed, or ``None`` when the
            soft restart is disabled.
        """
        if not self._soft_restart_enabled:
            return None
        summary: dict[str, Any] = {
            "prior_cycle": int(prior_cycle),
            "new_cycle": int(new_cycle),
        }
        # Reap leases, reclaim orphaned running tasks, prune DB.
        await run_lease_and_db_reclaim(self, summary, reason="cycle_soft_restart")
        log.info(
            "cycle soft-restart %d → %d: %s",
            int(prior_cycle),
            int(new_cycle),
            summary,
        )
        await self._coord.writeback._record_observation(
            "coordinator",
            "observation",
            {"kind": "cycle_soft_restart", **summary},
        )
        return summary

    async def _on_cycle_start_reprofile(self, *, from_phase: str) -> None:
        """Force a fresh analysis at the start of a reopened macro-cycle.

        Reached on every cycle start now. It used to be attached to the config-arm
        entry, and the reloop targeted FRAMEWORK_AGENT whenever the framework
        phase was enabled -- so with the default configuration this never ran,
        and each new cycle re-targeted the bottleneck the *previous* cycle
        measured. One phase means one entry, and the reprofile happens.

        Args:
            from_phase: The phase being left; only a SWEEP origin starts a cycle.
        """
        if (from_phase or "").upper() == _phase_state.PHASE_SWEEP and int(self.shared_state.macro_cycle or 0) > 0:
            task = await self._coord.phase_prelude._enqueue_internal_analysis_task(
                reason="cycle_start",
            )
            if task is None:
                return
            self.shared_state.auto_roofline_pending_task_id = task.task_id
            log.info(
                "cycle %d start: forced reprofile task=%s",
                int(self.shared_state.macro_cycle or 0),
                task.task_id,
            )
