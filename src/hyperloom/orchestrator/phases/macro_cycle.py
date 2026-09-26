# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Macro-cycle engine: cycle-focus planning, reloop, soft-restart, and the cycle-start reprofile."""

from __future__ import annotations

import logging as _logging
from typing import Any

from ..collaborator import CoordinatorCollaborator
from . import machine_state as _phase_state
from ..loop.maintenance import run_lease_and_db_reclaim

log = _logging.getLogger(__name__)

__all__ = ["MacroCycleCollaborator"]


class MacroCycleCollaborator(CoordinatorCollaborator):
    """Macro-cycle planning, focus scoring, soft-restart, reprofile, and orchestration-memory helpers."""

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

    def _cycle_strategy_block(self) -> str:
        """Render persisted cycle focus facts for the orchestration prompt."""
        rows = [r for r in (getattr(self.shared_state, "cycle_strategy_log", []) or []) if isinstance(r, dict)]
        if not rows:
            return ""
        cur_cycle = int(getattr(self.shared_state, "macro_cycle", 0) or 0)
        current = next((r for r in reversed(rows) if int(r.get("cycle", -1) or -1) == cur_cycle), rows[-1])
        lines = [
            f"=== Cycle {cur_cycle} strategy ===",
            f"focus={current.get('focus') or '(none)'} score={current.get('score', 0)}",
        ]
        rationale = str(current.get("rationale") or "").strip()
        if rationale:
            lines.append(f"rationale: {rationale}")
        saturated = current.get("saturated_at_start") or []
        if saturated:
            lines.append(f"saturated_at_start={saturated}")
        prior = [r for r in rows if int(r.get("cycle", -1) or -1) != cur_cycle][-5:]
        if prior:
            lines.append("previous cycles:")
            for row in prior:
                lines.append(
                    f"  - cycle={row.get('cycle')} focus={row.get('focus')} "
                    f"gain_delta={row.get('gain_delta')} saturated={row.get('saturated_at_start') or []}"
                )
        lines.append("Advisory only: use this as a prior, not a dispatch gate.")
        return "\n".join(lines)

    def _apply_macro_cycle_reloop(self, evidence: dict[str, Any]) -> None:
        """Open a new macro-cycle on a SWEEP loopback into FRAMEWORK_AGENT.

        Increments ``macro_cycle``, persists the no-gain streak + per-cycle gain
        anchor, and resets per-cycle counters (including re-opening FRAMEWORK) for
        a fresh budget / plateau evaluation. The explore ledger is preserved.

        Args:
            evidence: The loopback evidence dict from ``compute_next_phase``;
                may carry ``no_gain_cycle_streak_effective`` which is persisted
                onto the new cycle.
        """
        state = self.shared_state
        prior_cycle = int(getattr(state, "macro_cycle", 0) or 0)
        prev_delta = float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0) - float(
            getattr(state, "gain_at_cycle_start", 0.0) or 0.0
        )
        rows = [r for r in (getattr(state, "cycle_strategy_log", []) or []) if isinstance(r, dict)]
        for row in rows:
            if int(row.get("cycle", -1) or -1) == prior_cycle and row.get("gain_delta") is None:
                row["gain_delta"] = round(prev_delta, 6)
        state.cycle_strategy_log = rows[-50:]
        state.macro_cycle = prior_cycle + 1
        if isinstance(evidence, dict) and "no_gain_cycle_streak_effective" in evidence:
            state.no_gain_cycle_streak = int(evidence.get("no_gain_cycle_streak_effective", 0) or 0)
        # Anchor gain for the cycle we are about to start.
        try:
            state.gain_at_cycle_start = float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0)
        except (TypeError, ValueError):
            state.gain_at_cycle_start = 0.0
        # Reset per-cycle counters.
        state.reset_per_cycle_plateau_state()
        self._record_cycle_strategy_for_current_cycle()
        log.info(
            "Coordinator: macro-cycle reloop %d → %d (no_gain_streak=%d, gain_anchor=%.4f)",
            prior_cycle,
            state.macro_cycle,
            state.no_gain_cycle_streak,
            state.gain_at_cycle_start,
        )

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
        if not getattr(self, "_cycle_soft_restart", False):
            return None
        summary: dict[str, Any] = {
            "prior_cycle": int(prior_cycle),
            "new_cycle": int(new_cycle),
        }
        # 1) Capture the cycle's working memory, then rebuild the system prompt
        # for the new cycle around the directive that capture produced.
        summary["memory_captured"] = await self._capture_cycle_memory()
        summary["orch_prompt_reseeded"] = self._reseed_orch_prompt_for_cycle()
        # 2-3) Reap leases, reclaim orphaned running tasks, prune DB.
        await run_lease_and_db_reclaim(self, summary, reason="cycle_soft_restart")
        log.info(
            "cycle soft-restart %d → %d: %s",
            int(prior_cycle),
            int(new_cycle),
            summary,
        )
        try:
            await self._record_observation(
                "coordinator",
                "observation",
                {"kind": "cycle_soft_restart", **summary},
            )
        except Exception:
            log.exception("cycle soft-restart: observation write failed")
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
        if (from_phase or "").upper() == _phase_state.PHASE_SWEEP and int(
            getattr(self.shared_state, "macro_cycle", 0) or 0
        ) > 0:
            try:
                task = await self._enqueue_internal_analysis_task(
                    reason="cycle_start",
                )
                if task is None:
                    return
                self.shared_state.auto_roofline_pending_task_id = task.task_id
                log.info(
                    "cycle %d start: forced reprofile task=%s",
                    int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                    task.task_id,
                )
            except Exception:
                log.exception(
                    "cycle start: forced reprofile enqueue failed",
                )
