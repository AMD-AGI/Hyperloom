# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Macro-cycle engine: cycle-focus planning, reloop, soft-restart, and the cycle-start reprofile."""

from __future__ import annotations

import logging as _logging
from collections.abc import Iterable
from typing import Any

from hyperloom.common.env import env_bool

from ..actions.executors.explore import TS_SKIPPED_DEDUP
from ..collaborator import CoordinatorCollaborator
from . import machine_state as _phase_state
from ..loop.maintenance import run_lease_and_db_reclaim

log = _logging.getLogger(__name__)

__all__ = ["MacroCycleCollaborator"]


class _FocusScoreboard:
    """Per-domain cycle-focus scores and the evidence each one rests on."""

    def __init__(self, domains: Iterable[str]) -> None:
        self.scores: dict[str, float] = dict.fromkeys(domains, 0.0)
        self.reasons: dict[str, list[str]] = {domain: [] for domain in self.scores}

    def admit(self, domain: str) -> None:
        """Make ``domain`` selectable at its current score."""
        self.scores.setdefault(domain, 0.0)
        self.reasons.setdefault(domain, [])

    def credit(self, domain: str, delta: float, reason: str) -> None:
        """Move ``domain``'s score by ``delta`` and record why."""
        self.admit(domain)
        self.scores[domain] += delta
        self.reasons[domain].append(reason)

    def leader(self) -> str:
        """The highest-scoring domain, ties broken by name; empty when nothing is scored."""
        if not self.scores:
            return ""
        return max(self.scores.items(), key=lambda kv: (kv[1], kv[0]))[0]


class MacroCycleCollaborator(CoordinatorCollaborator):
    """Macro-cycle planning, focus scoring, soft-restart, reprofile, and orchestration-memory helpers."""

    def __init__(self, coordinator) -> None:
        super().__init__(coordinator)
        # Medium-intensity soft restart at each macro-cycle boundary.
        self._soft_restart_enabled = not env_bool("INFERENCE_OPTIMIZER_DISABLE_CYCLE_SOFT_RESTART")

    def _negative_ledger_domain_counts(self, *, recent_cycles: int = 3) -> dict[str, int]:
        """Summarise recent negative explore-ledger pressure by specialist domain."""
        state = self.shared_state
        cur_cycle = int(state.macro_cycle or 0)
        search = state.explore_search or {}
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
            # A row the round answered without measuring rejects nothing.
            if not isinstance(row, dict) or row.get("outcome") == TS_SKIPPED_DEDUP:
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

    def _score_bottleneck_shift(self, board: _FocusScoreboard, shift: dict[str, Any]) -> None:
        """Credit the domain the latest bottleneck shift points at."""
        to_domain = str(shift.get("to_domain") or "").strip()
        if to_domain:
            board.credit(to_domain, 5.0, f"matches current bottleneck shift to {shift.get('to') or to_domain}")

    def _score_roofline_saturation(self, board: _FocusScoreboard, saturated: Any) -> None:
        """Deprioritise the directions the latest roofline snapshot reports as saturated."""
        if not isinstance(saturated, dict):
            return
        for domain, row in saturated.items():
            if not isinstance(row, dict):
                continue
            name = str(domain or row.get("domain") or "").strip()
            if not name:
                continue
            if bool(row.get("saturated")):
                board.credit(name, -100.0, f"saturated at {row.get('within_pct')}% within roofline; deprioritized")
            else:
                board.credit(name, 1.0, "not saturated in latest roofline snapshot")

    def _score_cycle_history(self, board: _FocusScoreboard, log_rows: list[Any]) -> None:
        """Credit each earlier focus with the cycle gain it produced."""
        for row in log_rows:
            if not isinstance(row, dict):
                continue
            domain = str(row.get("focus") or "").strip()
            if not domain:
                continue
            board.admit(domain)
            gain_delta = row.get("gain_delta")
            if isinstance(gain_delta, (int, float)):
                board.credit(
                    domain,
                    max(-2.0, min(3.0, float(gain_delta))),
                    f"historical cycle gain_delta={float(gain_delta):+.2f}%",
                )

    def _score_unexplored(self, board: _FocusScoreboard, domains: list[str], log_rows: list[Any]) -> None:
        """Favour a domain no earlier cycle has focused on."""
        tried = {str(row.get("focus") or "") for row in log_rows if isinstance(row, dict)}
        for domain in domains:
            if domain not in tried:
                board.credit(domain, 1.5, "exploration bonus: not yet used as cycle focus")

    def _score_negative_ledger(self, board: _FocusScoreboard) -> None:
        """Penalise the domains the recent explore ledger keeps rejecting."""
        for domain, count in self._negative_ledger_domain_counts().items():
            penalty = min(4.0, 0.5 * float(count))
            board.credit(domain, -penalty, f"recent negative ledger count={count} penalty={penalty:.1f}")

    def plan_cycle_focus(self) -> dict[str, Any]:
        """Pick an advisory specialist-domain focus for the current macro-cycle."""
        from hyperloom.inference_optimizer.roofline_snapshot import BOTTLENECK_DOMAIN_HINTS

        state = self.shared_state
        domains = sorted({v[0] for v in BOTTLENECK_DOMAIN_HINTS.values()} | {"freeform_specialist"})
        board = _FocusScoreboard(domains)
        shift = state.bottleneck_shift or {}
        sat = state.saturated_directions or {}
        log_rows = list(state.cycle_strategy_log or [])
        self._score_bottleneck_shift(board, shift)
        self._score_roofline_saturation(board, sat)
        self._score_cycle_history(board, log_rows)
        self._score_unexplored(board, domains, log_rows)
        self._score_negative_ledger(board)
        focus = board.leader() or "freeform_specialist"
        rationale_bits = board.reasons.get(focus) or ["fallback focus; no stronger cycle-level evidence"]
        return {
            "cycle": int(state.macro_cycle or 0),
            "focus": focus,
            "score": round(float(board.scores.get(focus, 0.0)), 3),
            "rationale": "; ".join(rationale_bits[:4]),
            "bottleneck_at_start": str(shift.get("to") or self.shared_state.current_top_bottleneck() or ""),
            "saturated_at_start": sorted(
                str(k)
                for k, v in (sat.items() if isinstance(sat, dict) else [])
                if isinstance(v, dict) and bool(v.get("saturated"))
            ),
            "gain_at_start": float(state.gain_at_cycle_start or 0.0),
            "gain_delta": None,
        }

    def record_cycle_strategy_for_current_cycle(self) -> None:
        """Append/update the advisory cycle-strategy row for the current cycle."""
        state = self.shared_state
        planned = self.plan_cycle_focus()
        log_rows = [r for r in (state.cycle_strategy_log or []) if isinstance(r, dict)]
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

    async def run_cycle_soft_restart(
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
        await run_lease_and_db_reclaim(self._coord, summary, reason="cycle_soft_restart")
        log.info(
            "cycle soft-restart %d → %d: %s",
            int(prior_cycle),
            int(new_cycle),
            summary,
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {"kind": "cycle_soft_restart", **summary},
        )
        return summary

    async def on_cycle_start_reprofile(self, *, from_phase: str) -> None:
        """Force a fresh analysis at the start of a reopened macro-cycle.

        Reached on every cycle start, whichever phase the reloop targets, so a
        new cycle never re-targets the bottleneck the *previous* cycle measured.

        Args:
            from_phase: The phase being left; only a SWEEP origin starts a cycle.
        """
        if (from_phase or "").upper() == _phase_state.PHASE_SWEEP and int(self.shared_state.macro_cycle or 0) > 0:
            task = await self._coord.phase_prelude.enqueue_internal_analysis_task(
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
