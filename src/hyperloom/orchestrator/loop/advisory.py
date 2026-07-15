# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
from typing import Any
from ..phases import machine_state as _phase_state

import logging as _logging
log = _logging.getLogger(__name__)


class AdvisoryCollaborator:
    """Extracted collaborator; delegates unknown attrs to its Coordinator."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_coord"), name)

    def _plateau_advisory_block(self) -> str:
        """Render the plateau-judgment advisory block (EXPLORE/KERNEL/FRAMEWORK).

        KERNEL / FRAMEWORK plateaus are advisory only (never auto-exit the
        phase). An EXPLORE plateau is advisory in non-cyclic mode, but in cyclic
        mode (default) it deterministically advances EXPLORE → KERNEL_AGENT via
        ``explore_no_more_leverage`` (a non-terminal lever switch); the rendered
        footer states which regime is active.

        Returns:
            The rendered plateau advisory text, or ``""`` when no plateau
            signal is active for the current phase.
        """
        state = self.shared_state
        phase = (getattr(state, "phase", "") or "").strip().upper()
        overrides = getattr(state, "plateau_overrides", None) or {}
        if not isinstance(overrides, dict):
            overrides = {}
        lines: list[str] = []
        if phase == _phase_state.PHASE_EXPLORE:
            triggered, evidence = _phase_state.compute_plateau_explore(
                state,
                lookback=int(
                    overrides.get(
                        "explore_lookback",
                        _phase_state.DEFAULT_PLATEAU_EXPLORE_LOOKBACK,
                    )
                ),
                keep_gain_threshold_pct=float(
                    overrides.get(
                        "explore_keep_gain_pct",
                        _phase_state.DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT,
                    )
                ),
                empty_streak_threshold=int(
                    overrides.get(
                        "explore_empty_streak",
                        _phase_state.DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK,
                    )
                ),
            )
            if triggered:
                lines.append("EXPLORE plateau detected: low recent KEEP gain plus specialist empty streak.")
                lines.append(
                    "  recent_keep_gain_pct="
                    f"{evidence.get('recent_keep_gain_pct', 0.0)} "
                    f"threshold={evidence.get('keep_gain_threshold_pct', 0.0)} "
                    f"empty_streak={evidence.get('empty_streak', 0)} "
                    f"streak_threshold={evidence.get('empty_streak_threshold', 0)}"
                )
        elif phase == _phase_state.PHASE_KERNEL_AGENT:
            triggered, evidence = _phase_state.compute_plateau_kernel(
                state,
                lookback=int(
                    overrides.get(
                        "kernel_lookback",
                        _phase_state.DEFAULT_PLATEAU_KERNEL_LOOKBACK,
                    )
                ),
                revert_streak_threshold=int(
                    overrides.get(
                        "kernel_revert_streak",
                        _phase_state.DEFAULT_PLATEAU_KERNEL_REVERT_STREAK,
                    )
                ),
                keep_gain_threshold_pct=float(
                    overrides.get(
                        "kernel_keep_gain_pct",
                        _phase_state.DEFAULT_PLATEAU_KERNEL_KEEP_GAIN_PCT,
                    )
                ),
            )
            if triggered:
                lines.append("KERNEL_AGENT plateau detected: REVERT streak or low recent KEEP gain.")
                lines.append(
                    "  revert_streak="
                    f"{evidence.get('revert_streak', 0)} "
                    f"threshold={evidence.get('revert_streak_threshold', 0)} "
                    f"recent_keep_gain_pct={evidence.get('recent_keep_gain_pct', 0.0)} "
                    f"keep_gain_threshold_pct={evidence.get('keep_gain_threshold_pct', 0.0)}"
                )
        elif phase == _phase_state.PHASE_FRAMEWORK_AGENT:
            triggered, evidence = _phase_state.compute_plateau_framework_agent(
                state,
                lookback=int(
                    overrides.get(
                        "framework_lookback",
                        _phase_state.DEFAULT_FRAMEWORK_PLATEAU_LOOKBACK,
                    )
                ),
                keep_gain_threshold_pct=float(
                    overrides.get(
                        "framework_keep_gain_pct",
                        _phase_state.DEFAULT_FRAMEWORK_PLATEAU_KEEP_GAIN_PCT,
                    )
                ),
            )
            if triggered:
                lines.append("FRAMEWORK_AGENT plateau detected: recent batches all below keep-gain threshold.")
                lines.append(
                    "  lookback="
                    f"{evidence.get('lookback', 0)} "
                    f"keep_gain_pct_threshold={evidence.get('keep_gain_pct_threshold', 0.0)} "
                    f"batch_max_gains={evidence.get('batch_max_gains', [])}"
                )
        if not lines:
            return ""
        if phase == _phase_state.PHASE_EXPLORE and _phase_state.is_cyclic_phases_enabled():
            lines.append(
                "Note: in cyclic mode a detected EXPLORE plateau "
                "deterministically advances EXPLORE → KERNEL_AGENT (non-terminal "
                "lever switch, reason=explore_no_more_leverage); it does not "
                "end the run. You may still request an earlier advance with an "
                "escalate_strategy_change hint, or keep exploring until the "
                "plateau/budget gate fires."
            )
        else:
            lines.append(
                "Phase advance is driven only by hard limits (IR-6 force-exit, "
                "phase budget, terminal stop_reason) or explicit "
                "escalate_strategy_change hints; this block is informational."
            )
        return "\n".join(lines)

    def _dominant_roofline_direction(self) -> tuple[str, float]:
        """Return ``(direction, pct)`` for the most-saturated roofline direction
        in the latest snapshot; ``("", 0.0)`` when no snapshot is available.

        Returns:
            A ``(direction, pct)`` tuple for the dominant roofline direction, or
            ``("", 0.0)`` when no snapshot exists.
        """
        from ..kernel.roofline_snapshot import dominant_direction

        snaps = getattr(self.shared_state, "roofline_snapshots", None) or []
        if not snaps or not isinstance(snaps[-1], dict):
            return "", 0.0
        return dominant_direction(snaps[-1])

    def _bottleneck_redirect_advisory_block(self) -> str:
        """Render the R3 cyclic bottleneck-redirect advisory (EXPLORE only).

        Active only in cyclic mode when a prior cycle's plateau flagged
        ``pending_bottleneck_switch``. Names the bottleneck we plateaued on, the
        current dominant roofline direction, and a suggested specialist domain so
        Orchestration redirects the new cycle's dispatch. Advisory, never gates.

        Returns:
            The rendered bottleneck-redirect advisory text, or ``""`` when not
            applicable.
        """
        state = self.shared_state
        if not _phase_state.is_cyclic_phases_enabled():
            return ""
        if (getattr(state, "phase", "") or "").strip().upper() != _phase_state.PHASE_EXPLORE:
            return ""
        sat = getattr(state, "saturated_directions", {}) or {}
        saturated = {
            str(k): v
            for k, v in (sat.items() if isinstance(sat, dict) else [])
            if isinstance(v, dict) and bool(v.get("saturated"))
        }
        rows = [r for r in (getattr(state, "cycle_strategy_log", []) or []) if isinstance(r, dict)]
        cycle = int(getattr(state, "macro_cycle", 0) or 0)
        focus_row = next((r for r in reversed(rows) if int(r.get("cycle", -1) or -1) == cycle), {})
        has_switch = bool(getattr(state, "pending_bottleneck_switch", False))
        if not has_switch and not saturated and not focus_row:
            return ""
        prev = str(getattr(state, "last_cycle_bottleneck", "") or "")
        cur_top = state.current_top_bottleneck()
        direction, pct = self._dominant_roofline_direction()
        lines: list[str] = []
        if has_switch:
            lines.append(
                "The previous macro-cycle plateaued; redirect this cycle to a "
                "different bottleneck instead of re-mining the exhausted one."
            )
        if saturated:
            lines.append("Roofline ceiling signal: one or more lever families are saturated; deprioritize them.")
            for domain, row in sorted(saturated.items()):
                lines.append(
                    f"  saturated_domain={domain} direction={row.get('direction')} "
                    f"within={row.get('within_pct')}% threshold={row.get('threshold_pct')}%"
                )
        if focus_row:
            lines.append(
                f"  suggested_cycle_focus={focus_row.get('focus')} "
                f"score={focus_row.get('score')} rationale={focus_row.get('rationale')}"
            )
        if prev:
            lines.append(f"  plateaued_bottleneck={prev} (avoid re-targeting)")
        if cur_top:
            lines.append(f"  current_top_bottleneck={cur_top}")
        shift = getattr(state, "bottleneck_shift", {}) or {}
        if isinstance(shift, dict) and (shift.get("from") or shift.get("to")):
            lines.append(
                f"  bottleneck_shift: {shift.get('from') or 'unknown'} → {shift.get('to') or 'unknown'} "
                f"(within_delta={shift.get('within_delta')} gap_delta={shift.get('gap_delta')})"
            )
        if direction:
            from ..kernel.roofline_snapshot import BOTTLENECK_DOMAIN_HINTS

            hint = BOTTLENECK_DOMAIN_HINTS.get(direction)
            if hint:
                lines.append(
                    f"  dominant_direction={direction} ({pct:.1f}%) → "
                    f"suggested specialist domain={hint[0]} tag={hint[1]}"
                )
            else:
                lines.append(f"  dominant_direction={direction} ({pct:.1f}%)")
        lines.append(
            f"  macro_cycle={cycle}; KEEP'd variants stay de-duped permanently, "
            "but prior sub-threshold variants whose measured gain now meets the "
            "decayed KEEP bar are unblocked for re-test."
        )
        lines.append("Advisory only: pick the domain/tag yourself; this nudges focus, it does not gate dispatch.")
        return "\n".join(lines)

    def _acceptance_threshold_advisory_block(self) -> str:
        """Render the current decaying acceptance bar + re-testable prior variants.

        Active only in cyclic mode after at least one macro-cycle (when the bar
        has decayed below the first-cycle default). Surfaces the current KEEP /
        stack-stable thresholds and lists prior sub-threshold variants whose
        measured gain now meets the decayed bar (unblocked for re-test) plus a
        few still below it (reference only). Advisory; never gates dispatch.

        Returns:
            The rendered acceptance-threshold advisory text, or ``""`` when not
            applicable (non-cyclic mode or first cycle).
        """
        state = self.shared_state
        keep = self._decaying_keep_threshold_pct()
        if keep is None:
            return ""
        cycle = int(getattr(state, "macro_cycle", 0) or 0)
        if cycle < 1:
            return ""
        stable = keep / 2.0
        unlockable = {"REVERT", "KEEP_UNSTABLE", "no_promote"}
        search = getattr(state, "explore_search", None) or {}
        entries: list[dict[str, Any]] = []
        if isinstance(search, dict):
            tested = search.get("tested") or {}
            if isinstance(tested, dict):
                entries.extend(v for v in tested.values() if isinstance(v, dict))
            rejected = search.get("rejected") or []
            if isinstance(rejected, list):
                entries.extend(v for v in rejected if isinstance(v, dict))
        now_unblocked: list[tuple[str, float]] = []
        still_blocked: list[tuple[str, float]] = []
        for e in entries:
            if str(e.get("outcome") or "") not in unlockable:
                continue
            try:
                g = float(e.get("gain_pct"))
            except (TypeError, ValueError):
                continue
            name = str(e.get("name") or e.get("fingerprint") or "")[:48]
            (now_unblocked if g >= keep else still_blocked).append((name, g))
        lines: list[str] = [
            f"Current acceptance bar (macro_cycle={cycle}): KEEP>={keep:.2f}% stack_stable>={stable:.2f}%.",
            "KEEP'd variants stay de-duped permanently; prior sub-threshold "
            "variants are de-duped only while below the current KEEP bar.",
        ]
        if now_unblocked:
            now_unblocked.sort(key=lambda p: p[1], reverse=True)
            lines.append("Re-testable now (prior gain now clears the bar; re-propose if still relevant):")
            for name, g in now_unblocked[:8]:
                lines.append(f"  {name}: prior gain {g:+.2f}% >= {keep:.2f}%")
        if still_blocked:
            still_blocked.sort(key=lambda p: p[1], reverse=True)
            lines.append("Still below the bar (reference only, not re-tested):")
            for name, g in still_blocked[:5]:
                lines.append(f"  {name}: prior gain {g:+.2f}% < {keep:.2f}%")
        return "\n".join(lines)

    def _target_gap_advisory_block(self) -> str:
        """Build the advisory "External target gap" prompt block (current-best vs competitor target; never gates).

        Returns:
            The rendered external-target-gap advisory text, or ``""`` when
            disabled or no competitor target/current-best is available.
        """
        state = self.shared_state
        if not bool(getattr(state, "target_advisory_enabled", True)):
            return ""
        from ..knowledge import research_hints as _research_hints

        target = _research_hints.load_competitor_target(self.session_dir)
        if not target:
            return ""
        best = getattr(state, "current_best", None)
        if not isinstance(best, dict):
            return ""
        tput = best.get("tput")
        tpot = best.get("tpot_mean_ms")
        tp = int(getattr(state, "tp", 0) or 0)
        our_tput_per_gpu = float(tput) / tp if isinstance(tput, (int, float)) and tput > 0 and tp > 0 else None
        our_tpot_ms = float(tpot) if isinstance(tpot, (int, float)) and tpot > 0 else None
        conc = int(getattr(state, "conc", 0) or 0) or None
        gap = _research_hints.gap_analysis(
            target,
            our_tput_per_gpu=our_tput_per_gpu,
            our_tpot_ms=our_tpot_ms,
            conc=conc,
        )
        return _research_hints.full_gap_summary(gap)

    def _current_primary_gap(self) -> str | None:
        """Resolve the dominant external gap direction ('latency'/'throughput') from the competitor target, or None when advisory is off / no target. Fail-soft.

        Returns:
            The primary gap direction string, or ``None`` when the advisory is
            off, no target exists, or analysis fails.
        """
        state = self.shared_state
        if not bool(getattr(state, "target_advisory_enabled", True)):
            return None
        try:
            from ..knowledge import research_hints as _research_hints

            target = _research_hints.load_competitor_target(self.session_dir)
            if not target:
                return None
            best = getattr(state, "current_best", None)
            if not isinstance(best, dict):
                return None
            tput = best.get("tput")
            tpot = best.get("tpot_mean_ms")
            tp = int(getattr(state, "tp", 0) or 0)
            our_tput_per_gpu = float(tput) / tp if isinstance(tput, (int, float)) and tput > 0 and tp > 0 else None
            our_tpot_ms = float(tpot) if isinstance(tpot, (int, float)) and tpot > 0 else None
            conc = int(getattr(state, "conc", 0) or 0) or None
            gap = _research_hints.gap_analysis(
                target,
                our_tput_per_gpu=our_tput_per_gpu,
                our_tpot_ms=our_tpot_ms,
                conc=conc,
            )
        except Exception:  # noqa: BLE001 — defensive
            return None
        if not isinstance(gap, dict):
            return None
        return str(gap.get("primary_gap") or "").strip() or None

    def _recent_proposed_variants(
        self,
        *,
        max_rounds: int = 2,
    ) -> list[dict[str, Any]]:
        """Collect proposal_set rows from the most recent specialist rounds (deduped by name; fail-soft).

        Args:
            max_rounds: Number of most-recent specialist rounds to scan
                (default 2).

        Returns:
            A name-deduped list of proposal variant dicts.
        """
        rounds = [
            r
            for r in (getattr(self.shared_state, "specialist_rounds", []) or [])
            if isinstance(r, dict) and isinstance(r.get("proposal_set"), list)
        ]
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for r in rounds[-max_rounds:]:
            for variant in r.get("proposal_set") or []:
                if not isinstance(variant, dict):
                    continue
                name = str(variant.get("name") or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    out.append(variant)
        return out

    def _priors_match_advisory_block(self) -> str:
        """Flag recently proposed variants aligning with proven priors / dominant external gap (advisory ordering, fail-soft).

        Returns:
            The rendered priors-match advisory text, or ``""`` when there are no
            recent variants or rendering fails.
        """
        try:
            from ..knowledge import research_hints as _research_hints

            variants = self._recent_proposed_variants()
            if not variants:
                return ""
            hints = _research_hints.load_hints(self.session_dir)
            primary_gap = self._current_primary_gap()
            return _research_hints.priors_match_summary(
                variants,
                hints,
                primary_gap=primary_gap,
            )
        except Exception:  # noqa: BLE001 — defensive
            return ""
