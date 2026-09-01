"""Deterministic search-mode policy for forge-loop planning."""

from __future__ import annotations

import math
from dataclasses import dataclass


SEARCH_MODE_EXPLOIT = "EXPLOIT"
SEARCH_MODE_DIVERSIFY = "DIVERSIFY"
SEARCH_MODES = frozenset({SEARCH_MODE_EXPLOIT, SEARCH_MODE_DIVERSIFY})

OBJECTIVE_IMMEDIATE_CANONICAL_GAIN = "IMMEDIATE_CANONICAL_GAIN"
OBJECTIVE_DISCOVER_NEW_MECHANISM = "DISCOVER_NEW_MECHANISM"

# One empty diff can be a session that honestly found nothing worth changing.
# Two in a row mean the Implementer cannot express the current direction as an
# edit at all, which is a different fact from the no-improvement streak: nothing
# was measured, so no amount of further exploitation can resolve it.
NO_CHANGES_ESCALATION_THRESHOLD = 2

# How many recent iteration outcomes are scanned for that streak. Counted in
# outcomes rather than in raw log events: an iteration writes several events, so
# an event-counted window is spent by a handful of interleaved infrastructure
# failures and hides the streak it exists to find.
NO_CHANGES_STREAK_WINDOW = 16

# The smallest total gain a run of exploit iterations can produce and still be
# worth another one. Stated over the window below rather than per iteration,
# because one small step is ordinary and only a run of them is a trend: a
# campaign whose whole recent ladder moved the incumbent by less than this is
# refining a direction whose remaining steps are too small to reach what a
# different mechanism might, and zero returns are not the only reason to look
# elsewhere.
MARGINAL_GAIN_FLOOR = 0.05

# How many measured steps that window spans, counted in outcomes rather than
# iterations so the ones that measured nothing do not shorten it. Six steps need
# seven outcomes: the oldest is the anchor the gain is measured against, not a
# step of its own. Six is wide enough that a campaign still climbing at better
# than roughly one percent an iteration keeps its ladder, and long enough that
# filling the window is itself the cost ceiling on the trigger.
MARGINAL_GAIN_WINDOW = 6

# How many recent outcomes are scanned to fill that window. Longer than the
# window itself because outcomes that concluded nothing are transparent to it,
# for the same reason as the empty-diff scan above.
MARGINAL_GAIN_SCAN_WINDOW = 16


@dataclass(frozen=True)
class SearchPolicyDecision:
    """One auditable search-mode decision."""

    mode: str
    reason_codes: tuple[str, ...]
    objective_kind: str
    residence_iterations_remaining: int = 0

    def __post_init__(self) -> None:
        if self.mode not in SEARCH_MODES:
            raise ValueError(f"unsupported search mode: {self.mode}")
        if not self.reason_codes:
            raise ValueError("search policy reason_codes must not be empty")


class SearchPolicyEngine:
    """Choose EXPLOIT or DIVERSIFY from durable, measured state."""

    def decide(
        self,
        *,
        best_source: str,
        no_improvement_iters: int,
        stall_threshold: int,
        current_mode: str = SEARCH_MODE_EXPLOIT,
        residence_iterations_remaining: int = 0,
        diversification_cycle_completed: bool = False,
        consecutive_no_changes: int = 0,
        window_gain_ratio: float | None = None,
    ) -> SearchPolicyDecision:
        """Return a deterministic mode with stable reason codes.

        ``window_gain_ratio`` is the relative gain the incumbent made across the
        last full window of exploit outcomes, or ``None`` when the campaign has
        not produced a full window yet. ``None`` is not a gain of zero: a
        campaign that has not been measured enough times to have a trend is not
        a campaign whose ladder has flattened, and only the second of those is a
        reason to look for another mechanism.
        """
        threshold = max(1, int(stall_threshold))
        residence = max(0, int(residence_iterations_remaining))
        empty_diffs = max(0, int(consecutive_no_changes))
        window_gain = None if window_gain_ratio is None else float(window_gain_ratio)
        if window_gain is not None and not math.isfinite(window_gain):
            raise ValueError(f"window_gain_ratio must be finite: {window_gain_ratio!r}")

        # Outranks every other signal, mode residence included: those weigh how
        # promising the current direction is, while repeated empty diffs are
        # evidence it cannot be turned into a candidate at all, so staying in
        # EXPLOIT spends another session on a direction that produces no edit.
        if empty_diffs >= NO_CHANGES_ESCALATION_THRESHOLD:
            mode = SEARCH_MODE_DIVERSIFY
            reason = "REPEATED_NO_CHANGES"
        elif current_mode == SEARCH_MODE_EXPLOIT and residence > 0:
            mode = SEARCH_MODE_EXPLOIT
            reason = "MODE_RESIDENCE"
        elif diversification_cycle_completed:
            mode = SEARCH_MODE_EXPLOIT
            reason = "DIVERSIFY_PLAN_CREATED"
        elif no_improvement_iters >= threshold:
            mode = SEARCH_MODE_DIVERSIFY
            reason = "NO_IMPROVEMENT_STALL"
        elif window_gain is not None and window_gain < MARGINAL_GAIN_FLOOR:
            mode = SEARCH_MODE_DIVERSIFY
            reason = "DIMINISHING_RETURNS"
        elif (best_source or "").strip().lower() == "warm_start":
            mode = SEARCH_MODE_EXPLOIT
            reason = "KB_WARM_START_EXPLOIT"
        else:
            mode = SEARCH_MODE_EXPLOIT
            reason = "CANONICAL_GAIN_AVAILABLE"

        if mode == SEARCH_MODE_DIVERSIFY:
            return SearchPolicyDecision(
                mode=mode,
                reason_codes=(reason,),
                objective_kind=OBJECTIVE_DISCOVER_NEW_MECHANISM,
                residence_iterations_remaining=0,
            )

        if reason == "DIVERSIFY_PLAN_CREATED":
            next_residence = max(0, threshold - 1)
        elif reason == "MODE_RESIDENCE":
            next_residence = residence - 1
        else:
            next_residence = 0
        return SearchPolicyDecision(
            mode=mode,
            reason_codes=(reason,),
            objective_kind=OBJECTIVE_IMMEDIATE_CANONICAL_GAIN,
            residence_iterations_remaining=next_residence,
        )
