"""Deterministic search-mode policy for the forge-loop."""

from __future__ import annotations

import math
from dataclasses import dataclass


SEARCH_MODE_EXPLOIT = "EXPLOIT"
SEARCH_MODE_DIVERSIFY = "DIVERSIFY"
SEARCH_MODES = frozenset({SEARCH_MODE_EXPLOIT, SEARCH_MODE_DIVERSIFY})

OBJECTIVE_CANONICAL_GAIN = "CANONICAL_GAIN"
OBJECTIVE_DISCOVER_NEW_MECHANISM = "DISCOVER_NEW_MECHANISM"

# One empty diff can be a session that honestly found nothing worth changing.
NO_CHANGES_ESCALATION_THRESHOLD = 2

# How many recent iteration outcomes are scanned for that streak.
NO_CHANGES_STREAK_WINDOW = 16

# The smallest total gain a run of exploit iterations can produce and still be worth another one.
MARGINAL_GAIN_FLOOR = 0.05

# How many measured steps that window spans, counted in outcomes rather than iterations so the ones that measured
# nothing do not shorten it.
MARGINAL_GAIN_WINDOW = 6

# How many recent outcomes are scanned to fill that window.
MARGINAL_GAIN_SCAN_WINDOW = 16


# Rounds without a new best before the search changes direction, and before the Supervisor reviews it. A round is a
# whole direction, or several side by side: one that ends without a new best did not pay, and a second in a row is a
# search that needs another view.
DIVERSIFY_AFTER_ROUNDS = 1
SUPERVISE_AFTER_ROUNDS = 2


@dataclass(frozen=True)
class SearchPolicyDecision:
    """One auditable search-mode decision."""

    mode: str
    reason_codes: tuple[str, ...]
    objective_kind: str

    def __post_init__(self) -> None:
        if self.mode not in SEARCH_MODES:
            raise ValueError(f"unsupported search mode: {self.mode}")
        if not self.reason_codes:
            raise ValueError("search policy reason_codes must not be empty")


@dataclass(frozen=True)
class SearchState:
    """Where the search stands: the best's source and the stall streak."""

    best_source: str
    no_improvement_iters: int


@dataclass(frozen=True)
class RecentProgress:
    """What the recent iterations showed: the empty-diff streak and the gain ratio over the exploit window."""

    consecutive_no_changes: int = 0
    window_gain_ratio: float | None = None


class SearchPolicyEngine:
    """Choose EXPLOIT or DIVERSIFY from durable, measured state."""

    def decide(
        self,
        state: SearchState,
        *,
        diversify_after: int,
        recent: RecentProgress = RecentProgress(),
    ) -> SearchPolicyDecision:
        """Return a deterministic mode with stable reason codes.

        ``diversify_after`` is how many rounds without a new best turn the search away from the current line.
        """
        diversify_at = max(1, int(diversify_after))
        empty_diffs = max(0, int(recent.consecutive_no_changes))
        window_gain_ratio = recent.window_gain_ratio
        window_gain = None if window_gain_ratio is None else float(window_gain_ratio)
        if window_gain is not None and not math.isfinite(window_gain):
            raise ValueError(f"window_gain_ratio must be finite: {window_gain_ratio!r}")

        # Outranks every other signal: those weigh how promising the current direction is, while repeated empty
        # diffs are evidence it cannot be turned into a candidate at all, so staying in EXPLOIT spends another
        # session on a direction that produces no edit.
        if empty_diffs >= NO_CHANGES_ESCALATION_THRESHOLD:
            mode = SEARCH_MODE_DIVERSIFY
            reason = "REPEATED_NO_CHANGES"
        elif state.no_improvement_iters >= diversify_at:
            mode = SEARCH_MODE_DIVERSIFY
            reason = "NO_IMPROVEMENT_STALL"
        elif window_gain is not None and window_gain < MARGINAL_GAIN_FLOOR:
            mode = SEARCH_MODE_DIVERSIFY
            reason = "DIMINISHING_RETURNS"
        elif (state.best_source or "").strip().lower() == "warm_start":
            mode = SEARCH_MODE_EXPLOIT
            reason = "KB_WARM_START_EXPLOIT"
        else:
            mode = SEARCH_MODE_EXPLOIT
            reason = "CANONICAL_GAIN_AVAILABLE"
        objective = OBJECTIVE_DISCOVER_NEW_MECHANISM if mode == SEARCH_MODE_DIVERSIFY else OBJECTIVE_CANONICAL_GAIN
        return SearchPolicyDecision(mode=mode, reason_codes=(reason,), objective_kind=objective)
