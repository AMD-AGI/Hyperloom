"""Early-stop signal helpers — DESIGN §7.1.

Pure functions over :class:`SharedState` + :class:`Objective` + the
in-flight ``crash_count``. The Conductor's clock task calls
:func:`should_stop_early` every tick.

Five signals (DESIGN §7.1):

    1. target_reached   — objective.is_satisfied(state) returns True
    2. time_exhausted   — time_left_minutes < TIME_BUFFER_MIN
    3. no_more_leverage — every action in the registry would score < 1.0
    4. brier_plateau    — critic's brier_window mean ≥ plateau threshold
    5. emergency        — crash_count >= EMERGENCY_CRASH_THRESHOLD
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from .feature_flags import FeatureFlags
    from .objective import Objective
    from .scheduler import BudgetAwareScheduler
    from .shared_state import SharedState


# Constants ------------------------------------------------------------------
TIME_BUFFER_MIN: float = 5.0
NO_LEVERAGE_THRESHOLD: float = 1.0
BRIER_PLATEAU_THRESHOLD: float = 0.25
EMERGENCY_CRASH_THRESHOLD: int = 2


@dataclass(frozen=True)
class StopSignal:
    name: str
    rationale: str


# ---------------------------------------------------------------------------
def signal_target_reached(
    state: "SharedState", objective: "Objective"
) -> StopSignal | None:
    """DESIGN §7.1 #1 — objective acknowledges goal is hit."""
    if hasattr(objective, "is_satisfied") and objective.is_satisfied(state):
        return StopSignal(
            "target_reached",
            f"objective satisfied at gain={state.cumulative_gain:.2f}%",
        )
    return None


def signal_time_exhausted(state: "SharedState") -> StopSignal | None:
    """DESIGN §7.1 #2 — within ``TIME_BUFFER_MIN`` of MAX_HOURS."""
    if state.max_minutes <= 0:
        return None
    if state.time_left_minutes <= TIME_BUFFER_MIN:
        return StopSignal(
            "time_exhausted",
            f"time_left={state.time_left_minutes:.2f}m ≤ buffer={TIME_BUFFER_MIN}m",
        )
    return None


def signal_no_more_leverage(
    state: "SharedState",
    scheduler: "BudgetAwareScheduler | None",
    *,
    lock_summary: dict[str, Any] | None = None,
    history: Iterable[dict[str, Any]] | None = None,
) -> StopSignal | None:
    """DESIGN §7.1 #3 — every action's score is below 1.0."""
    if scheduler is None or scheduler.actions is None:
        return None
    candidates = list(scheduler.actions.allowed_for_mode(scheduler.mode))
    if not candidates:
        return StopSignal("no_more_leverage", "no actions allowed in this mode")
    scores = [
        scheduler.score(a, state, history=history, lock_summary=lock_summary).score
        for a in candidates
    ]
    if all(s < NO_LEVERAGE_THRESHOLD for s in scores):
        return StopSignal(
            "no_more_leverage",
            f"max action score={max(scores):.3f} < {NO_LEVERAGE_THRESHOLD}",
        )
    return None


def signal_brier_plateau(
    state: "SharedState",
    *,
    flags: "FeatureFlags | None" = None,
    brier_window: list[float] | None = None,
    threshold: float = BRIER_PLATEAU_THRESHOLD,
) -> StopSignal | None:
    """DESIGN §7.1 #4 — critic's prediction quality plateaus.

    Only fires when the Critic / Sage features are enabled (else we have
    no Brier data to read). ``brier_window`` is the running list of the
    last N Brier scores — a *higher* score is *worse*.
    """
    if flags is not None and not getattr(flags, "enable_critic", False):
        return None
    if brier_window is None or len(brier_window) < 5:
        return None
    avg = sum(brier_window) / len(brier_window)
    if avg >= threshold:
        return StopSignal(
            "brier_plateau",
            f"avg brier={avg:.3f} ≥ {threshold} over {len(brier_window)} ticks",
        )
    return None


def signal_emergency(state: "SharedState") -> StopSignal | None:
    """DESIGN §7.1 #5 — repeated crashes."""
    if state.crash_count >= EMERGENCY_CRASH_THRESHOLD:
        return StopSignal(
            "emergency",
            f"crash_count={state.crash_count} ≥ {EMERGENCY_CRASH_THRESHOLD}",
        )
    return None


# ---------------------------------------------------------------------------
def should_stop_early(
    state: "SharedState",
    objective: "Objective",
    *,
    scheduler: "BudgetAwareScheduler | None" = None,
    flags: "FeatureFlags | None" = None,
    brier_window: list[float] | None = None,
    lock_summary: dict[str, Any] | None = None,
    history: Iterable[dict[str, Any]] | None = None,
) -> StopSignal | None:
    """Return the first stop signal that fires (priority order DESIGN §7.1)."""
    for fn in (
        lambda: signal_target_reached(state, objective),
        lambda: signal_emergency(state),
        lambda: signal_time_exhausted(state),
        lambda: signal_brier_plateau(
            state, flags=flags, brier_window=brier_window
        ),
        lambda: signal_no_more_leverage(
            state, scheduler,
            lock_summary=lock_summary, history=history,
        ),
    ):
        sig = fn()
        if sig is not None:
            return sig
    return None


__all__ = [
    "TIME_BUFFER_MIN",
    "NO_LEVERAGE_THRESHOLD",
    "BRIER_PLATEAU_THRESHOLD",
    "EMERGENCY_CRASH_THRESHOLD",
    "StopSignal",
    "signal_target_reached",
    "signal_time_exhausted",
    "signal_no_more_leverage",
    "signal_brier_plateau",
    "signal_emergency",
    "should_stop_early",
]
