# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Time-budget helpers: cost estimation and the session and phase budget gates for actions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from hyperloom.common.deadline import Deadline

from ..bus.gpu_pool import GPU_LEASE_TTL_GRACE
from ..phases import machine_state as _phase_state
from ..policy.gate import PolicyDenied
from ..specialists.profile import resolve_specialist_profile, wall_budget_base_min

if TYPE_CHECKING:
    from ..state.shared_state import SharedState

# The phases whose own wall-clock budget bounds the search work started in them; every other phase answers only to
# the session budget.
PHASE_BUDGET_GATED_PHASES: frozenset[str] = frozenset({"FRAMEWORK_AGENT", "KERNEL_AGENT"})

# What a bench-capable specialist spends before its first benchmark round can start.
_BENCH_SPECIALIST_STARTUP_MIN = 10.0

# Actions that must stay startable no matter how little budget is left: they are how a session ends cleanly, so a time
# gate that refused them would strand the run with nothing to show.
TIME_BUDGET_EXEMPT_ACTIONS: frozenset[str] = frozenset(
    {
        "report",
        "session_breakdown",
    }
)

# The lanes that serialize GPU work: an action requiring one of them spends its time running a benchmark round, so
# what this session measured says more about it than a catalogue estimate does.
_GPU_BENCH_LANES: frozenset[str] = frozenset(
    {
        "benchmark_lane",
        "profile_lane",
    }
)

_SECONDS_PER_MINUTE = 60.0


def measured_baseline_runtime_sec(shared_state: SharedState) -> float:
    """Read this session's own measured baseline round, in seconds."""
    return max(0.0, shared_state.baseline_runtime_sec)


def _action_benches_on_gpu(meta: Any | None) -> bool:
    """Whether an action's cost is dominated by a benchmark round on the GPU."""
    lanes = getattr(meta, "requires_lanes", ()) or ()
    try:
        return any(str(lane) in _GPU_BENCH_LANES for lane in lanes)
    except TypeError:
        return False


def expected_action_cost_minutes(
    meta: Any | None,
    *,
    measured_baseline_sec: float = 0.0,
) -> float:
    """Read an action's expected cost, preferring what this session measured."""
    try:
        catalogue_min = float(getattr(meta, "typical_runtime_min", 0.0) or 0.0)
    except (TypeError, ValueError):
        catalogue_min = 0.0
    if measured_baseline_sec <= 0.0 or not _action_benches_on_gpu(meta):
        return catalogue_min
    if meta.admission_rounds > 0:
        return meta.admission_rounds * measured_baseline_sec / _SECONDS_PER_MINUTE
    return max(catalogue_min, measured_baseline_sec / _SECONDS_PER_MINUTE)


def lease_ttl_sec(meta: Any, *, measured_baseline_sec: float = 0.0) -> int:
    """Lease an action for the longer of its catalogue TTL and the cost it is admitted at, plus the GPU lease grace.

    Once this session has measured a baseline round, a GPU action is leased for at least one such round, so the
    remaining lease orchestration reads does not show a healthy round as overdue.
    """
    expected_sec = _SECONDS_PER_MINUTE * expected_action_cost_minutes(meta, measured_baseline_sec=measured_baseline_sec)
    return max(int(meta.lease_ttl_sec), math.ceil(expected_sec * (1.0 + GPU_LEASE_TTL_GRACE)))


def action_fits_time_budget(
    *,
    usable_sec: float | None,
    expected_cost_minutes: float,
) -> bool:
    """Decide whether an action's expected cost still fits the remaining budget."""
    if usable_sec is None:
        return True
    if expected_cost_minutes <= 0.0:
        return True
    return usable_sec >= expected_cost_minutes * _SECONDS_PER_MINUTE


def gated_phase_remaining_sec(shared_state: SharedState) -> float | None:
    """Seconds left in the current phase's budget when that budget bounds the phase's search, else ``None``."""
    if str(shared_state.phase or "").upper() not in PHASE_BUDGET_GATED_PHASES:
        return None
    return _phase_state.phase_budget_remaining_seconds(shared_state)


def phase_deadline(shared_state: SharedState) -> Deadline | None:
    """The instant the current phase's budget runs out, or ``None`` when no phase budget bounds its search."""
    remaining = gated_phase_remaining_sec(shared_state)
    return None if remaining is None else Deadline.after(remaining)


def specialist_admission_minutes(shared_state: SharedState, params: dict[str, Any] | None) -> float:
    """The shortest run worth starting a specialist for.

    That is its mode's base budget and, for one that benches, room for one measured benchmark round after startup.
    Admission asks only for this minimum; a dispatched specialist runs to its full wall budget, tightened to the
    session and phase ends.
    """
    floor_min = wall_budget_base_min(params)
    if resolve_specialist_profile(params).reserves_benchmark_lane:
        round_min = measured_baseline_runtime_sec(shared_state) / _SECONDS_PER_MINUTE
        floor_min = max(floor_min, round_min + _BENCH_SPECIALIST_STARTUP_MIN)
    return floor_min


def time_budget_denial(action: str, expected_cost_minutes: float, shared_state: SharedState) -> PolicyDenied | None:
    """Refuse an action that the session's usable time, or the gated phase's remaining time, cannot fit.

    Args:
        action: The action being admitted.
        expected_cost_minutes: What the action is expected to cost.
        shared_state: The live session state both budgets are read from.

    Returns:
        A :class:`PolicyDenied` naming the budget that cannot fit the action, else ``None``.
    """
    session_sec = shared_state.session_budget_usable_sec()
    if not action_fits_time_budget(usable_sec=session_sec, expected_cost_minutes=expected_cost_minutes):
        return PolicyDenied(
            _cost_exceeds(action, expected_cost_minutes, session_sec, "the session budget"),
            rule="time_budget",
            hint=(
                "the wall-clock budget cannot fit this action; pick an action that fits "
                "the time left, the Coordinator closes the session once none does"
            ),
        )
    phase_sec = gated_phase_remaining_sec(shared_state)
    if not action_fits_time_budget(usable_sec=phase_sec, expected_cost_minutes=expected_cost_minutes):
        return PolicyDenied(
            _cost_exceeds(action, expected_cost_minutes, phase_sec, f"the {shared_state.phase} phase budget"),
            rule="phase_time_budget",
            hint="the phase's wall-clock budget cannot fit this action; pick one that fits the time left in the phase",
        )
    return None


def _cost_exceeds(action: str, expected_cost_minutes: float, usable_sec: float | None, budget: str) -> str:
    """The denial reason: what the action needs against what ``budget`` has left."""
    remaining_min = (usable_sec or 0.0) / _SECONDS_PER_MINUTE
    return (
        f"action={action!r} denied: needs ~{expected_cost_minutes:.0f} min but only "
        f"{remaining_min:.0f} min of {budget} is left"
    )
