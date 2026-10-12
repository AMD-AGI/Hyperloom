# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Time-budget helpers: cost estimation and budget gate for session actions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from ..bus.gpu_pool import GPU_LEASE_TTL_GRACE

if TYPE_CHECKING:
    from ..state.shared_state import SharedState

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
    return max(catalogue_min, measured_baseline_sec / 60.0)


def lease_ttl_sec(meta: Any, *, measured_baseline_sec: float = 0.0) -> int:
    """Lease an action for the longer of its catalogue TTL and the cost it is admitted at, plus the GPU lease grace.

    Once this session has measured a baseline round, a GPU action is leased for at least one such round, so the
    remaining lease orchestration reads does not show a healthy round as overdue.
    """
    expected_sec = 60.0 * expected_action_cost_minutes(meta, measured_baseline_sec=measured_baseline_sec)
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
    return usable_sec >= expected_cost_minutes * 60.0
