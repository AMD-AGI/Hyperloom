# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's serving TP and configured GPU capacity, and the resource-pool block of the Orchestration prompt."""

from __future__ import annotations

from typing import Any

from ..bus.gpu_pool import SpecialistGpuPool
from ..bus.resource_lock import DEFAULT_LANE_CAPACITIES

__all__ = [
    "gpu_specialist_ceiling",
    "resource_pools_summary",
    "serving_tp_for_policy",
]


def gpu_specialist_ceiling(shared_state: Any) -> int:
    """Configured GPU specialist capacity (0 disables ``needs_gpu=true`` dispatch).

    Args:
        shared_state: SharedState carrying ``gpu_specialist_capacity``.

    Returns:
        int: The configured capacity, 0 when unset or unparseable.
    """
    try:
        return max(0, int(getattr(shared_state, "gpu_specialist_capacity", 0)))
    except (TypeError, ValueError):
        return 0


def serving_tp_for_policy(shared_state: Any) -> int:
    """The live serving process's TP size from ``shared_state.tp``; 0 when unset."""
    return max(0, int(shared_state.tp or 0))


def resource_pools_summary(
    shared_state: Any, *, serving_disjoint_pool: SpecialistGpuPool, whole_machine_pool: SpecialistGpuPool
) -> str:
    """Render the GPU pool / lane capacity block for prompt injection.

    Args:
        shared_state: SharedState carrying TP, the configured capacity and the research-lane capacity.
        serving_disjoint_pool: The pool a non-bench ``needs_gpu`` specialist leases from.
        whole_machine_pool: The pool a bench or enablement specialist leases from.
    """
    lines = [
        f"serving_tp={serving_tp_for_policy(shared_state)}",
        f"gpu_specialist_capacity={gpu_specialist_ceiling(shared_state)}",
        f"serving_disjoint_gpu_pool={serving_disjoint_pool.capacity}"
        "  (non-bench needs_gpu specialists admit against this)",
        f"whole_machine_gpu_pool={whole_machine_pool.capacity}"
        "  (bench / enablement-authoring specialists admit against this)",
        f"research_lane_capacity={max(0, int(shared_state.research_lane_capacity or 0))}  (concurrent CPU specialists)",
        f"gpu_research_lane_capacity={DEFAULT_LANE_CAPACITIES['gpu_research_lane']}"
        "  (mutually exclusive with serving / benchmark / profile)",
    ]
    return "\n".join(lines)
