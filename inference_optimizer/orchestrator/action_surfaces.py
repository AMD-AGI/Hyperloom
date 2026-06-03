"""Shared action-surface constants.

Keep ownership and transport classifications here so PolicyGate, prompt
rendering, and CLI wiring do not grow separate action-name lists.
"""

from __future__ import annotations


# Actions owned by the Kernel role. Orchestration must use
# request{target_agent="kernel", kind=...}, not delegate/propose_action.
KERNEL_OWNED_ACTIONS: frozenset[str] = frozenset({
    "kernel_opt",
    "integrate",
    "deep_kernel_analysis",
    "operator_tuning",
    "vendor_kernel_config",
    "gemm_tuning",
})


# Coordinator-managed actions that agents should not directly propose.
INTERNAL_ONLY_ACTION_NAMES: frozenset[str] = frozenset({
    "roofline",
    "profile",
    "replay_warm_recipe",
})


# Kept separate because PolicyGate emits a framework-specific denial hint.
FRAMEWORK_PR_INTERNAL_ACTION_NAMES: frozenset[str] = frozenset({
    "framework_pr",
})


COORDINATOR_INTERNAL_ACTIONS: frozenset[str] = (
    INTERNAL_ONLY_ACTION_NAMES | FRAMEWORK_PR_INTERNAL_ACTION_NAMES
)


# Actions whose prompt catalogue should advertise LLM-supplied grids.
GRID_INJECTABLE_ACTIONS: frozenset[str] = frozenset({
    "explore",
    "sweep",
})


__all__ = [
    "COORDINATOR_INTERNAL_ACTIONS",
    "FRAMEWORK_PR_INTERNAL_ACTION_NAMES",
    "GRID_INJECTABLE_ACTIONS",
    "INTERNAL_ONLY_ACTION_NAMES",
    "KERNEL_OWNED_ACTIONS",
]
