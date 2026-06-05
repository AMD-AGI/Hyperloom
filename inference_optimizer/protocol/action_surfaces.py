"""Shared action-surface constants.

Keep ownership, transport, and prompt-visibility classifications here so
PolicyGate, prompt rendering, and CLI wiring do not grow separate
action-name lists.
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


# Coordinator-internal actions that intentionally do not appear in
# phase_state.PHASE_ALLOWED_ACTIONS because they bypass LLM proposal /
# delegate policy entirely.
PHASE_ALLOWLIST_BYPASS_ACTIONS: frozenset[str] = frozenset({
    "replay_warm_recipe",
})


# Actions whose prompt catalogue should advertise LLM-supplied grids.
GRID_INJECTABLE_ACTIONS: frozenset[str] = frozenset({
    "explore",
    "sweep",
})


# Actions rendered in the Orchestration prompt for full kernel-enabled runs.
# This is prompt visibility, not phase permission: phase_state and PolicyGate
# still decide when an action is legal on a given tick.
FULL_ENABLED_ACTIONS: tuple[str, ...] = (
    "target_analysis", "baseline",
    "roofline", "deep_kernel_analysis",
    "explore",
    "specialist",
    "dynamic_action",
    "integrate_patch",
    "sweep",
    "kernel_opt", "integrate", "operator_tuning", "vendor_kernel_config",
    "gemm_tuning",
    "report",
    "recover",
)


# Prompt-visible actions for --no-kernel runs. Kernel-owned request actions
# and analysis actions that only feed kernel optimization stay hidden.
NO_KERNEL_ENABLED_ACTIONS: tuple[str, ...] = (
    "target_analysis", "baseline",
    "explore",
    "specialist",
    "dynamic_action",
    "integrate_patch",
    "sweep",
    "report",
    "recover",
)


__all__ = [
    "COORDINATOR_INTERNAL_ACTIONS",
    "FRAMEWORK_PR_INTERNAL_ACTION_NAMES",
    "FULL_ENABLED_ACTIONS",
    "GRID_INJECTABLE_ACTIONS",
    "INTERNAL_ONLY_ACTION_NAMES",
    "KERNEL_OWNED_ACTIONS",
    "NO_KERNEL_ENABLED_ACTIONS",
    "PHASE_ALLOWLIST_BYPASS_ACTIONS",
]
