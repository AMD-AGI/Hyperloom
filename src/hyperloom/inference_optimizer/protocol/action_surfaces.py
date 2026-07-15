# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared action-surface constants.

Keep ownership, transport, and prompt-visibility classifications here so
PolicyGate, prompt rendering, and CLI wiring do not grow separate
action-name lists.
"""

from __future__ import annotations


# Actions owned by the Kernel role; requested via request{target_agent="kernel_agent"}.
KERNEL_AGENT_OWNED_ACTIONS: frozenset[str] = frozenset(
    {
        "kernel_opt",
        "integrate",
        "deep_kernel_analysis",
        "operator_tuning",
        "vendor_kernel_config",
        "gemm_tuning",
    }
)


# Coordinator-managed actions that agents should not directly propose.
INTERNAL_ONLY_ACTION_NAMES: frozenset[str] = frozenset(
    {
        "roofline",
        "profile",
        "replay_warm_recipe",
    }
)


# Kept separate because PolicyGate emits a framework-specific denial hint.
FRAMEWORK_AGENT_INTERNAL_ACTION_NAMES: frozenset[str] = frozenset(
    {
        "framework_agent",
    }
)


COORDINATOR_INTERNAL_ACTIONS: frozenset[str] = INTERNAL_ONLY_ACTION_NAMES | FRAMEWORK_AGENT_INTERNAL_ACTION_NAMES


# Robustness-only actions (driven via its action-ladder); Orchestration must
# ALERT instead. ``recover`` walks SIGTERM/SIGKILL against server owners.
ROBUSTNESS_DELEGATE_ONLY_ACTIONS: frozenset[str] = frozenset(
    {
        "recover",
    }
)


# Actions whose prompt catalogue should advertise LLM-supplied grids.
GRID_INJECTABLE_ACTIONS: frozenset[str] = frozenset(
    {
        "explore",
        "sweep",
    }
)


# Actions rendered in the Orchestration prompt for full kernel-enabled runs.
# Prompt visibility only; phase_state and PolicyGate decide legality per tick.
FULL_ENABLED_ACTIONS: tuple[str, ...] = (
    "target_analysis",
    "baseline",
    "roofline",
    "deep_kernel_analysis",
    "explore",
    "specialist",
    "integrate_patch",
    "sweep",
    "kernel_opt",
    "integrate",
    "operator_tuning",
    "vendor_kernel_config",
    "gemm_tuning",
    "report",
)


# Prompt-visible actions for --no-kernel runs. Kernel-owned request actions
# and analysis actions that only feed kernel optimization stay hidden.
NO_KERNEL_AGENT_ENABLED_ACTIONS: tuple[str, ...] = (
    "target_analysis",
    "baseline",
    "explore",
    "specialist",
    "integrate_patch",
    "sweep",
    "report",
)


__all__ = [
    "COORDINATOR_INTERNAL_ACTIONS",
    "FRAMEWORK_AGENT_INTERNAL_ACTION_NAMES",
    "FULL_ENABLED_ACTIONS",
    "GRID_INJECTABLE_ACTIONS",
    "INTERNAL_ONLY_ACTION_NAMES",
    "KERNEL_AGENT_OWNED_ACTIONS",
    "NO_KERNEL_AGENT_ENABLED_ACTIONS",
    "ROBUSTNESS_DELEGATE_ONLY_ACTIONS",
]
