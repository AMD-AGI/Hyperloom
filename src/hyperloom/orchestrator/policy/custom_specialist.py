# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rules keeping Coordinator-dispatched specialists Coordinator-owned, and the scope of the custom-only mode."""

from __future__ import annotations

from typing import Any

from ..specialists.domains import authoring_domain_for_framework, get_domain, is_custom_specialist_dispatch
from .denial import PolicyDenied

__all__ = [
    "CUSTOM_ONLY_DENIED_ACTIONS",
    "custom_only_framework",
    "deny_coordinator_dispatched_delegate",
    "deny_search_under_custom_only",
    "skips_reactor_turn",
]

# What Orchestration may not run in FRAMEWORK_AGENT under ``HYPERLOOM_CUSTOM_SPECIALIST_ONLY``.
CUSTOM_ONLY_DENIED_ACTIONS: frozenset[str] = frozenset({"specialist", "explore", "integrate_patch"})
# Mirrors ``machine_state.PHASE_FRAMEWORK_AGENT``: the policy package may not import the phase machine.
_FRAMEWORK_AGENT_PHASE: str = "FRAMEWORK_AGENT"


def deny_coordinator_dispatched_delegate(params: dict[str, Any]) -> None:
    """Refuse a delegate naming a specialist the Coordinator alone dispatches: ``custom_specialist`` as domain or tag, or any domain Orchestration may not select."""
    if is_custom_specialist_dispatch(params):
        raise PolicyDenied(
            "delegate{action='specialist'}: custom_specialist is dispatched by the Coordinator, not delegated",
            rule="specialist_custom_coordinator_owned",
            hint="Pick a built-in domain; the operator-defined specialist runs once per macro cycle on its own.",
        )
    domain = get_domain(str(params.get("domain") or ""))
    if domain is not None and not domain.llm_selectable:
        raise PolicyDenied(
            f"delegate{{action='specialist'}}: {domain.key} is dispatched by the Coordinator, not delegated",
            rule="specialist_domain_coordinator_dispatched",
            hint=f"To author a fix alongside it, dispatch domain={authoring_domain_for_framework(params.get('framework'))!r}.",
        )


def custom_only_framework(shared_state: Any) -> bool:
    """Whether the session is in FRAMEWORK_AGENT in custom-specialist-only mode, where only the custom work runs."""
    if not getattr(shared_state, "custom_specialist_only", False):
        return False
    return str(shared_state.phase or "").strip().upper() == _FRAMEWORK_AGENT_PHASE


def skips_reactor_turn(agent_name: str, shared_state: Any) -> bool:
    """Whether ``agent_name`` sits the tick out: Orchestration under custom-only FRAMEWORK_AGENT, where all it may emit is denied."""
    return agent_name == "orchestration" and custom_only_framework(shared_state)


def deny_search_under_custom_only(shared_state: Any, action_name: str) -> None:
    """Refuse Orchestration's FRAMEWORK_AGENT search in custom-specialist-only mode; the Coordinator runs it."""
    if action_name in CUSTOM_ONLY_DENIED_ACTIONS and custom_only_framework(shared_state):
        raise PolicyDenied(
            f"{action_name!r}: custom-specialist-only mode runs FRAMEWORK_AGENT on the custom specialist alone",
            rule="custom_specialist_only",
            hint="Wait for the phase to move on; the Coordinator dispatches and benches the custom specialist itself.",
        )
