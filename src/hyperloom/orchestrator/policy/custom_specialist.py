# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ingress rules keeping the operator-defined specialist Coordinator-owned."""

from __future__ import annotations

from typing import Any

from ..specialists.domains import is_custom_specialist_dispatch
from .denial import PolicyDenied

__all__ = ["CUSTOM_ONLY_DENIED_ACTIONS", "deny_custom_specialist_delegate", "deny_search_under_custom_only"]

# What Orchestration may not run in FRAMEWORK_AGENT under ``--custom-specialist-only``.
CUSTOM_ONLY_DENIED_ACTIONS: frozenset[str] = frozenset({"specialist", "explore", "integrate_patch"})
# Mirrors ``machine_state.PHASE_FRAMEWORK_AGENT``: the policy package may not import the phase machine.
_FRAMEWORK_AGENT_PHASE: str = "FRAMEWORK_AGENT"


def deny_custom_specialist_delegate(params: dict[str, Any]) -> None:
    """Refuse a delegate naming ``custom_specialist`` as domain or tag; the Coordinator alone dispatches it."""
    if is_custom_specialist_dispatch(params):
        raise PolicyDenied(
            "delegate{action='specialist'}: custom_specialist is dispatched by the Coordinator, not delegated",
            rule="specialist_custom_coordinator_owned",
            hint="Pick a built-in domain; the operator-defined specialist runs once per macro cycle on its own.",
        )


def deny_search_under_custom_only(shared_state: Any, action_name: str) -> None:
    """Refuse Orchestration's FRAMEWORK_AGENT search under ``--custom-specialist-only``; the Coordinator runs it."""
    if action_name not in CUSTOM_ONLY_DENIED_ACTIONS or not getattr(shared_state, "custom_specialist_only", False):
        return
    if str(shared_state.phase or "").strip().upper() == _FRAMEWORK_AGENT_PHASE:
        raise PolicyDenied(
            f"{action_name!r}: --custom-specialist-only runs FRAMEWORK_AGENT on the custom specialist alone",
            rule="custom_specialist_only",
            hint="Wait for the phase to move on; the Coordinator dispatches and benches the custom specialist itself.",
        )
