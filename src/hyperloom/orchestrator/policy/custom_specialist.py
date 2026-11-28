# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rule keeping Coordinator-dispatched specialists Coordinator-owned."""

from __future__ import annotations

from typing import Any

from ..specialists.domains import authoring_domain_for_framework, get_domain, is_custom_specialist_dispatch
from .denial import PolicyDenied

__all__ = ["deny_coordinator_dispatched_delegate"]


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
