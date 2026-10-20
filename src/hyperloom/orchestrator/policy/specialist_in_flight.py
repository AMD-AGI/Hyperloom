# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A catalogue domain holds at most one queued or running specialist."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..specialists.in_flight import held_domain, in_flight_domains
from .denial import PolicyDenied
from .gate import SPECIALIST_ACTION_NAME

__all__ = ["SPECIALIST_DOMAIN_IN_FLIGHT", "specialist_in_flight_denial"]

SPECIALIST_DOMAIN_IN_FLIGHT = "specialist_domain_in_flight"


def specialist_in_flight_denial(action_name: str, params: Mapping[str, Any] | None, tasks: Any) -> PolicyDenied | None:
    """Refuse a specialist for a domain another queued or running specialist still holds."""
    domain = held_domain(params or {}) if action_name == SPECIALIST_ACTION_NAME else ""
    if not domain:
        return None
    holder = in_flight_domains([*tasks.queued_context_sync(), *tasks.running_context_sync()]).get(domain)
    if holder is None:
        return None
    return PolicyDenied(
        f"delegate{{action='specialist'}}: {domain} already has specialist task {holder.task_id} {holder.state}",
        rule=SPECIALIST_DOMAIN_IN_FLIGHT,
        hint=f"Wait for task {holder.task_id}'s result and fold new evidence into the next {domain} dispatch.",
    )
