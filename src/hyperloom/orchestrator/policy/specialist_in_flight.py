# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A catalogue domain holds at most one queued or running single-domain specialist, and a module group one."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..specialists.domains import gap_backfilled_domain
from ..specialists.in_flight import held_domain, held_module_group, in_flight_domains, in_flight_module_groups
from .denial import PolicyDenied
from .gate import SPECIALIST_ACTION_NAME

__all__ = ["SPECIALIST_DOMAIN_IN_FLIGHT", "SPECIALIST_MODULE_IN_FLIGHT", "specialist_in_flight_denial"]

SPECIALIST_DOMAIN_IN_FLIGHT = "specialist_domain_in_flight"
SPECIALIST_MODULE_IN_FLIGHT = "specialist_module_in_flight"


def specialist_in_flight_denial(
    action_name: str,
    params: Mapping[str, Any] | None,
    tasks: Any,
    find_gap: Callable[[str], Mapping[str, Any] | None],
) -> PolicyDenied | None:
    """Refuse a specialist for a domain or a module group another queued or running one still holds.

    Admission runs before warm-up, so the dispatch is judged with the domain warm-up will store for it. A dispatch
    naming no known module group is judged on its domain alone.
    """
    if action_name != SPECIALIST_ACTION_NAME:
        return None
    p = params or {}
    domain = held_domain({**p, "domain": gap_backfilled_domain(p, find_gap)})
    group = held_module_group(p)
    if not domain and not group:
        return None
    live = [*tasks.queued_context_sync(), *tasks.running_context_sync()]
    holder = in_flight_domains(live).get(domain) if domain else None
    if holder is not None:
        return PolicyDenied(
            f"delegate{{action='specialist'}}: {domain} already has specialist task {holder.task_id} {holder.state}",
            rule=SPECIALIST_DOMAIN_IN_FLIGHT,
            hint=f"Wait for task {holder.task_id}'s result and fold new evidence into the next {domain} dispatch.",
        )
    holder = in_flight_module_groups(live).get(group) if group else None
    if holder is None:
        return None
    return PolicyDenied(
        f"delegate{{action='specialist'}}: module group {group} already has specialist task "
        f"{holder.task_id} {holder.state}",
        rule=SPECIALIST_MODULE_IN_FLIGHT,
        hint=f"Wait for task {holder.task_id}'s result, or give this dispatch another module group.",
    )
