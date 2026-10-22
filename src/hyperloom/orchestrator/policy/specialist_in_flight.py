# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A catalogue domain holds at most one queued or running single-domain specialist."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..specialists.domains import gap_backfilled_domain
from ..specialists.in_flight import held_domain, in_flight_domains
from .denial import PolicyDenied
from .gate import SPECIALIST_ACTION_NAME

__all__ = ["SPECIALIST_DOMAIN_IN_FLIGHT", "specialist_in_flight_denial"]

SPECIALIST_DOMAIN_IN_FLIGHT = "specialist_domain_in_flight"


def specialist_in_flight_denial(
    action_name: str,
    params: Mapping[str, Any] | None,
    tasks: Any,
    find_gap: Callable[[str], Mapping[str, Any] | None],
) -> PolicyDenied | None:
    """Refuse a single-domain specialist for a domain another queued or running one still holds.

    Admission runs before warm-up, so the dispatch is judged with the domain warm-up will store for it.
    """
    if action_name != SPECIALIST_ACTION_NAME:
        return None
    p = params or {}
    domain = held_domain({**p, "domain": gap_backfilled_domain(p, find_gap)})
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
