# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which catalogue domain each queued or running specialist holds.

Orchestration names a fresh gap for each dispatch, so the domain, not the gap, is what identifies repeated work.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .domains import dispatch_domain, is_custom_specialist_dispatch
from .profile import SCOPE_DOMAIN, resolve_specialist_profile

__all__ = ["held_domain", "in_flight_domains"]


def held_domain(params: Mapping[str, Any]) -> str:
    """The catalogue domain a specialist dispatch holds, or ``""`` when it holds none.

    Only a dispatch whose profile scope is ``domain`` holds one. The Coordinator-owned custom specialist holds none
    either: Orchestration cannot delegate it, and its per-macro-cycle idempotency key already bounds it.
    """
    p = dict(params)
    domain = dispatch_domain(p)
    if domain is None or is_custom_specialist_dispatch(p):
        return ""
    return domain.key if resolve_specialist_profile(p, domain=domain).scope == SCOPE_DOMAIN else ""


def in_flight_domains(tasks: Iterable[Any]) -> dict[str, Any]:
    """The first of ``tasks`` holding each domain, among its specialists."""
    held: dict[str, Any] = {}
    for task in tasks:
        domain = held_domain(task.params or {}) if task.kind == "specialist" else ""
        if domain:
            held.setdefault(domain, task)
    return held
