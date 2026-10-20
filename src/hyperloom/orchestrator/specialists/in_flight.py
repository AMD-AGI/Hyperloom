# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which catalogue domain each queued or running specialist holds.

Orchestration names a fresh gap for each dispatch, so the domain, not the gap, is what identifies repeated work.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .domains import dispatch_domain
from .profile import resolve_specialist_profile

__all__ = ["held_domain", "in_flight_domains"]


def held_domain(params: Mapping[str, Any]) -> str:
    """The catalogue domain a specialist dispatch holds; ``""`` for a freeform one, which no domain bounds."""
    domain = dispatch_domain(dict(params))
    if domain is None or resolve_specialist_profile(dict(params), domain=domain).is_freeform:
        return ""
    return domain.key


def in_flight_domains(tasks: Iterable[Any]) -> dict[str, Any]:
    """The first of ``tasks`` holding each domain, among its specialists."""
    held: dict[str, Any] = {}
    for task in tasks:
        domain = held_domain(task.params or {}) if task.kind == "specialist" else ""
        if domain:
            held.setdefault(domain, task)
    return held
