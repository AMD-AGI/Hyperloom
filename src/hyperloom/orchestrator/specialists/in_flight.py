# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which catalogue domain and which decode module group each queued or running specialist holds.

Orchestration names a fresh gap for each dispatch, so the domain, not the gap, is what identifies repeated work.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from kernelforge.decode_modules import MODULE_RULES

from ..measurement.cycle_bench import cycle_bench_enabled
from .domains import dispatch_domain, is_custom_specialist_dispatch
from .profile import SCOPE_DOMAIN, resolve_specialist_profile

__all__ = ["MODULE_GROUPS", "held_domain", "held_module_group", "in_flight_domains", "in_flight_module_groups"]

#: The decode module groups a specialist may own, named as the module table names them.
MODULE_GROUPS: tuple[str, ...] = tuple(module for module, _ in MODULE_RULES)


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


def held_module_group(params: Mapping[str, Any]) -> str:
    """The module group a dispatch's ``module_group`` names, whatever its scope; ``""`` when it names none.

    Only with the cycle bench switched on, and never for the Coordinator-owned custom specialist.
    """
    group = str(params.get("module_group") or "").strip().lower()
    if not cycle_bench_enabled() or group not in MODULE_GROUPS or is_custom_specialist_dispatch(dict(params)):
        return ""
    return group


def in_flight_domains(tasks: Iterable[Any]) -> dict[str, Any]:
    """The first of ``tasks`` holding each domain, among its specialists."""
    return _first_holders(tasks, held_domain)


def in_flight_module_groups(tasks: Iterable[Any]) -> dict[str, Any]:
    """The first of ``tasks`` holding each module group, among its specialists."""
    return _first_holders(tasks, held_module_group)


def _first_holders(tasks: Iterable[Any], held: Callable[[Mapping[str, Any]], str]) -> dict[str, Any]:
    holders: dict[str, Any] = {}
    for task in tasks:
        key = held(task.params or {}) if task.kind == "specialist" else ""
        if key:
            holders.setdefault(key, task)
    return holders
