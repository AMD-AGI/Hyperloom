# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a ``compute_optimizations[]`` member of TraceLens' ``analysis.json`` is named, for every reader of the report."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyperloom_common.coerce import first_str

__all__ = [
    "member_kernel_names",
    "member_name",
    "task_operation",
]


def task_operation(task: Mapping[str, Any]) -> str:
    """The task's ``operation``, empty when TraceLens wrote none."""
    return first_str(task.get("operation")).strip()


def member_kernel_names(member: Mapping[str, Any]) -> list[str]:
    """The member's device symbols in TraceLens' order, blank ones dropped."""
    return [str(n) for n in (member.get("kernel_name") or []) if str(n).strip()]


def member_name(task: Mapping[str, Any], member: Mapping[str, Any]) -> str:
    """The task's ``operation``, or, for a graph-collapsed member whose task names none, its first device symbol: the
    identity TraceLens' own deterministic-fallback grouping keys on."""
    kernel_names = member_kernel_names(member)
    return task_operation(task) or (kernel_names[0] if kernel_names else "")
