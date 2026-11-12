# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How TraceLens' ``analysis.json`` is located and how its task members are named, for every reader of the report."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom_common.coerce import first_str, to_float
from hyperloom_common.jsonio import read_json

__all__ = [
    "analysis_json_path",
    "fusion_findings",
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


def analysis_json_path(report_path: str | Path) -> Path:
    """The ``analysis.json`` of a TraceLens report: the report itself, or the one rendered beside its ``analysis.md``."""
    path = Path(report_path)
    return path if path.suffix == ".json" else path.with_name("analysis.json")


def fusion_findings(report_path: str | Path) -> list[dict[str, Any]]:
    """The report's ``fusion_optimizations[]`` in TraceLens' order (largest projected saving first).

    Each finding keeps the pattern ``kind``, ``operation``, the projected E2E saving (``impact.mid``, percent) and, per
    member in kernel order, its device symbols and launcher path. TraceLens truncates long symbols to a prefix ending in
    ``...``. A missing report or key reads as no findings.
    """
    report = read_json(analysis_json_path(report_path), default={}, require_dict=True)
    findings: list[dict[str, Any]] = []
    for task in report.get("fusion_optimizations") or []:
        if not isinstance(task, dict):
            continue
        members = [
            {
                "kernel_names": member_kernel_names(member),
                "kernel_launcher_path": first_str(member.get("kernel_launcher_path")),
            }
            for member in task.get("members") or []
            if isinstance(member, dict)
        ]
        impact = task.get("impact") if isinstance(task.get("impact"), dict) else {}
        findings.append(
            {
                "kind": first_str(task.get("kind")),
                "operation": task_operation(task),
                "projected_e2e_pct": to_float(impact.get("mid")),
                "members": members,
            }
        )
    return findings
