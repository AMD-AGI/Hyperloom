# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pod-health signal driven by the robustness-server pod snapshot.

Emits alerts when a ``session_pods`` row has a non-empty ``phase`` other than
``Running``.
"""

from __future__ import annotations

from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity


_POD_RUNNING_PHASES: frozenset[str] = frozenset(
    {
        "Running",
        "Succeeded",
        "Pending",  # transient; we do not flag Pending here
        "",  # missing phase
    }
)


def evaluate_health_signals(
    ctx: ReactorContext,
    data: SourceData,
) -> list[Symptom]:
    """Emit pod-health symptoms from the server pod snapshot.

    Flags pods in a non-running phase (``pod_not_running``).

    Args:
        ctx (ReactorContext): Reactor context; unused, kept for the evaluator
            signature.
        data (SourceData): Collected source data including ``session_pods``.

    Returns:
        list[Symptom]: All pod-health symptoms found this tick, possibly empty.
    """
    del ctx
    out: list[Symptom] = []

    for assignment in data.session_pods:
        if not isinstance(assignment, dict):
            continue
        pod = _pod_dict(assignment)
        phase = _phase(assignment, pod)
        if phase and phase not in _POD_RUNNING_PHASES:
            ns = str(pod.get("namespace") or "")
            name = str(pod.get("name") or "")
            role = str(assignment.get("role") or "")
            severity = SymptomSeverity.HIGH if phase == "Failed" else SymptomSeverity.MEDIUM
            out.append(
                Symptom(
                    name="pod_not_running",
                    severity=severity,
                    summary=f"pod {ns}/{name} ({role}) phase={phase}",
                    evidence={
                        "namespace": ns,
                        "name": name,
                        "role": role,
                        "phase": phase,
                        "assignment_id": assignment.get("assignment_id"),
                    },
                    subject={"namespace": ns, "name": name, "phase": phase},
                    source="server",
                    suggestion=(
                        "kill_task on the related task and escalate strategy"
                        if phase == "Failed"
                        else "escalate_strategy_change to monitor recovery"
                    ),
                )
            )

    return out


def _pod_dict(entry: dict[str, Any]) -> dict[str, Any]:
    """Extract the nested ``pod`` dict from an assignment/summary entry.

    Args:
        entry (dict[str, Any]): A session-pod assignment or summary row.

    Returns:
        dict[str, Any]: The nested ``pod`` dict, or an empty dict when absent.
    """
    pod = entry.get("pod")
    if isinstance(pod, dict):
        return pod
    return {}


def _phase(entry: dict[str, Any], pod: dict[str, Any]) -> str:
    """Resolve a pod's phase from the entry or its nested pod dict.

    Args:
        entry (dict[str, Any]): The assignment/summary row.
        pod (dict[str, Any]): The nested pod dict (see :func:`_pod_dict`).

    Returns:
        str: The trimmed phase string, or an empty string when unset.
    """
    phase = entry.get("phase") or pod.get("phase")
    return str(phase or "").strip()


__all__ = ["evaluate_health_signals"]
