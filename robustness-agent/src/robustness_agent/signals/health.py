# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pod-health signal driven by robustness-server's session snapshot.

Emits alerts when a ``session_pods`` row has a non-empty ``phase`` other than ``Running``,
or when ``session_summary.pods`` shows empty ``available_metrics`` for a pod older than
``no_metrics_warn_s`` (alive but no telemetry → LOW).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity


@dataclass
class HealthConfig:
    """Tunables for :func:`evaluate_health_signals`.

    Attributes:
        no_metrics_warn_s (float): Minimum pod age in seconds before an empty
            ``available_metrics`` list triggers a ``pod_no_metrics`` symptom.
    """

    no_metrics_warn_s: float = 600.0


_POD_RUNNING_PHASES: frozenset[str] = frozenset({
    "Running",
    "Succeeded",
    "Pending",  # transient; we do not flag Pending here
    "",         # missing phase
})


def evaluate_health_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: HealthConfig | None = None,
) -> list[Symptom]:
    """Emit pod-health symptoms from the server session snapshot.

    Flags pods in a non-running phase (``pod_not_running``) and hands pods that
    have produced no metric series for longer than the configured age
    (``pod_no_metrics``).

    Args:
        ctx (ReactorContext): Reactor context (provides the current unix time).
        data (SourceData): Collected source data including ``session_pods`` and
            ``session_summary``.
        config (HealthConfig | None): Tunables; defaults to :class:`HealthConfig`
            when ``None``.

    Returns:
        list[Symptom]: All pod-health symptoms found this tick, possibly empty.
    """
    cfg = config or HealthConfig()
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
            severity = (
                SymptomSeverity.HIGH if phase == "Failed" else SymptomSeverity.MEDIUM
            )
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
                        "kill_task on the related task and escalate"
                        " strategy" if phase == "Failed" else
                        "escalate_strategy_change to monitor recovery"
                    ),
                )
            )

    summary = data.session_summary
    if isinstance(summary, dict):
        for entry in summary.get("pods") or []:
            if not isinstance(entry, dict):
                continue
            available = entry.get("available_metrics")
            if available not in (None, []):
                continue
            t_start = entry.get("t_start")
            if not isinstance(t_start, (int, float)):
                continue
            age = ctx.now_unix - float(t_start)
            if age < cfg.no_metrics_warn_s:
                continue
            pod = _pod_dict(entry)
            ns = str(pod.get("namespace") or "")
            name = str(pod.get("name") or "")
            role = str(entry.get("role") or "")
            out.append(
                Symptom(
                    name="pod_no_metrics",
                    severity=SymptomSeverity.LOW,
                    summary=(
                        f"pod {ns}/{name} ({role}) has no metric series for "
                        f"{int(age)}s"
                    ),
                    evidence={
                        "namespace": ns,
                        "name": name,
                        "role": role,
                        "age_seconds": int(age),
                    },
                    subject={"namespace": ns, "name": name, "kind": "no_metrics"},
                    source="server",
                    suggestion="verify exporter / cluster_proxy data path",
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


__all__ = ["HealthConfig", "evaluate_health_signals"]
