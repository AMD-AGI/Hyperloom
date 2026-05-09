"""Pod-health signal driven by robustness-server's session snapshot.

The current session may include both brain (shared) and hands
(session-bound) pods.  ``session_pods`` rows shaped by
``robustness-server`` carry ``pod.namespace`` / ``pod.name`` /
``role`` / ``t_start`` / ``t_end`` and (when summary is fetched)
``available_metrics``.

For M1 we emit medium-severity alerts when:

* a pod's record carries a non-empty ``phase`` other than ``Running``
  (data shape varies; we accept ``phase`` at the top level or under
  ``pod.phase``).
* ``session_summary.pods`` reports an empty ``available_metrics`` list
  for a hands pod that started more than ``no_metrics_warn_s`` seconds
  ago (likely Pod is alive but no telemetry — surface low severity).

GPU thermal / utilisation symptoms move into M2 once the cluster
proxy endpoints land.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity


@dataclass
class HealthConfig:
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
    pod = entry.get("pod")
    if isinstance(pod, dict):
        return pod
    return {}


def _phase(entry: dict[str, Any], pod: dict[str, Any]) -> str:
    phase = entry.get("phase") or pod.get("phase")
    return str(phase or "").strip()


__all__ = ["HealthConfig", "evaluate_health_signals"]
