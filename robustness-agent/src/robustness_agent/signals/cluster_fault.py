# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Symptoms derived from cluster-fault snapshots (M2).

Consumes :data:`SourceData.cluster_faults` rows shaped like the upstream
robust-api ``FaultSummary``::

    {
        "name": "g53-gpu_ecc",
        "monitor_id": "gpu_ecc",
        "node_name": "g53",
        "phase": "Isolating",         # Isolating / Succeeded / Failed
        "auto_repair": false,
        "action": "isolate",
        "created_at": "...",
        "affected_workload_count": 3,
        "affected_gpu_count": 8,
    }

Severity: ``Failed`` -> HIGH; ``Isolating`` -> MEDIUM, escalated to HIGH
on a wide blast radius (>= ``high_workload_threshold`` workloads or
>= ``high_gpu_threshold`` GPUs); ``Succeeded`` -> silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity


@dataclass
class ClusterFaultConfig:
    """Tunables for the cluster_fault rule."""

    # Actionable phases; "Succeeded" is excluded (auto-repair already cleaned up).
    actionable_phases: frozenset[str] = frozenset(
        {"Isolating", "Failed"}
    )
    high_workload_threshold: int = 4
    high_gpu_threshold: int = 8


def evaluate_cluster_fault_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: ClusterFaultConfig | None = None,
) -> list[Symptom]:
    cfg = config or ClusterFaultConfig()
    if not data.cluster_faults:
        return []
    out: list[Symptom] = []
    for entry in data.cluster_faults:
        if not isinstance(entry, dict):
            continue
        sym = _fault_to_symptom(entry, cfg)
        if sym is not None:
            out.append(sym)
    return out


def _fault_to_symptom(
    entry: dict[str, Any],
    cfg: ClusterFaultConfig,
) -> Symptom | None:
    phase = str(entry.get("phase") or "")
    if phase not in cfg.actionable_phases:
        return None

    name = str(entry.get("name") or "")
    monitor_id = str(entry.get("monitor_id") or "")
    node = str(entry.get("node_name") or "")
    affected_workloads = _coerce_int(entry.get("affected_workload_count"))
    affected_gpus = _coerce_int(entry.get("affected_gpu_count"))
    auto_repair = bool(entry.get("auto_repair"))

    severity = _severity_for(
        phase=phase,
        affected_workloads=affected_workloads,
        affected_gpus=affected_gpus,
        cfg=cfg,
    )

    return Symptom(
        name="cluster_fault",
        severity=severity,
        summary=(
            f"cluster fault {name or monitor_id!r} on node {node or 'unknown'} "
            f"phase={phase} workloads={affected_workloads} gpus={affected_gpus}"
        ),
        evidence={
            "fault_name": name,
            "monitor_id": monitor_id,
            "node": node,
            "phase": phase,
            "auto_repair": auto_repair,
            "action": entry.get("action"),
            "affected_workload_count": affected_workloads,
            "affected_gpu_count": affected_gpus,
            "created_at": entry.get("created_at"),
        },
        # Fault name in subject so the ladder cools down per-fault.
        subject={"node": node, "fault": name or monitor_id},
        source="server",
        suggestion=_suggestion(phase, severity, auto_repair),
    )


def _severity_for(
    *,
    phase: str,
    affected_workloads: int,
    affected_gpus: int,
    cfg: ClusterFaultConfig,
) -> SymptomSeverity:
    if phase == "Failed":
        return SymptomSeverity.HIGH
    if affected_workloads >= cfg.high_workload_threshold:
        return SymptomSeverity.HIGH
    if affected_gpus >= cfg.high_gpu_threshold:
        return SymptomSeverity.HIGH
    return SymptomSeverity.MEDIUM


def _suggestion(phase: str, severity: SymptomSeverity, auto_repair: bool) -> str:
    if phase == "Failed":
        return (
            "auto-repair failed; delegate(server_lifecycle) or escalate "
            "strategy to drain affected workloads"
        )
    if severity is SymptomSeverity.HIGH:
        return (
            "blast radius is wide; escalate strategy or pause new "
            "dispatches to the affected node"
        )
    if auto_repair:
        return "auto-repair in progress; observe and re-evaluate next tick"
    return "monitor the fault; alert orchestration if it persists"


def _coerce_int(raw: Any) -> int:
    if isinstance(raw, bool):
        # bool subclasses int; reject so ``True == 1`` doesn't pollute thresholds.
        return 0
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return 0
    return 0


__all__ = ["ClusterFaultConfig", "evaluate_cluster_fault_signals"]
