# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Idle and low-compute gates on each node of a PD multi-trace TraceLens report.

Each node is gated on its own Executive Summary column, never the Total column. A node whose high idle share is a
graph under-recording artifact of its own raw trace is exempt. Policy ``worst`` (default) drops every candidate when
any non-exempt node fails; ``per_node`` drops only the failing nodes' candidates.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._trace_gates import WorkloadShares, _evaluate_idle_gate_with_graph_guard, _evaluate_low_compute_gate

PD_GATE_POLICY_ENV = "HYPERLOOM_MN_PD_GATE_POLICY"
PD_GATE_POLICY_WORST = "worst"
PD_GATE_POLICY_PER_NODE = "per_node"
_FAILING_CODES = ("high_gpu_idle_pct", "low_gpu_compute_pct")


def pd_gate_policy() -> str:
    """The PD gate policy, ``HYPERLOOM_MN_PD_GATE_POLICY``: ``worst`` (default) or ``per_node``."""
    policy = os.environ.get(PD_GATE_POLICY_ENV, "").strip().lower()
    return PD_GATE_POLICY_PER_NODE if policy == PD_GATE_POLICY_PER_NODE else PD_GATE_POLICY_WORST


def node_shares(analysis_json: Path) -> dict[str, WorkloadShares]:
    """Each node's shares from ``analysis.json`` ``executive_summary.node_metrics``, in column order."""
    try:
        report = json.loads(analysis_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    summary = report.get("executive_summary") if isinstance(report, dict) else None
    metrics = summary.get("node_metrics") if isinstance(summary, dict) else None
    if not isinstance(metrics, dict):
        return {}
    return {
        str(node): WorkloadShares(
            idle_pct=values.get("idle_pct"),
            compute_pct=values.get("compute_pct"),
            exposed_communication_pct=values.get("exposed_communication_pct"),
        )
        for node, values in metrics.items()
        if isinstance(values, dict)
    }


@dataclass(frozen=True)
class NodeGateVerdict:
    """One node's shares and the gate warnings it raised; ``exempt`` when its idle share is under-recorded."""

    node: str
    shares: WorkloadShares
    warnings: list[dict[str, Any]] = field(default_factory=list)
    exempt: bool = False

    @property
    def failing(self) -> bool:
        """Whether a gate fired on this non-exempt node."""
        return not self.exempt and any(w.get("code") in _FAILING_CODES for w in self.warnings)


def evaluate_node_gates(
    shares: dict[str, WorkloadShares], raw_traces: dict[str, Path], report_path: Path
) -> list[NodeGateVerdict]:
    """Gate every node on its own shares; under-recording is checked against the node's own raw trace."""
    verdicts: list[NodeGateVerdict] = []
    for node, node_share in shares.items():
        _, high_idle, under_recorded = _evaluate_idle_gate_with_graph_guard(
            node_share.idle_pct, report_path, raw_traces.get(node)
        )
        if under_recorded is not None:
            verdicts.append(NodeGateVerdict(node, node_share, [{**under_recorded, "node": node}], exempt=True))
            continue
        _, low_compute = _evaluate_low_compute_gate(
            node_share.compute_pct, node_share.exposed_communication_pct, report_path
        )
        warnings = [
            {**warning, "node": node, "message": f"{node}: {warning.get('message', '')}"}
            for warning in (high_idle, low_compute)
            if warning is not None
        ]
        verdicts.append(NodeGateVerdict(node, node_share, warnings))
    return verdicts


def node_shares_record(verdicts: list[NodeGateVerdict], policy: str) -> dict[str, Any]:
    """The ``pd_node_shares`` finding listing every node's shares, exemption and verdict under ``policy``."""
    return {
        "code": "pd_node_shares",
        "severity": "info",
        "policy": policy,
        "nodes": {
            verdict.node: {
                "idle_pct": verdict.shares.idle_pct,
                "compute_pct": verdict.shares.compute_pct,
                "exposed_communication_pct": verdict.shares.exposed_communication_pct,
                "exempt": verdict.exempt,
                "failing": verdict.failing,
            }
            for verdict in verdicts
        },
    }


def gate_pd_candidates(
    candidates: list[dict[str, Any]], verdicts: list[NodeGateVerdict], policy: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Apply ``policy`` to ``candidates``; return the kept candidates, the gate warnings and whether all were dropped.

    ``per_node`` matches a candidate to its node through ``tracelens_node``; a candidate without one is kept.
    """
    warnings = [warning for verdict in verdicts for warning in verdict.warnings]
    failing = {verdict.node for verdict in verdicts if verdict.failing}
    if not failing:
        return candidates, warnings, False
    if policy == PD_GATE_POLICY_PER_NODE:
        kept = [cand for cand in candidates if cand.get("tracelens_node") not in failing]
        return kept, warnings, len(failing) == len(verdicts)
    return [], warnings, True
