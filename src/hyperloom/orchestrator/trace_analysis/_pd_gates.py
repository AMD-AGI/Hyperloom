# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Idle and low-compute gates on each node's own Executive Summary column of a PD multi-trace TraceLens report.

A node whose high idle share is a graph under-recording artifact of its own raw trace is exempt; every candidate is
dropped when any other node fails.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ._analysis_json import load_report_tasks
from ._collective_candidates_inject import _inject_collective_candidates
from ._io_utils import append_log
from ._trace_gates import WorkloadShares, _evaluate_idle_gate_with_graph_guard, _evaluate_low_compute_gate
from .tracelens_skill_runner import TraceLensSkillRunResult

_FAILING_CODES = ("high_gpu_idle_pct", "low_gpu_compute_pct")


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


def node_shares_record(verdicts: list[NodeGateVerdict]) -> dict[str, Any]:
    """The ``pd_node_shares`` finding listing every node's shares, exemption and verdict."""
    return {
        "code": "pd_node_shares",
        "severity": "info",
        "nodes": {
            verdict.node: {**asdict(verdict.shares), "exempt": verdict.exempt, "failing": verdict.failing}
            for verdict in verdicts
        },
    }


def pd_route_candidates(
    skill_result: TraceLensSkillRunResult,
    analysis_json: Path,
    raw_traces: dict[str, Path],
    warnings: list[dict[str, Any]],
    log_path: Path,
) -> list[dict[str, Any]]:
    """Gate each PD node on its own ``analysis.md`` column and read the surviving candidates from ``analysis.json``.

    The node shares and every gate warning are recorded in ``warnings``; a failing node drops every candidate.
    """
    report_path = skill_result.report_path
    verdicts = evaluate_node_gates(node_shares(analysis_json), raw_traces, report_path)
    warnings.append(node_shares_record(verdicts))
    for verdict in verdicts:
        for warning in verdict.warnings:
            warnings.append(warning)
            append_log(log_path, f"PD gate {warning.get('node')}: {warning.get('code')}")
    if any(verdict.failing for verdict in verdicts):
        return []
    kept = load_report_tasks(analysis_json)
    return _inject_collective_candidates(skill_result.output_dir, kept, log_path=log_path, health_warnings=warnings)
