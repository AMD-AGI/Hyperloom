"""Per-node TraceLens summaries for PD-disaggregated multi-node clusters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyperloom.common.tracelens_report import member_name

# The TraceLens input each PD role's trace yields: a decode engine's trace holds decode steps only, so its steady-state
# chunk is taken with or without the decode-only label; a prefill engine's few steps are analysed whole.
ROLE_STEADY_STATE_MODE = {"prefill": "unsplit", "decode": "decode_only_or_generic"}

_TOP_KERNELS = 5


def node_trace_summaries(
    analysis_json: Path, node_traces: dict[str, str], warnings: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Return ``{node: summary}`` with each node's top kernels in ``analysis.json`` order and its warning codes.

    ``node_traces`` lists the nodes in job order, the first being the focus; a task or warning without a ``node``
    counts toward the focus node. An unreadable report marks every node failed.
    """
    nodes = list(node_traces)
    if not nodes:
        return {}
    try:
        report = json.loads(analysis_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            node: {"status": "failed", "trace": trace, "top": [], "warnings": [], "error": str(exc)[:200]}
            for node, trace in node_traces.items()
        }
    tops: dict[str, list[dict[str, Any]]] = {node: [] for node in nodes}
    tasks = report.get("compute_optimizations") if isinstance(report, dict) else None
    for task in tasks if isinstance(tasks, list) else []:
        if not isinstance(task, dict):
            continue
        top = tops.setdefault(str(task.get("node") or nodes[0]), [])
        for member in task.get("members") or []:
            if isinstance(member, dict) and len(top) < _TOP_KERNELS:
                top.append(
                    {
                        "name": member_name(task, member),
                        "category": str(member.get("category") or ""),
                        "gpu_pct": member.get("pct_e2e"),
                        "bound": str(member.get("bound") or ""),
                    }
                )
    codes: dict[str, list[str]] = {node: [] for node in tops}
    for warning in warnings:
        codes.setdefault(str(warning.get("node") or nodes[0]), []).append(str(warning["code"]))
    return {
        node: {"status": "ok", "trace": node_traces.get(node, ""), "top": tops.get(node, []), "warnings": codes[node]}
        for node in codes
    }
