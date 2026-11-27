# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-role (prefill / decode) TraceLens analysis for PD-disaggregated multi-node clusters."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hyperloom.orchestrator.trace_analysis import DECODE_ONLY_OR_GENERIC_MODE, UNSPLIT_MODE
from hyperloom.orchestrator.trace_analysis.analysis_route import AnalysisRoute

from .trace_analyze import PD_ROLE_TRACE_KEY

log = logging.getLogger(__name__)

# The TraceLens input each PD role's trace yields: a decode engine's trace holds decode steps only, so its steady-state
# chunk is taken with or without the decode-only label; a prefill engine's few steps are analysed whole.
ROLE_STEADY_STATE_MODE = {"prefill": UNSPLIT_MODE, "decode": DECODE_ONLY_OR_GENERIC_MODE}

_TOP_KERNELS = 5


def hot_kernels(ta_result: dict[str, Any]) -> list[Any]:
    """The hot kernels an analysis surfaced."""
    return ta_result.get("hot_kernels_top15") or ta_result.get("hot_kernels") or []


def role_analysis_payload(role: str, trace_path: str, framework: str) -> dict[str, Any]:
    """Build the non-LLM trace_analyze request for one role's trace."""
    return {
        "trace_input": trace_path,
        "framework": framework,
        "analysis_route": AnalysisRoute.NON_LLM.value,
        "steady_state_mode": ROLE_STEADY_STATE_MODE[role],
        "roofline_output_name": f"kernel_roofline_{role}.json",
        PD_ROLE_TRACE_KEY: True,
    }


def _role_trace_summary(trace_path: str, result: dict[str, Any]) -> dict[str, Any]:
    """Compact one role's trace_analyze result into the fields the orchestration prompt renders."""
    hot = hot_kernels(result)
    top = [
        {
            "name": str(entry.get("name") or entry.get("kernel_id") or "?"),
            "category": str(entry.get("kernel_category") or ""),
            "gpu_pct": entry.get("gpu_pct"),
            "bound": str(entry.get("bound_type") or entry.get("bottleneck") or ""),
        }
        for entry in (hot if isinstance(hot, list) else [])[:_TOP_KERNELS]
        if isinstance(entry, dict)
    ]
    warnings = [
        str(w.get("code")) for w in (result.get("trace_health_warnings") or []) if isinstance(w, dict) and w.get("code")
    ]
    summary: dict[str, Any] = {
        "status": "ok" if result.get("status") == "ok" else "failed",
        "trace": trace_path,
        "top": top,
        "warnings": warnings,
    }
    if summary["status"] != "ok":
        summary["error"] = str(result.get("error_class") or result.get("error") or "trace_analyze failed")[:200]
    return summary


async def analyze_role_traces(
    role_traces: dict[str, str],
    *,
    session_dir: Path,
    framework: str,
    known: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Run TraceLens (non-LLM) on each role's trace and return ``{role: summary}``.

    Roles already present in ``known`` (role -> raw trace_analyze result) are summarised without re-running.
    """
    from .trace_analyze import trace_analyze_handler

    summaries: dict[str, dict[str, Any]] = {}
    for role, trace_path in role_traces.items():
        if not trace_path:
            continue
        if role in known:
            summaries[role] = _role_trace_summary(trace_path, known[role])
            continue
        payload = role_analysis_payload(role, trace_path, framework)
        try:
            result = await trace_analyze_handler(payload, session_dir=session_dir)
        except Exception as exc:  # isolation boundary: one role's analysis must not fail the round
            log.warning("pd role trace: %s analysis raised: %r", role, exc, exc_info=True)
            result = {"status": "failed", "error_class": type(exc).__name__, "error": str(exc)}
        summaries[role] = _role_trace_summary(trace_path, result)
    return summaries
