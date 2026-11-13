# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-role (prefill / decode) TraceLens analysis for PD-disaggregated multi-node clusters."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# The TraceLens input each PD role's trace yields: a decode engine's trace holds decode steps only, so its steady-state
# chunk is taken with or without the decode-only label; a prefill engine's few steps are analysed whole.
ROLE_STEADY_STATE_MODE = {"prefill": "unsplit", "decode": "decode_only_or_generic"}

_TOP_KERNELS = 5


def role_analysis_payload(role: str, trace_path: str, framework: str, workspace_path: str = "") -> dict[str, Any]:
    """Build the non-LLM trace_analyze request for one role's trace; the splitter runs without LLM phase hints."""
    payload: dict[str, Any] = {
        "trace_input": trace_path,
        "framework": framework,
        "analysis_route": "non-llm",
        "steady_state_mode": ROLE_STEADY_STATE_MODE.get(role, "mixed"),
        "split_llm_inference": False,
        "roofline_output_name": f"kernel_roofline_{role}.json",
    }
    if workspace_path:
        payload["workspace_path"] = workspace_path
    return payload


def role_trace_summary(trace_path: str, result: dict[str, Any] | None) -> dict[str, Any]:
    """Compact one role's trace_analyze result into the fields the orchestration prompt renders."""
    result = result if isinstance(result, dict) else {}
    hot = result.get("hot_kernels_top15") or result.get("hot_kernels") or []
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
    workspace_path: str = "",
    known: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run TraceLens (non-LLM) on each role's trace and return ``{role: summary}``.

    Roles already present in ``known`` (role -> raw trace_analyze result) are summarised without re-running.
    """
    from .trace_analyze import trace_analyze_handler

    summaries: dict[str, dict[str, Any]] = {}
    for role, trace_path in role_traces.items():
        if not trace_path:
            continue
        if known and role in known:
            summaries[role] = role_trace_summary(trace_path, known[role])
            continue
        payload = role_analysis_payload(role, trace_path, framework, workspace_path)
        try:
            result = await trace_analyze_handler(payload, session_dir=session_dir)
        except Exception as exc:  # isolation boundary: one role's analysis must not fail the round
            log.warning("pd role trace: %s analysis raised: %r", role, exc, exc_info=True)
            result = {"status": "failed", "error_class": type(exc).__name__, "error": str(exc)}
        summaries[role] = role_trace_summary(trace_path, result)
    return summaries
