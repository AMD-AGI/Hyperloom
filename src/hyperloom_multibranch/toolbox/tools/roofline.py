# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``roofline`` tool (the kernel roofline of the trace a profile test left) and the ``tracelens`` tool (analysis of a
given trace).

Both read a trace that already exists and need no card: ``roofline`` runs the orchestrator's ``non-llm`` analysis, which reads
the trace in memory and writes ``kernel_roofline.json`` (every hot kernel with its category, share of the GPU time and bound),
with no model session; ``tracelens`` runs TraceLens' report. The roofline of the orchestrator's own flow profiles first, on a
card, in the process that runs it; a toolbox does not offer that.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.framework_registry import is_scriptable
from hyperloom.orchestrator.actions.executors.trace_analyze import trace_analyze_handler
from hyperloom.orchestrator.trace_analysis.analysis_route import AnalysisRoute
from hyperloom_multibranch.toolbox.tools.base import ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools.executor_tool import ScratchHandlerTool
from hyperloom_multibranch.toolbox.tools.op_shares import with_op_shares
from hyperloom_multibranch.toolbox.tools.tracelens_report import (
    DEFAULT_PLATFORM,
    REPORT_ROUTE,
    report_handler,
    unusable_trace,
    with_window_shares,
)

ROOFLINE_TOOL = "roofline"
TRACELENS_TOOL = "tracelens"
#: The only route of the roofline: the orchestrator's analysis with no model session (``agent`` starts one on the host).
ROOFLINE_ROUTE = AnalysisRoute.NON_LLM.value
#: The error class of a roofline that answered ``ok`` with no hot kernel.
ROOFLINE_WITHOUT_KERNELS = "roofline_without_kernels"
HOT_KERNELS_KEY = "hot_kernels"


class _SharesOps:
    """Mixin: the tool's pointer carries the analysed profile's per-op shares, when the request names its commit."""

    def output(self, result: Mapping[str, Any], request: ToolRequest) -> ToolOutput:
        return with_op_shares(super().output(result, request), request)  # type: ignore[misc]


class _TracelensTool(_SharesOps, ScratchHandlerTool):
    pass


async def roofline_handler(payload: dict, *, session_dir: Path) -> dict:
    """The kernel roofline of ``payload['trace_input']``: ``non-llm`` analysis for the platform the request names (else
    :data:`~.tracelens_report.DEFAULT_PLATFORM`). Any other ``analysis_route`` fails the job by name instead of being
    ignored, and the route the host's environment names is not read: it could start a model session. A trace that is missing or
    holds no GPU kernel fails by name: the analysis itself answers ``ok`` with no kernel for both. So does an analysis that
    answers ``ok`` with no hot kernel for a trace that has some (it scanned none of its events)."""
    route = str(payload.get("analysis_route") or "").strip().lower()
    if route and route != ROOFLINE_ROUTE:
        return {
            "status": "failed",
            "error_class": "invalid_analysis_route",
            "error": f"the roofline runs only the {ROOFLINE_ROUTE!r} route, not {route!r}",
        }
    unusable = await asyncio.to_thread(unusable_trace, str(payload["trace_input"]))
    if unusable is not None:
        return unusable
    platform = str(payload.get("target_platform") or "").strip() or DEFAULT_PLATFORM
    result = await trace_analyze_handler(
        {**payload, "analysis_route": ROOFLINE_ROUTE, "target_platform": platform}, session_dir=session_dir
    )
    return _without_kernels(result) or result


def _without_kernels(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """The failed result of an ``ok`` analysis that found no hot kernel in a trace that has GPU kernels; ``None`` otherwise."""
    if result.get("status") != "ok" or result.get(HOT_KERNELS_KEY):
        return None
    warnings = [str(w.get("code") or "") for w in result.get("trace_health_warnings") or () if isinstance(w, Mapping)]
    return {
        **result,
        "status": "failed",
        "error_class": ROOFLINE_WITHOUT_KERNELS,
        "error": (
            "the roofline analysis answered ok with no hot kernel, though the trace has GPU kernels: it scanned none of "
            f"their events (trace health warnings: {', '.join(filter(None, warnings)) or 'none'})"
        ),
    }


def roofline_tool() -> ScratchHandlerTool:
    """Analyse the roofline of a torch trace that already exists. Params: ``trace_input`` (a trace directory or file),
    ``framework``, ``model_name`` and ``target_platform`` (a GPU name such as MI355X, the default). Its pointer carries no op
    shares (they come from the TraceLens report of ``tracelens``); the kernels are in the result."""
    return ScratchHandlerTool(
        ROOFLINE_TOOL,
        roofline_handler,
        ok_text="roofline analysed",
        required=("trace_input",),
    )


async def tracelens_handler(payload: dict, *, session_dir: Path) -> dict:
    """``analysis_route`` (else ``HYPERLOOM_TRACE_ANALYSIS_ROUTE``) picks the analysis: ``report`` is TraceLens' deterministic
    report with no agent, ``agent`` and ``non-llm`` are the orchestrator's own routes. Unnamed, a serving framework gets
    ``report`` and a scriptable one (xDiT, custom: no steady-state window to split) the orchestrator's default."""
    raw = payload.get("analysis_route")
    route = ("" if raw is None else str(raw).strip()) or os.environ.get("HYPERLOOM_TRACE_ANALYSIS_ROUTE", "").strip()
    if not route and not is_scriptable(payload.get("framework")):
        route = REPORT_ROUTE
    if route.lower() == REPORT_ROUTE:
        return await report_handler(payload, session_dir=session_dir)
    result = await trace_analyze_handler(payload, session_dir=session_dir)
    return await asyncio.to_thread(with_window_shares, result, str(payload.get("trace_input") or ""), session_dir)


def tracelens_tool() -> ScratchHandlerTool:
    """Analyse a torch trace that already exists. Params: ``trace_input`` (a trace directory or file), optional
    ``analysis_route`` (``report`` by default, ``agent``, ``non-llm``), ``framework`` and ``model_name`` (the orchestrator's
    routes); ``commit_id`` (the commit the trace was profiled on) and, when the trace came from a PROFILE test,
    ``profile_request_id``, which the pointer carries with the op shares."""
    return _TracelensTool(
        TRACELENS_TOOL,
        tracelens_handler,
        ok_text="trace analysed",
        required=("trace_input",),
    )
