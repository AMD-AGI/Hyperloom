# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``roofline`` tool (profile, then TraceLens analysis) and the ``tracelens`` tool (analysis of a given trace)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.orchestrator.actions.executors.roofline import make_roofline_executor
from hyperloom.orchestrator.actions.executors.trace_analyze import trace_analyze_handler
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom_multibranch.toolbox.tools.base import ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools.executor_tool import ExecutorFn, ExecutorTool, HandlerTool
from hyperloom_multibranch.toolbox.tools.op_shares import with_op_shares

ROOFLINE_TOOL = "roofline"
TRACELENS_TOOL = "tracelens"


class _SharesOps:
    """Mixin: the tool's pointer carries the analysed profile's per-op shares, when the request names its commit."""

    def output(self, result: Mapping[str, Any], request: ToolRequest) -> ToolOutput:
        return with_op_shares(super().output(result, request), request)  # type: ignore[misc]


class _RooflineTool(_SharesOps, ExecutorTool):
    pass


class _TracelensTool(_SharesOps, HandlerTool):
    pass


def _roofline_executor(workdir: Path) -> ExecutorFn:
    return make_roofline_executor(shared_state=SharedState.load_or_init(workdir))


def roofline_tool() -> ExecutorTool:
    """Params are the roofline executor's own (``reason``, ``extra_server_args``, ``extra_envs``, ...), plus the
    optional ``commit_id`` the profile runs on, without which the pointer carries no op shares."""
    return _RooflineTool(ROOFLINE_TOOL, "roofline", _roofline_executor, ok_text="roofline analysed")


def tracelens_tool() -> HandlerTool:
    """Analyse a torch trace that already exists. Params: ``trace_input`` (a trace directory or file), optional
    ``framework`` and ``model_name``; ``commit_id`` (the commit the trace was profiled on) and, when the trace came from a
    PROFILE test, ``profile_request_id``, which the pointer carries with the op shares."""
    return _TracelensTool(
        TRACELENS_TOOL,
        trace_analyze_handler,
        ok_text="trace analysed",
        required=("trace_input",),
    )
