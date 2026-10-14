# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``roofline`` tool (profile, then TraceLens analysis) and the ``tracelens`` tool (analysis of a given trace)."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors.roofline import make_roofline_executor
from hyperloom.orchestrator.actions.executors.trace_analyze import trace_analyze_handler
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom_multibranch.toolbox.tools.executor_tool import ExecutorFn, ExecutorTool, HandlerTool

ROOFLINE_TOOL = "roofline"
TRACELENS_TOOL = "tracelens"


def _roofline_executor(workdir: Path) -> ExecutorFn:
    return make_roofline_executor(shared_state=SharedState.load_or_init(workdir))


def roofline_tool() -> ExecutorTool:
    """Params are the roofline executor's own (``reason``, ``extra_server_args``, ``extra_envs``, ...)."""
    return ExecutorTool(ROOFLINE_TOOL, "roofline", _roofline_executor, ok_text="roofline analysed")


def tracelens_tool() -> HandlerTool:
    """Analyse a torch trace that already exists. Params: ``trace_input`` (a trace directory or file), optional
    ``framework`` and ``model_name``."""
    return HandlerTool(
        TRACELENS_TOOL,
        trace_analyze_handler,
        ok_text="trace analysed",
        required=("trace_input",),
    )
