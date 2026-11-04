# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Asynchronous tools of the toolbox: each is one coroutine; :class:`ToolJobs` pushes every result to the inbox."""

from hyperloom_multibranch.toolbox.tools.base import (
    AsyncTool,
    InvalidRequest,
    JobStatus,
    ToolFailure,
    ToolJobs,
    ToolOutput,
    ToolRequest,
    ToolResultSink,
    UnknownTool,
)
from hyperloom_multibranch.toolbox.tools.accuracy import ACCURACY_TOOL, accuracy_tool
from hyperloom_multibranch.toolbox.tools.build import BUILD_TOOL, build_tool
from hyperloom_multibranch.toolbox.tools.kernel_tune import KERNEL_TUNE_TOOL, KernelTuneTool, kernel_tune_tool
from hyperloom_multibranch.toolbox.tools.profile import PROFILE_TOOL, profile_tool
from hyperloom_multibranch.toolbox.tools.roofline import ROOFLINE_TOOL, TRACELENS_TOOL, roofline_tool, tracelens_tool
from hyperloom_multibranch.toolbox.tools.upstream_pr import UPSTREAM_PR_TOOL, UpstreamPrSearch


def all_tools() -> tuple[AsyncTool, ...]:
    """Every tool of this package that needs nothing from the toolbox, for the runner that serves them.

    ``kernel_tune`` is not among them: it leases a card from the toolbox's pool (:func:`kernel_tune_tool`).
    """
    return (
        profile_tool(),
        UpstreamPrSearch(),
        roofline_tool(),
        tracelens_tool(),
        build_tool(),
        accuracy_tool(),
    )


__all__ = [
    "ACCURACY_TOOL",
    "BUILD_TOOL",
    "KERNEL_TUNE_TOOL",
    "PROFILE_TOOL",
    "ROOFLINE_TOOL",
    "TRACELENS_TOOL",
    "UPSTREAM_PR_TOOL",
    "AsyncTool",
    "InvalidRequest",
    "JobStatus",
    "KernelTuneTool",
    "ToolFailure",
    "ToolJobs",
    "ToolOutput",
    "ToolRequest",
    "ToolResultSink",
    "UnknownTool",
    "UpstreamPrSearch",
    "accuracy_tool",
    "all_tools",
    "build_tool",
    "kernel_tune_tool",
    "profile_tool",
    "roofline_tool",
    "tracelens_tool",
]
