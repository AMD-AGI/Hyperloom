# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``build`` tool: a targeted framework or kernel build, run by the existing targeted-build executor."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors.targeted_build_executor import TargetedBuildExecutor
from hyperloom_multibranch.toolbox.tools.executor_tool import ExecutorFn, ExecutorTool

BUILD_TOOL = "build"


def _build_executor(workdir: Path) -> ExecutorFn:
    return TargetedBuildExecutor()


def build_tool() -> ExecutorTool:
    """Params are the targeted build action's own (see ``TargetedBuildAction.from_state``); a failed build fails the
    job with the build's failure summary."""
    return ExecutorTool(
        BUILD_TOOL,
        "targeted_build",
        _build_executor,
        ok_text="build finished",
        summary_keys=("attempt_root", "build_log_path"),
    )
