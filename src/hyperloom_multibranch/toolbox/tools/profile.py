# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``profile`` tool: a benchmark round with the torch profiler on, run by the existing profile executor."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor
from hyperloom_multibranch.toolbox.tools.executor_tool import ExecutorFn, ExecutorTool

PROFILE_TOOL = "profile"


def _profile_executor(workdir: Path) -> ExecutorFn:
    return ProfileExecutor(session_dir=workdir)


def profile_tool() -> ExecutorTool:
    """Params are the executor's own: ``extra_server_args``, ``extra_envs``, ``model_path``, ``tp`` and so on.

    The result carries ``trace_dir``, ``trace_files`` and ``main_trace_path``; the summary names the main trace.
    """
    return ExecutorTool(
        PROFILE_TOOL,
        "profile",
        _profile_executor,
        ok_text="profile captured",
        summary_keys=("main_trace_path", "trace_dir"),
    )
