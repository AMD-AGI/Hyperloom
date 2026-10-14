# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``accuracy`` tool: one benchmark round with the accuracy evaluation on, run by the existing baseline executor.

The task kind is ``accuracy``, not ``baseline``, so the round never replaces the session's quality reference.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.env import is_truthy
from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom_multibranch.toolbox.tools.base import InvalidRequest, ToolRequest
from hyperloom_multibranch.toolbox.tools.executor_tool import ExecutorFn, ExecutorTool

_ACCURACY = "accuracy"
ACCURACY_TOOL = _ACCURACY


def _accuracy_executor(workdir: Path) -> ExecutorFn:
    return BaselineExecutor(session_dir=workdir)


class AccuracyTool(ExecutorTool):
    def validate(self, request: ToolRequest) -> None:
        if is_truthy(request.params.get("disable_run_eval")):
            raise InvalidRequest("the accuracy tool exists to run the evaluation; disable_run_eval is not allowed")


def accuracy_tool() -> AccuracyTool:
    """Params are the executor's own (``extra_server_args``, ``extra_envs``, ``model_path``, ...); the result carries
    ``accuracy``."""
    return AccuracyTool(
        ACCURACY_TOOL,
        _ACCURACY,
        _accuracy_executor,
        ok_text="accuracy evaluated",
        summary_keys=(_ACCURACY, "accuracy_pass"),
    )
