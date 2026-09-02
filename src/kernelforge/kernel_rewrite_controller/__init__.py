# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Autonomous multi-operator coordination for KernelForge rewrite campaigns."""

from kernelforge.kernel_rewrite_controller.contracts import (
    HandoffBundle,
    HandoffContractError,
    KernelRewriteTask,
    TaskContractError,
    TaskParseResult,
    TaskState,
    TaskStateError,
)
from kernelforge.kernel_rewrite_controller.controller import (
    ControllerRunError,
    ControllerRunState,
    run_controller,
)
from kernelforge.kernel_rewrite_controller.dispatcher import (
    SingleTaskResult,
    dispatch_single_task,
)
from kernelforge.kernel_rewrite_controller.handoff import read_handoff
from kernelforge.kernel_rewrite_controller.paths import ControllerLayout, TaskLayout
from kernelforge.kernel_rewrite_controller.state import TaskStateStore
from kernelforge.kernel_rewrite_controller.task import (
    discover_task_dirs,
    load_task,
    parse_task_payload,
    sort_tasks,
)

__all__ = [
    "ControllerLayout",
    "ControllerRunError",
    "ControllerRunState",
    "HandoffBundle",
    "HandoffContractError",
    "KernelRewriteTask",
    "SingleTaskResult",
    "TaskContractError",
    "TaskLayout",
    "TaskParseResult",
    "TaskState",
    "TaskStateError",
    "TaskStateStore",
    "discover_task_dirs",
    "dispatch_single_task",
    "load_task",
    "parse_task_payload",
    "read_handoff",
    "run_controller",
    "sort_tasks",
]
