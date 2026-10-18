# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``kernel_tune`` tool (GEAK, or Forge when the session selects it) and the ``forge_fusion`` tool."""

from __future__ import annotations

from hyperloom.orchestrator.kernel.request_handlers import run_fusion_handler, run_gemm_tuning_handler
from hyperloom_multibranch.toolbox.tools.executor_tool import HandlerTool

KERNEL_TUNE_TOOL = "kernel_tune"
FORGE_FUSION_TOOL = "forge_fusion"


def kernel_tune_tool() -> HandlerTool:
    """GEMM tuning with the existing handler, which picks GEAK or Forge from ``KERNEL_OPT_BACKEND_ORDER``."""
    return HandlerTool(KERNEL_TUNE_TOOL, run_gemm_tuning_handler, ok_text="kernel tuning finished")


def forge_fusion_tool() -> HandlerTool:
    """Autonomous kernel fusion with the existing Forge fusion handler."""
    return HandlerTool(FORGE_FUSION_TOOL, run_fusion_handler, ok_text="fusion finished")
