# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``forge_fusion`` tool (the ``kernel_tune`` tool, which needs a card of its own, is :mod:`.kernel_tune`)."""

from __future__ import annotations

from hyperloom.orchestrator.kernel.request_handlers import run_fusion_handler
from hyperloom_multibranch.toolbox.tools.executor_tool import HandlerTool

FORGE_FUSION_TOOL = "forge_fusion"


def forge_fusion_tool() -> HandlerTool:
    """Autonomous kernel fusion with the existing Forge fusion handler."""
    return HandlerTool(FORGE_FUSION_TOOL, run_fusion_handler, ok_text="fusion finished")
