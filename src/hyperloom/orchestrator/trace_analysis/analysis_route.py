# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The routes a trace analysis can take, shared by the handler that validates one and the tool that runs it."""

from __future__ import annotations

from enum import Enum


class AnalysisRoute(Enum):
    """How a trace is analysed: ``agent`` runs the TraceLens skill under a model, ``non-llm`` runs TraceLens alone."""

    AGENT = "agent"
    NON_LLM = "non-llm"
