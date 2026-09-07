# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Section renderer + LLM compose layer that turns a"""

from .base import Decision, RenderedSection
from .compose import ComposeResult, LLMClient, render_session_report
from .cross_section import GlobalFacts

__all__ = [
    "ComposeResult",
    "Decision",
    "GlobalFacts",
    "LLMClient",
    "RenderedSection",
    "render_session_report",
]
