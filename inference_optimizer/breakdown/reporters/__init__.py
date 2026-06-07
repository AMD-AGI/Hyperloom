# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Section renderer + LLM compose layer that turns a
``session_breakdown.json`` dict into a user-facing markdown report.

Public API:

* :func:`render_session_report` — main entry. Returns
  :class:`ComposeResult` whose ``markdown`` attribute is the report
  text.

* :class:`RenderedSection`, :class:`Decision`, :class:`GlobalFacts` —
  surfaced for callers that want to plug in their own composer or
  consume the structured facts directly (UI, dashboards).

See :mod:`inference_optimizer.breakdown.reporters.compose` for the
high-level design notes and ``SKILL.md`` for operator guidance.
"""

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
