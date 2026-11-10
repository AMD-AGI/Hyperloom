# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The plan Orchestration publishes when the critic's review could not be worked into a revised plan.

This module owns that plan end to end: Orchestration renders it, the critic's review in front of the original draft,
and ``kernelforge export-runs`` reads the draft back out of it.
"""

from __future__ import annotations

_PREFACE = (
    "The Orchestration revision was unavailable. Treat the "
    "Critic review below as mandatory planning guidance before "
    "editing; use the draft only as historical context."
)
_DRAFT_HEADING = "## Original draft"


def render_critic_fallback(*, draft_plan: str, critic_review: str, critic_verdict: str) -> str:
    """Preserve critic corrections when the model revision is unavailable."""
    return "\n".join(
        (
            "# Optimization plan",
            "",
            _PREFACE,
            "",
            f"## Critic verdict: {critic_verdict}",
            "",
            critic_review.strip(),
            "",
            _DRAFT_HEADING,
            "",
            draft_plan.strip(),
        )
    ).strip()


def original_draft(plan: str) -> str:
    """The plan as its planner wrote it: the draft a critic fallback carries, or ``plan`` itself when it is none."""
    review, heading, draft = plan.partition(f"\n{_DRAFT_HEADING}\n")
    return draft if heading and _PREFACE in review else plan
