# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Analysis Agent delivers a brief the Implementer acts on, measures only the canonical kernel, and works to a time box."""

from __future__ import annotations

import pytest

from kernelforge.workflows.forge_loop.roles.analysis_prompts import (
    ANALYSIS_SESSION_BRIEF,
    INCREMENTAL_REANALYSIS_PROMPT,
    PROFILED_SYSTEM_PROMPT,
    STATIC_SYSTEM_PROMPT,
)

_TIERS = pytest.mark.parametrize("prompt", [PROFILED_SYSTEM_PROMPT, STATIC_SYSTEM_PROMPT], ids=["profiled", "static"])


def _prose(prompt: str) -> str:
    return " ".join(prompt.split())


@_TIERS
def test_the_report_is_a_brief_ordered_from_where_time_goes_to_what_to_try(prompt) -> None:
    prose = _prose(prompt)

    positions = [
        prose.index(heading)
        for heading in (
            "1. Time distribution: for each scored case",
            "2. Bottlenecks, ranked by their effect on the score",
            "3. Directions: for each bottleneck, the optimization you expect to pay",
        )
    ]
    assert positions == sorted(positions)
    assert "the Implementer reads it in full before it edits" in prose
    assert "Cite raw evidence by path instead of pasting counter tables into the brief" in prose


@_TIERS
def test_trying_a_change_is_left_to_the_implementer(prompt) -> None:
    prose = _prose(prompt)

    assert "Do not edit a copy of the kernel, sweep a parameter, or benchmark a variant" in prose
    assert "a number taken from a variant is not evidence about the canonical kernel" in prose


@_TIERS
def test_the_session_timeout_is_a_budget_the_brief_must_survive(prompt) -> None:
    prose = _prose(prompt)

    assert "analysis_session_timeout_sec in the request is this session's wall-clock budget, not a target to fill" in (
        prose
    )
    assert "so the brief exists whenever the session ends" in prose
    assert "within analysis_session_timeout_sec" in ANALYSIS_SESSION_BRIEF["instructions"]


def test_a_profiled_session_takes_one_collection_per_case_unless_the_brief_would_have_to_guess() -> None:
    prose = _prose(PROFILED_SYSTEM_PROMPT)

    assert "One collection per case you profile is the default" in prose
    assert "add another only when report.md would otherwise have to guess" in prose


def test_a_group_member_is_told_what_it_takes_to_count_as_analyzed() -> None:
    """A member without its own evidence files publishes the bundle as PARTIAL, which buys a second session."""
    prose = _prose(PROFILED_SYSTEM_PROMPT)

    assert "copy the representative's profile files and normalized_metrics.json into the member's directory" in prose
    assert "publishes the bundle as PARTIAL" in prose


def test_a_refresh_costs_what_the_change_costs() -> None:
    prose = _prose(INCREMENTAL_REANALYSIS_PROMPT)

    assert "The cost of this session should follow the size of the change, not the size of the kernel" in prose
    assert "Re-profile only the cases whose previous measurements the change invalidates, once each" in prose
    assert "A case the change does not reach keeps its parent evidence" in prose
    assert "without the ones the KEEP sequence already took" in prose
