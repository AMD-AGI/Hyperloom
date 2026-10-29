# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Summary line a plan opens with, and the prompt rule that asks the planner for it.

The export that turns a run into Experiences reads the summary back, so the parser and
the prompt rule that asks for the line are one contract. The implementer's record is
the lesson turn, not a log beside the plan.
"""

from __future__ import annotations

import pytest

from kernelforge.workflows.forge_loop.roles.summary_line import PLAN_SUMMARY_RULE, SUMMARY_LABEL, read_summary


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "Summary: Small-M cases idle most CUs. This plan splits K.\n\n# Optimization plan\nStep 1.",
            "Small-M cases idle most CUs. This plan splits K.",
            id="first-line",
        ),
        pytest.param(
            "Summary: The K loop runs on too few\n  workgroups, so most CUs   idle.\nSplit K four ways.\n\nRest.",
            "The K loop runs on too few workgroups, so most CUs idle. Split K four ways.",
            id="wrapped-and-collapsed",
        ),
        pytest.param("Summary: Fill the GPU.\n## Evidence\nThe K loop.", "Fill the GPU.", id="ends-at-heading"),
        pytest.param("Summary: Fill the GPU.\n#\nmore", "Fill the GPU.", id="ends-at-empty-heading"),
        pytest.param(
            "Summary: Use #pragma unroll.\n#pragma is not a heading",
            "Use #pragma unroll. #pragma is not a heading",
            id="hash-word-continues",
        ),
        pytest.param("Summary:\nThe label stands alone.\n\nRest.", "The label stands alone.", id="label-alone"),
        pytest.param(
            "# Optimization plan\n\n   Summary:   Indented, after a heading.\n",
            "Indented, after a heading.",
            id="first-match-anywhere",
        ),
        pytest.param("Summary: First.\n\nSummary: Second.", "First.", id="first-of-two"),
    ],
)
def test_the_summary_paragraph_is_read_back(text, expected):
    assert read_summary(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("# Optimization plan\nSplit K across 4 workgroups.\n", id="no-label"),
        pytest.param("Summary:\n\nThe text after a blank line is not the summary.", id="empty-paragraph"),
        pytest.param("Summary:   \n# Plan", id="blank-label"),
        pytest.param("**Summary:** Bold is not the label.", id="bold-label"),
        pytest.param("## Summary\nA heading is not the label.", id="heading-label"),
    ],
)
def test_text_without_a_summary_reads_as_none(text):
    assert read_summary(text) is None


def test_the_planner_is_shown_an_example_the_parser_reads():
    """The example line in the rule is the format the export expects, sentence for sentence."""
    example = read_summary(PLAN_SUMMARY_RULE)

    assert example is not None
    assert example.startswith("The small-M cases leave most CUs idle")
    assert example.endswith("speed those cases up.")
    assert "\\" not in example


def test_the_planner_rule_asks_for_the_label_the_parser_reads():
    assert f"`{SUMMARY_LABEL}`" in PLAN_SUMMARY_RULE
