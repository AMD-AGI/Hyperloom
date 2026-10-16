# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Summary line a plan and a reasoning log open with, the prompt section naming the log, and where the log lives.

The export that turns a run into Experiences reads these back, so the parsers, the
path rule, the section and the prompt rules that ask for the line are one contract.
"""

from __future__ import annotations

import pytest

from kernelforge.orchestrator.summary_line import (
    PLAN_SUMMARY_RULE,
    REASONING_LOG_RULE,
    SUMMARY_LABEL,
    read_reasoning_log_section,
    read_summary,
    reasoning_log_path,
    reasoning_log_section,
)

LANE_LOG = "/runs/softmax/ws/forge_experiments/orchestration/iter_002/lane_002.reasoning.md"
#: A session's first prompt as its transcript keeps it: the deadline, the section naming the session's own log, then
#: prior experience that names the log an earlier session kept.
TRANSCRIPT_PROMPT = (
    "## Session deadline\nYou have about 90 minutes of wall-clock for THIS session.\n"
    f"\n## Reasoning log\nKeep this session's reasoning log at `{LANE_LOG}`.\n"
    "\n## Prior experience (distilled constraints + recent iterations)\n"
    "- tool: Edit `/runs/softmax/ws/forge_experiments/orchestration/iter_001/optimization_plan.reasoning.md`\n"
)


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


@pytest.mark.parametrize(
    ("plan", "log"),
    [
        ("iter_001/optimization_plan.md", "iter_001/optimization_plan.reasoning.md"),
        ("iter_007/lane_002.md", "iter_007/lane_002.reasoning.md"),
    ],
)
def test_a_reasoning_log_sits_beside_its_plan(tmp_path, plan, log):
    for given in (str(tmp_path / plan), tmp_path / plan):
        assert reasoning_log_path(given) == tmp_path / log


def test_the_invocation_section_names_the_absolute_path_in_the_words_transcripts_keep(tmp_path, monkeypatch):
    """Transcripts keep the section in the words their session was given, and the export reads those words back to
    find each session's log: a new wording must go on reading the old one."""
    monkeypatch.chdir(tmp_path)

    section = reasoning_log_section("iter_001/optimization_plan.reasoning.md")

    log = tmp_path.resolve() / "iter_001" / "optimization_plan.reasoning.md"
    assert section == f"\n## Reasoning log\nKeep this session's reasoning log at `{log}`.\n"


def test_the_section_is_read_back_from_the_prompt_it_was_written_into(tmp_path):
    log = tmp_path / "iter_001" / "optimization_plan.reasoning.md"

    prompt = f"## Target\nsoftmax_kernel.py\n{reasoning_log_section(log)}## Prior experience\n(none)\n"

    assert read_reasoning_log_section(prompt) == str(log.resolve())


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        pytest.param(TRANSCRIPT_PROMPT, LANE_LOG, id="as-transcripts-keep-it"),
        pytest.param(TRANSCRIPT_PROMPT.replace("## Reasoning log\n", ""), None, id="a-log-outside-the-section"),
        pytest.param("## Reasoning log\nKeep no log this time.\n`/runs/x.reasoning.md`\n", None, id="no-path-under-it"),
        pytest.param("", None, id="empty"),
    ],
)
def test_only_the_section_names_the_sessions_own_log(prompt, expected):
    assert read_reasoning_log_section(prompt) == expected


@pytest.mark.parametrize("path", [None, ""])
def test_a_session_without_a_reasoning_log_gets_no_section(path):
    assert reasoning_log_section(path) == ""


def test_the_planner_is_shown_an_example_the_parser_reads():
    """The example line in the rule is the format the export expects, sentence for sentence."""
    example = read_summary(PLAN_SUMMARY_RULE)

    assert example is not None
    assert example.startswith("The small-M cases leave most CUs idle")
    assert example.endswith("speed those cases up.")
    assert "\\" not in example


@pytest.mark.parametrize("rule", [PLAN_SUMMARY_RULE, REASONING_LOG_RULE])
def test_each_rule_asks_for_the_label_the_parser_reads(rule):
    assert f"`{SUMMARY_LABEL}`" in rule


def test_the_reasoning_log_rule_keeps_the_log_out_of_the_deliverable():
    """Without this the workspace hygiene rule reads as an order to delete the log before ending the turn."""
    flat = " ".join(REASONING_LOG_RULE.split())

    assert "update the Summary" in flat
    assert "never commit it" in flat
    assert "never put it in the kernel tree" in flat
    assert "leave it in place" in flat


def test_the_reasoning_log_rule_asks_for_a_work_record_not_disclosed_reasoning():
    """Opus 5.5's ``[reasoning_extraction]`` safeguard blocks a system prompt that asks the model to disclose its
    own reasoning. The rule must frame the log as an engineering record of actions and results, not internal
    thinking, or every writable implementer session on Opus 5.5 is refused before it starts."""
    flat = " ".join(REASONING_LOG_RULE.split()).lower()

    # Phrasing that reads as chain-of-thought extraction to the classifier.
    for banned in ("the reasoning itself", "your hypotheses", "your thinking", "your thoughts", "chain of thought"):
        assert banned not in flat, banned
    # It still asks for a record of the engineering work actually done.
    assert "record the work" in flat
    for action in ("examined", "changed", "measured"):
        assert action in flat, action
