# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What reaches a caller as one planning agent's answer.

A planning session reads source across many turns, so a long one can exhaust
its context window. The provider CLI answers that by compacting the session and
prepending a summary of everything so far to the next reply -- and that summary
is not the answer. Published as one, it hands the Implementer a hundred lines of
conversation recap before the plan it is supposed to execute.
"""

from __future__ import annotations

import pytest

from kernelforge.agent_backends.base import AgentRunResult
from kernelforge.orchestrator.agent_response import (
    AgentResponseIncompleteError,
    validated_agent_text,
)


_COMPACTED = """\
This session is being continued from a previous conversation that ran out of \
context. The summary below covers the earlier portion of the conversation.

Summary:
1. **Primary Request and Intent:**

   Produce the lane plan.

8. **Current Work:**

   Six file reads, no edits.

If you need specific details from before compaction (like exact code snippets, \
error messages, or content you generated), read the full transcript at: \
/root/.claude/projects/-root-ws/db43258a.jsonl
Continue the conversation from where it left off without asking the user any \
further questions. Resume directly -- do not acknowledge the summary, do not \
recap what was happening, do not preface with "I'll continue" or similar. Pick \
up the last task as if the break never happened.
# Lane plan

Retime the MFMA issue schedule.
"""


def test_a_compacted_session_publishes_its_plan_and_not_its_recap():
    """The summary belongs to the session, not to the answer it went on to give."""
    text = validated_agent_text(AgentRunResult(text=_COMPACTED), role="orchestration lane 1 synthesis")

    assert text.startswith("# Lane plan")
    assert "Retime the MFMA issue schedule." in text
    assert "ran out of context" not in text
    assert "Primary Request and Intent" not in text


def test_a_compaction_is_reported_because_the_plan_came_from_a_lossy_view(caplog):
    """Cutting it silently would hide that the session outgrew its window.

    Everything the planner read before the compaction reaches the plan only
    through a summary of it, which is worth knowing when the plan disappoints.
    """
    with caplog.at_level("WARNING"):
        validated_agent_text(AgentRunResult(text=_COMPACTED), role="orchestration lane 1 synthesis")

    assert "orchestration lane 1 synthesis" in caplog.text
    assert "ran out of context" in caplog.text


def test_an_answer_that_is_only_a_recap_is_no_answer():
    """A compaction with nothing after it left the caller nothing to publish."""
    only_recap = _COMPACTED.split("# Lane plan")[0]

    with pytest.raises(AgentResponseIncompleteError):
        validated_agent_text(AgentRunResult(text=only_recap), role="synthesis")


def test_an_ordinary_answer_is_untouched():
    """Guards the cut from reaching every response that never compacted."""
    plan = "# Lane plan\n\nStage the scale stream through LDS."

    assert validated_agent_text(AgentRunResult(text=plan), role="synthesis") == plan


def test_a_half_written_compaction_marker_is_left_alone():
    """Without its terminator the boundary is a guess, and guessing cuts the plan."""
    truncated = (
        "This session is being continued from a previous conversation that ran "
        "out of context.\n\n# Lane plan\n\nRetime the schedule."
    )

    assert validated_agent_text(AgentRunResult(text=truncated), role="synthesis") == truncated
