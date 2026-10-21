# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``Summary:`` line that opens every plan.

This module owns the format end to end: the prompt rule that asks the planner for it,
and the parser that reads a summary back. The implementer's record of an iteration is
the lesson turn, not a log kept beside the plan.
"""

from __future__ import annotations

import re

SUMMARY_LABEL = "Summary:"

_HEADING = re.compile(r"#{1,6}(?:\s|$)")

PLAN_SUMMARY_RULE = f"""\
The one fixed part of the plan is its very first line, before any heading: it
starts with `{SUMMARY_LABEL}` and gives one to three plain sentences stating the
problem this plan addresses (what the evidence shows is limiting the kernel)
and the plan's goal (what it changes and what it expects to gain). Write it as
plain text, with no heading marker, bold or bullet before the label. For example:

{SUMMARY_LABEL} The small-M cases leave most CUs idle because the K loop runs on \
too few workgroups. This plan splits K across 4 workgroups (split-K) with a \
reduction pass, aiming to fill the GPU and speed those cases up.
"""


def read_summary(text: str) -> str | None:
    """The first ``Summary:`` paragraph of ``text``, label removed and whitespace collapsed.

    The paragraph starts at the first line whose stripped text starts with the label
    and runs on through the lines that wrap it, up to a blank line or a Markdown
    heading. None when there is no such line or the paragraph is empty. Sentences are
    not counted, and a plan that wraps its draft (the critic fallback) is split by the
    caller.
    """
    lines = [line.strip() for line in text.splitlines()]
    start = next((index for index, line in enumerate(lines) if line.startswith(SUMMARY_LABEL)), None)
    if start is None:
        return None
    paragraph = [lines[start][len(SUMMARY_LABEL) :]]
    for line in lines[start + 1 :]:
        if not line or _HEADING.match(line):
            break
        paragraph.append(line)
    return " ".join(" ".join(paragraph).split()) or None
