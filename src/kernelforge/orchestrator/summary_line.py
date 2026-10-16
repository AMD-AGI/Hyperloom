# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``Summary:`` line that opens every plan and every implementer reasoning log.

This module owns the format end to end: the prompt rules that ask the planner and the
implementer for it, the per-invocation section that names an implementer's reasoning
log, the parser that reads a summary back, and where a plan's reasoning log lives.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

SUMMARY_LABEL = "Summary:"

_HEADING = re.compile(r"#{1,6}(?:\s|$)")
_LOG_HEADING = "## Reasoning log"

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

REASONING_LOG_RULE = (
    "When the invocation names a reasoning log file, keep a running work record "
    "at exactly that path: a plain Markdown file you write with the Write and "
    "Edit tools. "
    f"Its first line is `{SUMMARY_LABEL}` followed by one to three plain "
    "sentences stating the problem this kernel has and the change you are making "
    "to address it. Below the summary, record the work as an engineering log: "
    "what you examined in the code and the evidence, what you changed, what each "
    "change measured, and which directions you kept or dropped on those "
    "measurements. Start the file before your first edit, and update the Summary "
    "before you finish so it describes the optimization now in the file. The log "
    "is not part of the deliverable and not a scratch file to clean up: never "
    "commit it, never put it in the kernel tree, and leave it in place when you "
    "end the turn."
)


def reasoning_log_section(path: str | os.PathLike[str] | None) -> str:
    """The per-invocation prompt section naming a session's reasoning log; empty when it keeps none."""
    if not path:
        return ""
    return f"\n{_LOG_HEADING}\nKeep this session's reasoning log at `{Path(path).resolve()}`.\n"


def reasoning_log_path(plan_path: str | os.PathLike[str]) -> Path:
    """The reasoning log beside a plan: ``optimization_plan.md`` -> ``optimization_plan.reasoning.md``."""
    return Path(plan_path).with_suffix(".reasoning.md")


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
