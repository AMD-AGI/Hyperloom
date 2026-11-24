# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every direction a campaign measured as worse, one line per iteration, for a session to check its own against."""

from __future__ import annotations

from collections.abc import Sequence

from kernelforge.workflows.forge_loop.memory.lessons import (
    LessonScope,
    LessonStore,
    listed_negatives,
    parse_scope_line,
    scope_conflicts,
)

# The size of the whole list in a prompt. Only the most recent lesson documents fit a prompt in full, so this list is
# what keeps the earlier iterations' negatives in front of a session.
MAX_LIST_CHARS = 6000

# The least room one iteration's line is given when many share that budget.
MIN_ENTRY_CHARS = 200

IN_SCOPE = "IN SCOPE"
RE_OPENABLE = "RE-OPENABLE"
UNSCOPED = "UNSCOPED"

_HEADER = (
    "## Measured negatives from every past iteration\n"
    "Each line is what one iteration measured as worse than where it started, with the conditions it was measured "
    "under. Check a direction against this list before you choose it. A line marked "
    f"{IN_SCOPE} stands: take that direction again only after changing a condition the line names. A line marked "
    f"{RE_OPENABLE} or {UNSCOPED} closes nothing until it is measured again, for the reason it gives. The full "
    "record behind a line is that iteration's document in the lesson directory named below."
)


def _standing(
    scope: LessonScope | None, *, current_cases: Sequence[str], kernel_source: str | Sequence[str | None] | None
) -> str:
    """Whether a line's negatives can be cited as they are, and if not, why."""
    if scope is None:
        return f"{UNSCOPED}: the conditions it was measured under were not recorded"
    reasons = scope_conflicts(scope, current_cases=current_cases, kernel_source=kernel_source)
    return IN_SCOPE if not reasons else f"{RE_OPENABLE}: " + "; ".join(reasons)


def _conditions(scope: LessonScope | None) -> str:
    """The cases and pinned constants a line was measured under, as far as they were recorded."""
    if scope is None:
        return ""
    parts = []
    if scope.held_fixed:
        parts.append("held fixed " + ", ".join(f"{name}={value}" for name, value in scope.held_fixed))
    if scope.cases:
        parts.append("measured on " + ", ".join(scope.cases))
    return " | ".join(parts)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 4)].rstrip() + " ..."


def render_measured_negatives(
    lessons: LessonStore,
    *,
    current_cases: Sequence[str] = (),
    kernel_source: str | Sequence[str | None] | None = None,
) -> str:
    """The list for the next session's prompt, or "" when no iteration recorded a negative."""
    entries: list[tuple[int, str, str]] = []
    for iteration in lessons.existing_iterations():
        document = lessons.read(iteration)
        negatives = listed_negatives(document)
        if not negatives:
            continue
        scope = parse_scope_line(document)
        standing = _standing(scope, current_cases=current_cases, kernel_source=kernel_source)
        conditions = _conditions(scope)
        entries.append((iteration, standing, negatives + (f" | {conditions}" if conditions else "")))
    if not entries:
        return ""
    room = max(MIN_ENTRY_CHARS, MAX_LIST_CHARS // len(entries))
    lines = [f"- iter {iteration} [{standing}] {_clip(body, room)}" for iteration, standing, body in entries]
    return _HEADER + "\n\n" + "\n".join(lines)
