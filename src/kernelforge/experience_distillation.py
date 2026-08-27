# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a run remembers from its own failures.

Both ledgers -- the loop's per-iteration one and forge-fuse's per-attempt one --
turn an error blob into a single informative line, match that line against a
table of known failure modes, and carry the resulting constraints into the next
prompt. Only the rules, the wording and the entry shape differ, so those stay
with each ledger and the mechanism lives here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


def extract_signature(text: str, *, markers: Sequence[str], limit: int) -> str:
    """Pull one normalized, informative line out of an error/outcome blob."""
    lines = [stripped for line in text.splitlines() if (stripped := line.strip())]
    for line in lines:
        if any(marker in line.lower() for marker in markers):
            return line[:limit]
    return lines[0][:limit] if lines else ""


class ConstraintMemory:
    """Deduped, insertion-ordered constraints, capped at the most recent ones."""

    def __init__(
        self,
        rules: Sequence[tuple[re.Pattern, str]],
        *,
        max_constraints: int,
    ) -> None:
        self.rules = rules
        self.max_constraints = max_constraints
        self.constraints: list[str] = []

    def add(self, constraint: str) -> None:
        """Remember one constraint, dropping the oldest past the cap."""
        constraint = constraint.strip()
        if not constraint or constraint in self.constraints:
            return
        self.constraints.append(constraint)
        if len(self.constraints) > self.max_constraints:
            # Newer, task-specific findings are worth more than the first ones.
            self.constraints = self.constraints[-self.max_constraints :]

    def distill(self, error_text: str, outcome: str) -> None:
        """Promote every known failure mode the evidence matches."""
        blob = f"{error_text}\n{outcome}"
        for pattern, constraint in self.rules:
            if pattern.search(blob):
                self.add(constraint)


def render_ledger(
    *,
    constraints_heading: str,
    constraints: Sequence[str],
    entries_heading: str,
    entry_lines: Sequence[Sequence[str]],
) -> str:
    """Lay out the constraints section and then one block per entry."""
    out: list[str] = []
    if constraints:
        out.append(constraints_heading)
        out.extend(f"- {constraint}" for constraint in constraints)
        out.append("")
    if entry_lines:
        out.append(entries_heading)
        for block in entry_lines:
            out.extend(block)
    return "\n".join(out).strip()
