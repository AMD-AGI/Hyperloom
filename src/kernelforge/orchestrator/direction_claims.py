# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a plan names the optimization direction it pursues, so its candidate can be attributed to it."""

from __future__ import annotations

import re
from dataclasses import dataclass

# What every planner is told when the ledger is on; the line is how a candidate is attributed.
DIRECTION_RULE = (
    "Begin every plan (and every lane's plan) with one line naming the direction it pursues: "
    "`Direction: <id>` for a direction in this ledger, or `Direction: new | <category> | <title>` for one it "
    "does not list. Prefer OPEN directions, then ACTIVE and KEPT ones; an EXHAUSTED direction has failed "
    "repeatedly since its last KEEP and needs new evidence before it is tried again."
)

_CLAIM_RE = re.compile(r"^\W*direction\W*:\s*(.+?)\s*$", re.IGNORECASE)
_CLAIM_SCAN_LINES = 8


@dataclass(frozen=True)
class DirectionClaim:
    """What a plan's ``Direction:`` line names: an existing id, or a new direction."""

    direction_id: str = ""
    new_title: str = ""
    new_category: str = ""


def parse_direction_line(text: str) -> DirectionClaim | None:
    """The first ``Direction:`` line among a plan's opening lines, or None when it names none."""
    lines = [line for line in (text or "").splitlines() if line.strip()][:_CLAIM_SCAN_LINES]
    for line in lines:
        match = _CLAIM_RE.match(line.strip())
        if not match:
            continue
        value = match.group(1).strip().strip("`*")
        parts = [part.strip().strip("`*") for part in value.split("|")]
        if parts[0].lower() == "new":
            if len(parts) >= 3 and parts[2]:
                return DirectionClaim(new_title=parts[2], new_category=parts[1])
            return None
        token = parts[0].split()[0].strip("`*") if parts[0] else ""
        return DirectionClaim(direction_id=token) if token else None
    return None
