# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist's record that its direction is done on a head: one branch-log entry, ``EXHAUSTED`` and a JSON object.

It is kept for the record and stops nothing. When the branch gives up, the runtime quotes the latest one in its notice
and deals that direction to no new branch on that head. ``why`` is a closed vocabulary so that give-ups can be counted
by cause.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from hyperloom_multibranch.contracts.notebook.records import BranchLogRecord

EXHAUSTED_PREFIX = "EXHAUSTED "


class ExhaustedWhy(str, Enum):
    NO_CEILING_ABOVE_BAR = "no_ceiling_above_bar"
    ALL_TESTED_OR_CLAIMED = "all_tested_or_claimed"
    BLOCKED_BY_TOOL = "blocked_by_tool"
    NEEDS_CAPABILITY = "needs_capability"


@dataclass(frozen=True)
class Exhausted:
    direction: str
    head: str
    tried: tuple[str, ...]
    why: ExhaustedWhy
    detail: str
    reopen_if: str

    def summary(self) -> str:
        return f"{self.why.value}: {self.detail}; worth again if {self.reopen_if}"


def read_exhausted(content: str) -> Exhausted | None:
    """The record ``content`` holds, or ``None`` when it is no record or a malformed one."""

    if not content.startswith(EXHAUSTED_PREFIX):
        return None
    try:
        document = json.loads(content[len(EXHAUSTED_PREFIX) :])
        return Exhausted(
            str(document["direction"]),
            str(document["head"]),
            tuple(str(request) for request in document["tried"]),
            ExhaustedWhy(document["why"]),
            str(document["detail"]),
            str(document["reopen_if"]),
        )
    except (ValueError, KeyError, TypeError):
        return None


def latest_exhausted(records: Iterable[BranchLogRecord]) -> Exhausted | None:
    """The latest well-formed record among a branch's log entries."""

    found = None
    for record in records:
        found = read_exhausted(record.entry.content) or found
    return found


__all__ = ["EXHAUSTED_PREFIX", "Exhausted", "ExhaustedWhy", "latest_exhausted", "read_exhausted"]
