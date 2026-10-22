# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist's record that its direction is done on a head: one branch-log entry, ``EXHAUSTED`` and a JSON object.

It is kept for the record and stops nothing; one that cites no test is no record, as an untested direction is not
done. The runtime deals the session's direction a record names to no new branch on that record's head, and when the
branch gives up, it quotes the latest one in its notice. ``why`` is a closed vocabulary so that give-ups can be counted by cause.
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
        if not document["tried"]:
            return None
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


def exhausted_records(records: Iterable[BranchLogRecord]) -> list[Exhausted]:
    """The well-formed records among a branch's log entries, oldest first."""

    return [found for record in records if (found := read_exhausted(record.entry.content)) is not None]


def latest_exhausted(records: Iterable[BranchLogRecord]) -> Exhausted | None:
    """The latest well-formed record among a branch's log entries."""

    found = exhausted_records(records)
    return found[-1] if found else None


__all__ = ["EXHAUSTED_PREFIX", "Exhausted", "ExhaustedWhy", "exhausted_records", "latest_exhausted", "read_exhausted"]
