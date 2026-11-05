# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist's record that its direction is done on a head: one branch-log entry, ``EXHAUSTED`` and a JSON object.

It is kept for the record and stops nothing; one that cites no test is no record, as an untested direction is not
done. The runtime deals the session's direction a record names to no new branch on that record's head, and when the
branch gives up, it quotes the latest one in its notice. ``why`` is a closed vocabulary so that give-ups can be counted by cause.
A ``blocked_by_tool`` or ``needs_capability`` record whose detail names no file or call tried, or no error it gave, is
refused (:func:`exhausted_refusal`): one that names neither can be neither checked nor reopened.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from hyperloom_multibranch.contracts.blackboard.records import BranchLogRecord
from hyperloom_multibranch.specialist.directions import DIRECTIONS, direction_name

EXHAUSTED_PREFIX = "EXHAUSTED "

# A detail is prose, not a schema, so these only look for the words a tried path cannot be written without.
_FILE = re.compile(
    r"\b[\w./-]*\w\.(?:py|cu|cuh|cpp|cc|cxx|c|h|hpp|hip|csv|json|ya?ml|toml|sh|so|log|txt)(?::\d+)?\b"
    r"|(?<![\w/])(?:~|\.{1,2})?/\w[\w.-]*/[\w./-]*"
)
_CALL = re.compile(r"\b[A-Za-z_]\w*\(|\b[A-Za-z_]\w+(?:\.[A-Za-z_]\w+)+|\b[a-z]+[A-Z][a-z]\w*|\b[A-Za-z]\w*_\w+")
_ERROR = re.compile(
    r"error|exception|traceback|fail|crash|abort|segfault|illegal|missing|not found|no such|no module|unsupported"
    r"|undefined|timed out|timeout|out of memory|\boom\b|\bima\b|exit (?:code|status)|\brc ?= ?-?\d"
    r"|does not apply|did not apply|doesn't apply|refused|rejected|does not exist|doesn't exist|not supported"
    r"|cannot (?:find|open|load|import)|can't (?:find|open|load|import)",
    re.IGNORECASE,
)


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
    """The record ``content`` holds, anywhere in the entry, or ``None`` when it is no record, a malformed one or one
    :func:`exhausted_refusal` refuses: specialists write it after a note as often as on its own."""

    record = _parsed(content)
    return record if record is not None and not _lacks(record) else None


def exhausted_refusal(content: str) -> str:
    """Why the record ``content`` holds is refused, written to its specialist; ``""`` when it holds none or one that
    stands. Only a blocked_by_tool or needs_capability record is refused, for what its detail lacks of the path tried."""

    record = _parsed(content)
    lacks = _lacks(record) if record is not None else ""
    if not lacks:
        return ""
    return (
        f"your EXHAUSTED record of {record.direction!r} was refused: a {record.why.value} record's detail must name "
        f"the path you tried, and {record.detail!r} {lacks}; append it again naming the file or the call you tried "
        "and the error it gave"
    )


def _lacks(record: Exhausted) -> str:
    """What ``record``'s detail lacks of the path tried, ``""`` when nothing or when its ``why`` asks for none."""

    if record.why not in (ExhaustedWhy.BLOCKED_BY_TOOL, ExhaustedWhy.NEEDS_CAPABILITY):
        return ""
    tried = _FILE.search(record.detail) or _CALL.search(record.detail)
    error = _ERROR.search(record.detail)
    missing = [what for what, named in (("no file or call you tried", tried), ("no error it gave", error)) if not named]
    return f"names {' and '.join(missing)}" if missing else ""


def _parsed(content: str) -> Exhausted | None:
    start = content.find(EXHAUSTED_PREFIX + "{")
    if start < 0:
        return None
    try:
        document, _ = json.JSONDecoder().raw_decode(content, start + len(EXHAUSTED_PREFIX))
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


def session_direction(named: str) -> str | None:
    """The session direction a record's ``direction`` names: its name alone, or followed by a qualifier."""

    named = named.strip().lower()
    for direction in sorted(DIRECTIONS, key=lambda d: -len(direction_name(d))):
        name = direction_name(direction).lower()
        if named == name or (named.startswith(name) and not named[len(name)].isalnum()):
            return direction
    return None


def exhausted_records(records: Iterable[BranchLogRecord]) -> list[Exhausted]:
    """The well-formed records among a branch's log entries that are not refused, oldest first."""

    return [found for record in records if (found := read_exhausted(record.entry.content)) is not None]


def latest_exhausted(records: Iterable[BranchLogRecord]) -> Exhausted | None:
    """The latest well-formed record among a branch's log entries that is not refused."""

    found = exhausted_records(records)
    return found[-1] if found else None


__all__ = [
    "EXHAUSTED_PREFIX",
    "Exhausted",
    "ExhaustedWhy",
    "exhausted_records",
    "exhausted_refusal",
    "latest_exhausted",
    "read_exhausted",
    "session_direction",
]
