# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The notes a test's use of the compiled-code cache leaves in its result, short enough that none is lost (design 4.21).

The runner keeps a bounded message for the application, and a compiler that rewrites a hundred cached files must not push
out the note that says the cache failed verification. So a note is one of three kinds, and they come out in this order:

* a **problem**: the cache was unreadable, a stored snapshot failed verification, or a contribution was refused;
* a **fact**: a kind is not used, or only read, and why;
* a **tally**: files of one kind that were all treated the same way, said once with their number and the first few names.

Everything is clipped, and what does not fit in the budget is counted, never silently dropped.
"""

from __future__ import annotations

__all__ = ["LISTED", "SessionNotes"]

#: How many file names a tally names; the rest are counted.
LISTED = 3
_BUDGET = 1200
_LONGEST = 300


class SessionNotes:
    """The notes of one test, in the order they are to be read."""

    def __init__(self) -> None:
        self._problems: list[str] = []
        self._facts: list[str] = []
        self._tallies: dict[tuple[str, str], list[str]] = {}

    def problem(self, text: str) -> None:
        self._problems.append(_clip(text))

    def fact(self, text: str) -> None:
        self._facts.append(_clip(text))

    def tally(self, kind: str, what: str, name: str) -> None:
        """One more file of ``kind`` that ``what`` happened to; ``what`` completes "N files ..."."""
        self._tallies.setdefault((kind, what), []).append(name)

    def lines(self) -> tuple[str, ...]:
        tallies = [_tally_line(kind, what, names) for (kind, what), names in sorted(self._tallies.items())]
        return _within_budget([*self._problems, *self._facts, *tallies])


def _tally_line(kind: str, what: str, names: list[str]) -> str:
    unlisted = len(names) - LISTED
    listed = ", ".join(names[:LISTED]) + (f", and {unlisted} more" if unlisted > 0 else "")
    noun = "file" if len(names) == 1 else "files"
    return _clip(f"jit cache: {kind}: {len(names)} {noun} {what} ({listed})")


def _clip(text: str) -> str:
    return text if len(text) <= _LONGEST else text[: _LONGEST - 3] + "..."


def _within_budget(lines: list[str]) -> tuple[str, ...]:
    kept: list[str] = []
    used = 0
    for position, line in enumerate(lines):
        if kept and used + len(line) > _BUDGET:
            return (*kept, f"jit cache: {len(lines) - position} more notes left out")
        kept.append(line)
        used += len(line) + 2
    return tuple(kept)
