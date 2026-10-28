# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Markdown for the MB2 study reports: a table, and numbers that may be missing."""

from __future__ import annotations

from collections.abc import Sequence

NOT_AVAILABLE = "n/a"


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    """``headers`` and ``rows`` as the lines of a markdown table."""

    def line(cells: Sequence[object]) -> str:
        return "| " + " | ".join(str(cell) for cell in cells) + " |"

    return [line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)]


def number(value: float | None, digits: int = 2) -> str:
    """``value`` with ``digits`` decimals; :data:`NOT_AVAILABLE` when there is none."""

    return NOT_AVAILABLE if value is None else f"{value:.{digits}f}"


def share(part: int, whole: int) -> str:
    """``part`` of ``whole`` as a percentage with both counts; :data:`NOT_AVAILABLE` of nothing."""

    return NOT_AVAILABLE if whole == 0 else f"{part / whole:.0%} ({part}/{whole})"


__all__ = ["NOT_AVAILABLE", "number", "share", "table"]
