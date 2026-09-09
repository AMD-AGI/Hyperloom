# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The bring-up round ledger, read out of the session database for the breakdown."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ._common import _str_or_empty

#: Rounds carried in the section, newest first. ``round_count`` still reports
#: the true total.
_MAX_ROUNDS = 50

#: Where the round store keeps its two tables.
_DB_RELPATH = ("storage", "coordinator.db")

__all__ = ["collect_round_ledger"]


def collect_round_ledger(session_dir: Path, warnings: list[str]) -> dict[str, Any]:
    """Read this session's bring-up rounds and what they observed.

    Args:
        session_dir: Absolute session root.
        warnings: Shared warnings list, mutated in place on database errors.

    Returns:
        dict[str, Any]: ``rounds`` (newest first), ``round_count``,
        ``round_outcomes`` (outcome -> count), and the open round's
        ``round_id`` / ``round_holder_task_id`` when one still holds the
        machine. Empty when the session recorded no round.
    """
    conn = _open(session_dir, warnings)
    if conn is None:
        return {}
    try:
        rows = _query(
            conn,
            "SELECT round_id, state, outcome, holder_task_id, fence, opened_unix,"
            "       settled_unix"
            "  FROM bringup_rounds ORDER BY opened_unix DESC",
        )
    finally:
        conn.close()
    if not rows:
        return {}

    out: dict[str, Any] = {
        "rounds": [_round_summary(r) for r in rows[:_MAX_ROUNDS]],
        "round_count": len(rows),
        "round_outcomes": _outcome_counts(rows),
    }
    for row in rows:
        if _str_or_empty(row["state"]) == "open":
            out["round_id"] = _str_or_empty(row["round_id"])
            out["round_holder_task_id"] = _str_or_empty(row["holder_task_id"])
            break
    return out


def _open(session_dir: Path, warnings: list[str]) -> sqlite3.Connection | None:
    """Open the session database read-only, or report why it could not be.

    Args:
        session_dir: Absolute session root.
        warnings: Shared warnings list, mutated in place.

    Returns:
        sqlite3.Connection | None: The open connection, ``None`` when the
        database is absent or cannot be opened.
    """
    db_path = session_dir.joinpath(*_DB_RELPATH)
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", timeout=2.0, uri=True)
    except sqlite3.Error as exc:
        warnings.append(f"round ledger: open {db_path} failed: {exc!r}")
        return None
    conn.row_factory = sqlite3.Row
    return conn


def _query(conn: sqlite3.Connection, sql: str) -> list[sqlite3.Row]:
    """Run one read, treating a missing table as an empty result."""
    try:
        return list(conn.execute(sql).fetchall())
    except sqlite3.OperationalError:
        return []


def _round_summary(row: sqlite3.Row) -> dict[str, Any]:
    """Project one ``bringup_rounds`` row onto its breakdown shape."""
    settled = row["settled_unix"]
    return {
        "round_id": _str_or_empty(row["round_id"]),
        "state": _str_or_empty(row["state"]),
        "outcome": _str_or_empty(row["outcome"]),
        "holder_task_id": _str_or_empty(row["holder_task_id"]),
        "fence": _int(row["fence"]),
        "opened_unix": _float(row["opened_unix"]),
        "settled_unix": None if settled is None else _float(settled),
    }


def _outcome_counts(rows: list[sqlite3.Row]) -> dict[str, int]:
    """Count settled rounds by outcome, open rounds excluded."""
    counts: dict[str, int] = {}
    for row in rows:
        outcome = _str_or_empty(row["outcome"])
        if outcome:
            counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def _int(value: Any) -> int:
    """Coerce a column to a non-negative int, ``0`` when it is not one."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    """Coerce a column to a float, ``0.0`` when it is not one."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
