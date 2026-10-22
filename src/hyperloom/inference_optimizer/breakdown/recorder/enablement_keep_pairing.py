# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pair each KEEP round with the revalidation windows that name it.

A KEEP opens windows, and every window row carries the reference of the KEEP it
judges and how it ended. A KEEP landed when a window ended ``promoted``. It is
awaiting revalidation when it is the KEEP the lane is currently validating and
no window has ended ``promoted``, ``rejected`` or ``cleared``; a ``stopped``
window is a pause, not a verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

VERDICT_PROMOTED = "promoted"
VERDICT_REJECTED = "rejected"
VERDICT_STOPPED = "stopped"
VERDICT_CLEARED = "cleared"

#: Event-level field naming the KEEP the lane is validating, and the window field naming the KEEP it judges.
VALIDATING_KEEP_FIELD = "validating_keep"
KEEP_REF_FIELD = "keep_ref"

_CONCLUDED = frozenset({VERDICT_PROMOTED, VERDICT_REJECTED, VERDICT_CLEARED})


def keep_ref(row: Mapping[str, Any]) -> str:
    """The reference a KEEP round is named by: its specialist task, else its ordinal."""
    task = str(row.get("task_id") or "").strip()
    return task or f"round{int(row.get('attempt') or 0)}"


def pair_keep_rounds(
    attempts: Sequence[Mapping[str, Any]],
    revalidations: Sequence[Mapping[str, Any]],
    validating: str,
    *,
    kept_status: str,
) -> list[dict[str, Any]] | None:
    """Copies of ``attempts`` whose KEEP rows carry ``landed`` / ``validation_pending``.

    ``None`` when nothing was recorded by reference, which is a session that
    predates it and keeps reading as it did.
    """
    if not validating and not any(window.get(KEEP_REF_FIELD) for window in revalidations):
        return None
    verdicts: dict[str, set[str]] = {}
    for window in revalidations:
        verdicts.setdefault(str(window.get(KEEP_REF_FIELD) or ""), set()).add(str(window.get("verdict") or ""))
    rows = [dict(row) for row in attempts]
    for row in rows:
        if str(row.get("status") or "") != kept_status:
            continue
        ref = keep_ref(row)
        ended = verdicts.get(ref, set())
        row["landed"] = bool(row.get("landed")) or VERDICT_PROMOTED in ended
        row["validation_pending"] = not row["landed"] and ref == validating and not ended & _CONCLUDED
    return rows
