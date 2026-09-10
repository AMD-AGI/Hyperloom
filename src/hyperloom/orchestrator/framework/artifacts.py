# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK candidate-level artifacts + outcome classification."""

from __future__ import annotations

from typing import Any


# Per-candidate terminal statuses that mean the candidate reached the apply/bench stage (as opposed to being filtered
# before any source change).
_TESTED_STATUSES: frozenset[str] = frozenset({"kept", "reverted", "applied_no_bench", "apply_failed", "bench_reverted"})


def candidate_key(row: dict[str, Any] | None) -> str:
    """Canonical dedup/progress key for a FRAMEWORK candidate or progress row."""
    if not isinstance(row, dict):
        return ""
    return str(row.get("candidate_id") or row.get("pr_url") or row.get("ref") or "")


def summarize_candidate_outcomes(
    progress: list[dict[str, Any]] | None,
    *,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Classify FRAMEWORK progress rows into a phase-outcome summary."""
    rows = [r for r in (progress or []) if isinstance(r, dict)]
    if batch_id is not None:
        rows = [r for r in rows if str(r.get("batch_id") or "") == str(batch_id)]
    by_status: dict[str, int] = {}
    keeps = 0
    tested = 0
    for r in rows:
        st = str(r.get("status") or "")
        by_status[st] = by_status.get(st, 0) + 1
        if bool(r.get("kept")) or st == "kept":
            keeps += 1
        if st in _TESTED_STATUSES:
            tested += 1
    if not rows:
        outcome_class = "empty_discovery"
    elif keeps > 0:
        outcome_class = "tested_with_keep"
    else:
        outcome_class = "tested_no_keep"
    return {
        "total": len(rows),
        "keeps": keeps,
        "tested": tested,
        "by_status": by_status,
        "outcome_class": outcome_class,
    }


__all__ = [
    "candidate_key",
    "summarize_candidate_outcomes",
]
