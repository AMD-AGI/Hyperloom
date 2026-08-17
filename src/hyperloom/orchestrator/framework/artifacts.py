# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK candidate-level artifacts + outcome classification.

Deterministic, LLM-free observability helpers for the FRAMEWORK_AGENT phase:

- :func:`candidate_key` is the canonical candidate identity (precedence
  ``candidate_id or pr_url or ref``) used for candidate selection, dedup,
  progress-row keying, and task idempotency across the whole pump.
- :func:`candidate_slug` is the shared path-slug helper for per-candidate paths.
- :func:`summarize_candidate_outcomes` classifies a batch's progress rows
  into ``empty_discovery`` / ``tested_no_keep`` / ``tested_with_keep`` so the
  phase-done summary, report, and robustness advisory can tell "discovered
  nothing" apart from "tested candidates but none cleared the gate".

All helpers here are pure. The per-candidate ``decision.json`` /
``semantic_audit.json`` writers are gone: nothing read them back, and the
progress-row and journal fact-write paths carry the same information.
"""

from __future__ import annotations

import logging
from typing import Any


log = logging.getLogger(__name__)


# Per-candidate terminal statuses that mean the candidate reached the apply/bench
# stage (as opposed to being filtered before any source change).
_TESTED_STATUSES: frozenset[str] = frozenset({"kept", "reverted", "applied_no_bench", "apply_failed", "bench_reverted"})


def candidate_key(row: dict[str, Any] | None) -> str:
    """Canonical dedup/progress key for a FRAMEWORK candidate or progress row.

    The single source of truth for "which candidate is this" across the whole
    FRAMEWORK_AGENT pump: candidate selection, dedup, progress-row keying, task
    idempotency, and the known-id set all derive from this so a candidate that
    carries only a ``pr_url`` (no ``candidate_id``) can never dedup against a
    progress row keyed on its ``candidate_id`` (and vice-versa).

    Precedence is ``candidate_id or pr_url or ref``. Progress rows persist this
    value in their ``candidate_id`` field, so passing a progress row back
    through here is idempotent.

    Args:
        row: A candidate dict or ``framework_agent_phase_progress`` row (or
            ``None``).

    Returns:
        The candidate key, or ``""`` when none of the identity fields are set.
    """
    if not isinstance(row, dict):
        return ""
    return str(row.get("candidate_id") or row.get("pr_url") or row.get("ref") or "")


def candidate_slug(candidate_id: str) -> str:
    """Filesystem-safe slug for a candidate id (PR url / ref / synthetic id).

    Args:
        candidate_id: The candidate identifier (may contain ``/``, ``:`` …).

    Returns:
        A lowercased slug with non-``[a-z0-9._-]`` runs collapsed to ``-``,
        capped at 96 chars, defaulting to ``"candidate"`` when empty.
    """
    out: list[str] = []
    for ch in str(candidate_id).lower():
        out.append(ch if (ch.isalnum() or ch in ".-_") else "-")
    slug = "".join(out).strip("-")
    return (slug or "candidate")[:96]



def summarize_candidate_outcomes(
    progress: list[dict[str, Any]] | None,
    *,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Classify FRAMEWORK progress rows into a phase-outcome summary.

    Args:
        progress: ``framework_agent_phase_progress`` rows (each a dict carrying
            ``status`` / ``kept`` / ``batch_id``).
        batch_id: When set, only rows for this batch are counted; otherwise all
            rows are counted.

    Returns:
        ``{"total", "keeps", "tested", "by_status", "outcome_class"}`` where
        ``outcome_class`` is one of ``empty_discovery`` (no rows),
        ``tested_with_keep`` (>=1 KEEP), or ``tested_no_keep``.
    """
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
    "candidate_slug",
    "summarize_candidate_outcomes",
]
