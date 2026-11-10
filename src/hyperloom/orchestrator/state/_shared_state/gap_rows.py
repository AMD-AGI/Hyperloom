# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rows one open gap contributes to the prompt: what it looks like, and what has been tried against it."""

from __future__ import annotations

from typing import Any

# A gap symptom, truncated for its prompt row.
_SYMPTOM_CHARS = 200


def recency_key(gap: dict[str, Any]) -> str:
    """Sort key putting the most recently updated gap first (fallback to first_seen_ts/insertion)."""
    return str(gap.get("last_updated_ts") or gap.get("first_seen_ts") or "")


def summary_row(gap: dict[str, Any]) -> str:
    """One gap's identity, where it sits, what it looks like, and how often it has been attempted."""
    cid = str(gap.get("canonical_id") or "?")
    layer = str(gap.get("layer") or "?")
    severity = str(gap.get("severity") or "?")
    symptom = str(gap.get("symptom") or "").replace("\n", " ").strip()
    if len(symptom) > _SYMPTOM_CHARS:
        symptom = symptom[: _SYMPTOM_CHARS - 3] + "..."
    attempts = gap.get("attempts") or []
    attempt_n = len(attempts) if isinstance(attempts, list) else 0
    last_tag = ""
    if isinstance(attempts, list) and attempts:
        last = attempts[-1]
        if isinstance(last, dict):
            last_tag = f" last={last.get('action', '?')}:{last.get('outcome', '?')}"
    return f"  - {cid} [{layer}/{severity}] {symptom}\n      attempts={attempt_n}{last_tag}"


def attempt_rows(gap: dict[str, Any], max_attempts: int) -> list[str]:
    """The newest ``max_attempts`` attempts against one gap, each with what it tried and how it ended."""
    attempts = gap.get("attempts") or []
    if max_attempts <= 0 or not isinstance(attempts, list):
        return []
    rows: list[str] = []
    for a in attempts[-max_attempts:]:
        if not isinstance(a, dict):
            continue
        fid = a.get("failure_id") or ""
        rows.append(
            f"        attempt: {a.get('action', '?')} outcome={a.get('outcome', '?')}"
            f" err={a.get('error_class', '')}" + (f" fid={fid}" if fid else "")
        )
    return rows
