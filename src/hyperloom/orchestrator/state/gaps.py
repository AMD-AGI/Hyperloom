# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gap ledger: upsert, dedup, and gap-based signal extraction."""

from __future__ import annotations

from typing import Any

from hyperloom.common.timeutil import now_iso

from ._shared_state import gap_rows

__all__ = ["GapsStateMixin"]

# gap ledger caps; both enforced in upsert_gap.
_GAPS_MAX_ENTRIES = 50
_GAPS_ATTEMPTS_HISTORY = 20

# Free-text gap fields an incoming row may restate; ``severity`` defaults to "medium" rather than "".
_GAP_TEXT_FIELDS: tuple[str, ...] = ("symptom", "layer", "severity", "domain_hint", "source", "provenance")


def _merge_gap_attempts(existing: list[Any], incoming: list[Any]) -> list[Any]:
    """Merge ``incoming`` attempts into ``existing`` once each, newest-last, capped at the history limit.

    ``refresh_gaps`` re-derives every attempt from the whole failure log on each call, so an equal attempt (same
    failure: action, error class, variant and ts) is the same one, not a new one. Equal rows take the incoming
    position so a re-derived log keeps its newest-last order under the cap.
    """
    older = [a for a in existing if a not in incoming]
    merged: list[Any] = []
    for attempt in older + incoming:
        if attempt not in merged:
            merged.append(attempt)
    # Capped tail; callers supply newest-last lists (convention).
    return merged[-_GAPS_ATTEMPTS_HISTORY:]


class GapsStateMixin:
    """Gap ledger read/write helpers mixed into SharedState."""

    def find_gap(self, canonical_id: str) -> dict[str, Any] | None:
        """Return the gap entry matching ``canonical_id`` (or ``None``)."""
        if not canonical_id:
            return None
        cid = str(canonical_id)
        for gap in self.gaps:
            if isinstance(gap, dict) and str(gap.get("canonical_id") or "") == cid:
                return gap
        return None

    def upsert_gap(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Insert or update one gap row, keyed by ``canonical_id``. Coordinator-only writer. Returns the merged entry."""
        if not isinstance(entry, dict):
            return {}
        cid = str(entry.get("canonical_id") or "").strip()
        if not cid:
            return {}
        now = now_iso()
        existing = self.find_gap(cid)
        if existing is None:
            merged: dict[str, Any] = {
                "canonical_id": cid,
                # ``provenance`` is an optional origin reference (PR/blog URL).
                **{key: str(entry.get(key) or "") for key in _GAP_TEXT_FIELDS},
                "severity": str(entry.get("severity") or "medium"),
                "first_seen_ts": str(entry.get("first_seen_ts") or now),
                "last_updated_ts": now,
                "attempts": list(entry.get("attempts") or [])[-_GAPS_ATTEMPTS_HISTORY:],
            }
            self.gaps.append(merged)
        else:
            # Field-wise merge: incoming non-empty values win except ``first_seen_ts``.
            for key in _GAP_TEXT_FIELDS:
                incoming = entry.get(key)
                if incoming:
                    existing[key] = str(incoming)
            existing.setdefault("first_seen_ts", str(entry.get("first_seen_ts") or now))
            existing["last_updated_ts"] = now
            incoming_attempts = list(entry.get("attempts") or [])
            if incoming_attempts:
                existing["attempts"] = _merge_gap_attempts(list(existing.get("attempts") or []), incoming_attempts)
            merged = existing
        # Enforce global cap, trimming oldest after the upsert so the just-touched gap is retained.
        if len(self.gaps) > _GAPS_MAX_ENTRIES:
            others = sorted((g for g in self.gaps if g is not merged), key=gap_rows.recency_key)
            self.gaps = others[-(_GAPS_MAX_ENTRIES - 1) :] + [merged]
        return merged

    def append_gap_attempt(
        self,
        canonical_id: str,
        attempt: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Append one attempt row to an existing gap; returns the gap or ``None`` when unknown."""
        gap = self.find_gap(canonical_id)
        if gap is None:
            return None
        attempts = list(gap.get("attempts") or [])
        attempts.append(dict(attempt) | {"ts": str(attempt.get("ts") or now_iso())})
        if len(attempts) > _GAPS_ATTEMPTS_HISTORY:
            attempts = attempts[-_GAPS_ATTEMPTS_HISTORY:]
        gap["attempts"] = attempts
        gap["last_updated_ts"] = now_iso()
        return gap
