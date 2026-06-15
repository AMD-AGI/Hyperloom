# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic trajectory review — synthesise candidate directions from the
whole optimization lineage when the search stalls.

Reads the optimization journal (KEEP / REVERT / no_promote lineage), the
``explore_search`` winners history, and the roofline snapshots already kept on
:class:`SharedState`, then renders an advisory block that names exhausted
directions to avoid and an under-exploited bottleneck to push. Read-only and
fail-soft; never raises and never gates phase advance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .optimization_journal import (
    OUTCOME_NO_PROMOTE,
    OUTCOME_REVERT,
    Journal,
)
from .roofline_snapshot import BOTTLENECK_DOMAIN_HINTS, dominant_direction

log = logging.getLogger(__name__)

#: A cluster is "exhausted" once it accumulates this many non-promoting attempts.
_EXHAUSTED_MIN_ATTEMPTS: int = 2
#: Marginal gain (%) below which a non-promoting cluster is treated as a dead end.
_EXHAUSTED_MAX_GAIN_PCT: float = 1.0


def _load_journal_entries(session_dir: Path, shared_state: Any) -> list[Any]:
    """Return journal entries (empty on miss); header fields are read-only here."""
    try:
        journal = Journal.load_or_create(
            session_dir,
            session_id=str(getattr(shared_state, "session_id", "") or ""),
            model=str(getattr(shared_state, "model_name", "") or ""),
            hardware=str(getattr(shared_state, "hardware", "") or ""),
        )
    except Exception:  # noqa: BLE001 — review must never crash
        return []
    return list(getattr(journal, "entries", []) or [])


def _exhausted_clusters(entries: list[Any]) -> list[dict[str, Any]]:
    """Group repeated REVERT / no_promote attempts by (kind, change); dead ends first."""
    clusters: dict[tuple[str, str], dict[str, Any]] = {}
    for e in entries:
        outcome = getattr(e, "outcome", "")
        if outcome not in (OUTCOME_REVERT, OUTCOME_NO_PROMOTE):
            continue
        key = (str(getattr(e, "kind", "")), str(getattr(e, "change", "")))
        row = clusters.setdefault(
            key, {"kind": key[0], "change": key[1], "count": 0, "max_gain": None},
        )
        row["count"] += 1
        gain = getattr(e, "gain_pct", None)
        if isinstance(gain, (int, float)):
            row["max_gain"] = (
                gain if row["max_gain"] is None else max(row["max_gain"], gain)
            )
    dead = [
        r for r in clusters.values()
        if r["count"] >= _EXHAUSTED_MIN_ATTEMPTS
        and (r["max_gain"] is None or r["max_gain"] < _EXHAUSTED_MAX_GAIN_PCT)
    ]
    dead.sort(key=lambda r: r["count"], reverse=True)
    return dead


def _stalled_cycle_count(shared_state: Any) -> int:
    """Consecutive most-recent macro-cycles that produced no explore winner."""
    search = getattr(shared_state, "explore_search", None) or {}
    wh = search.get("winners_history") if isinstance(search, dict) else None
    cycles_with_winner = {
        row.get("cycle")
        for row in (wh or [])
        if isinstance(row, dict) and row.get("cycle") is not None
    }
    cycle = int(getattr(shared_state, "macro_cycle", 0) or 0)
    stalled = 0
    while cycle >= 0 and cycle not in cycles_with_winner:
        stalled += 1
        cycle -= 1
    return stalled


def build_trajectory_digest(
    session_dir: str | Path,
    shared_state: Any,
    *,
    max_directions: int = 5,
) -> str:
    """Render the advisory trajectory-review block; ``""`` when nothing to report.

    Deterministic aggregation only (no model call). Surfaces stall depth,
    exhausted directions to avoid, and the dominant roofline bottleneck with a
    suggested specialist lever to redirect exploration.
    """
    try:
        entries = _load_journal_entries(Path(session_dir), shared_state)
        snaps = [
            s for s in (getattr(shared_state, "roofline_snapshots", None) or [])
            if isinstance(s, dict)
        ]
        dead = _exhausted_clusters(entries)
        stalled = _stalled_cycle_count(shared_state)
        if not dead and not snaps and stalled == 0:
            return ""

        lines: list[str] = []
        validated = getattr(shared_state, "cumulative_gain_validated", None)
        if isinstance(validated, (int, float)):
            lines.append(
                f"progress: validated_gain={validated:.2f}%  "
                f"stalled_cycles={stalled}"
            )
        elif stalled:
            lines.append(f"progress: stalled_cycles={stalled}")

        if dead:
            lines.append("exhausted_directions (avoid re-proposing):")
            for row in dead[:max_directions]:
                gain = row["max_gain"]
                gain_str = (
                    f"max_gain {gain:.2f}%" if isinstance(gain, (int, float))
                    else "no positive gain"
                )
                change = row["change"] or "(unspecified)"
                lines.append(
                    f"  - [{row['kind']}] {change}: "
                    f"{row['count']} non-promoting attempts, {gain_str}"
                )

        if snaps:
            direction, pct = dominant_direction(snaps[-1])
            lever = BOTTLENECK_DOMAIN_HINTS.get(direction)
            if direction and lever:
                lines.append(
                    f"under-explored: dominant bottleneck={direction} "
                    f"({pct:.1f}%) → consider {lever[0]} (tag {lever[1]})"
                )

        if not lines:
            return ""
        lines.append(
            "advisory only: redirect exploration with this; it does not gate "
            "phase advance."
        )
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — review must never crash
        log.exception("trajectory_reviewer: build_trajectory_digest failed")
        return ""


__all__ = ["build_trajectory_digest"]
