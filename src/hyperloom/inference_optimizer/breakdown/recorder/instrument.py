# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Author-time instrumentation for ``session_breakdown.json``.

These helpers are called from the producing code (the Coordinator's
``SharedState``) to record breakdown facts where they are born, instead of
having the exporter re-walk artifacts later.

What lives here: the Coordinator's state snapshots.

Every helper is best-effort: spool failures degrade the section and never
propagate into the run they are describing.
Payloads are shaped to the matching ``schema.py`` TypedDict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom_common.timeutil import iso_z

from .recorder import recorder_for
from .session_metadata import PRODUCER_COORDINATOR, snapshot_metadata
from .trace import trace_skip


def snapshot_state_sections(session_dir: Path | str | None, state: Any) -> None:
    """Snapshot every state-owned breakdown section from a live ``SharedState``."""
    if not session_dir or state is None:
        trace_skip(reason="no session_dir" if not session_dir else "no state", section="session")
        return
    rec = recorder_for(session_dir, producer=PRODUCER_COORDINATOR)

    _snapshot_session(rec, state)
    snapshot_metadata(rec, state)


def _snapshot_session(rec, st: Any) -> None:
    """Snapshot the ``session`` singleton from ``st`` (no-op without a session id)."""
    session_id = str(getattr(st, "session_id", "") or "")
    if not session_id:
        return
    stop_reason = str(getattr(st, "stop_reason", "") or "")
    row = {
        "session_id": session_id,
        "claw_session_id": getattr(st, "claw_session_id", "") or "",
        "sandbox_user_id": getattr(st, "sandbox_user_id", "") or "",
        "start_ts": str(getattr(st, "start_ts", "") or ""),
        # A resumed run clears its reason but not necessarily the stale timestamp, so the pair is only ever
        # emitted together.
        "ended_at_utc": iso_z(getattr(st, "stop_ts", "")) if stop_reason else "",
        "stop_reason": stop_reason,
        "phase": str(getattr(st, "phase", "") or ""),
        "max_minutes": int(getattr(st, "max_minutes", 0) or 0),
        "tick_count": int(getattr(st, "tick", 0) or 0),
    }
    rec.record_singleton("session", row)


__all__ = [
    "snapshot_state_sections",
]
