# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helpers for timeline event recorders.

Each event type owns how it is opened. What they share is republishing a
row that lands after the event was already closed, and declining to open
an event when no session is bound.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from .event_rows import rows_for_event
from .event_timeline import finish_event
from .recorder_warnings import RECORDING_ERRORS, note_failure

log = logging.getLogger(__name__)


def decline_unbound(label: str) -> bool:
    """True when no session is bound; logs why this event will be missing."""
    from ...session.session_binding import session_is_bound

    if session_is_bound():
        return False
    log.warning(
        "%s timeline: no session bound; this event will be missing from the "
        "breakdown. The coordinator binds at startup, so this means either "
        "that never happened or the work ran outside the session's context",
        label,
    )
    return True


def event_header(parts: Mapping[str, list[dict[str, Any]]], section: str, event: str) -> dict[str, Any]:
    """The first event-level fragment for ``event``, or ``{}``."""
    rows = rows_for_event(parts.get(section) or [], event)
    return rows[0] if rows else {}


def republish_closed_event(
    event: str,
    *,
    section: str,
    event_type: str,
    kind: str,
    load_parts: Callable[[], Mapping[str, Any]],
    assemble: Callable[..., tuple[dict[str, Any], str]],
    end_time: Callable[[Mapping[str, Any], dict[str, Any]], str],
) -> None:
    """Re-assemble a closed event so a fragment written after it is published.

    The export reads the durable timeline rather than re-assembling it, so a
    row landing after the close is in the spool but not in the event. Updating
    the same storage sequence puts it there. An event that is still running is
    left alone.

    Spool failures are parked. A bug in assembly raises.
    """
    from ...session.sbd_v6 import timeline_sequence

    try:
        parts = load_parts()
        header = event_header(parts, section, event)
        closed_end = end_time(parts, header)
        if not closed_end:
            return
        ext, derived = assemble(parts, event=event)
        finish_event(
            event_type=event_type,
            event=event,
            sequence=timeline_sequence(header),
            status=derived,
            ext=ext,
            kind=kind,
            start_time=str(header.get("start_time") or ""),
            end_time=closed_end,
        )
    except RECORDING_ERRORS as exc:
        note_failure(section=section, error=exc, detail=f"re-publishing closed event {event}")
