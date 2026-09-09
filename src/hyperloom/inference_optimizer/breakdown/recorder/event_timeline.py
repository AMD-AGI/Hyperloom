# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The two timeline writes every event makes, and what a killed session leaves."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from .event_ids import parse_event_id
from .event_rows import EVENT_ID_FIELD
from .event_sink import EventSink

__all__ = [
    "EVENT_STATUS_INTERRUPTED",
    "EVENT_STATUS_RUNNING",
    "RESIDUAL_NO_EVENT",
    "RESIDUAL_RUNNING",
    "TERMINAL_EVENT_STATUSES",
    "TIMELINE_SEQUENCE_FIELD",
    "ResidualEvent",
    "build_envelope",
    "finish_event",
    "open_event",
    "residual_events",
]

log = logging.getLogger(__name__)

#: The field the event-level fragment stores its timeline sequence under.
TIMELINE_SEQUENCE_FIELD = "timeline_sequence"

#: Status of an event whose shell has been written and which is still running.
EVENT_STATUS_RUNNING = "running"

#: Status of an event that never got its closing write. Distinct from every
#: terminal status because nothing judged the run: the fragments survived, the
#: verdict was never reached.
EVENT_STATUS_INTERRUPTED = "interrupted"

#: Statuses that mean an event closed. Anything outside this set, including a
#: missing status, leaves the event open as far as finalize is concerned.
TERMINAL_EVENT_STATUSES: frozenset[str] = frozenset(
    {
        "succeeded",
        "failed",
        "degraded",
        "skipped",
        EVENT_STATUS_INTERRUPTED,
    }
)

#: An event on disk as ``running`` whose closing write never ran. Its sequence
#: is on the event-level fragment, so finalize updates that same entry.
RESIDUAL_RUNNING = "running"

#: Fragments with no event behind them. Finalize has to allocate a sequence.
RESIDUAL_NO_EVENT = "no_event"


class ResidualEvent(NamedTuple):
    """One event a killed session left behind."""

    event_id: str
    sequence: int | None
    state: str


def build_envelope(
    *,
    event_type: str,
    event: str,
    status: str,
    kind: str = "",
    start_time: str = "",
    end_time: str = "",
    ext: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the envelope every event type shares."""
    parse_event_id(event)
    envelope: dict[str, Any] = {
        "type": str(event_type),
        "id": str(event),
        "status": str(status),
    }
    if kind:
        envelope["kind"] = str(kind)
    if start_time:
        envelope["start_time"] = str(start_time)
    if end_time:
        envelope["end_time"] = str(end_time)
    envelope["ext"] = dict(ext or {})
    return envelope


def open_event(
    *,
    event_type: str,
    event: str,
    event_section: str,
    producer: str,
    kind: str = "",
    start_time: str = "",
    ext: Mapping[str, Any] | None = None,
) -> int | None:
    """Write the event shell and store its sequence on the event-level fragment."""
    from ...session.sbd_v6 import record_write_warning, timeline_sequence, write_timeline_event

    already = _opened_sequence(event, event_section=event_section)
    if already is not None:
        return already

    envelope = build_envelope(
        event_type=event_type,
        event=event,
        status=EVENT_STATUS_RUNNING,
        kind=kind,
        start_time=start_time,
        ext=ext,
    )
    try:
        write_timeline_event(envelope)
    except Exception as exc:  # noqa: BLE001 — observability cannot change phase behavior
        log.debug("timeline: failed to open %s event %s", event_type, event, exc_info=True)
        _park(record_write_warning, component=f"timeline.{event_type}.open", exc=exc)
        return None

    sequence = timeline_sequence(envelope)
    sink = EventSink(event, producer=producer)
    payload: dict[str, Any] = {TIMELINE_SEQUENCE_FIELD: sequence}
    if start_time:
        payload["start_time"] = str(start_time)
    sink.record(event_section, payload)
    return sequence


def finish_event(
    *,
    event_type: str,
    event: str,
    sequence: int | None,
    status: str,
    ext: Mapping[str, Any],
    kind: str = "",
    start_time: str = "",
    end_time: str = "",
) -> Path | None:
    """Update the event in place with its assembled ``ext`` and final status."""
    from ...session.sbd_v6 import record_write_warning, set_timeline_sequence, write_timeline_event

    envelope = build_envelope(
        event_type=event_type,
        event=event,
        status=status,
        kind=kind,
        start_time=start_time,
        end_time=end_time,
        ext=ext,
    )
    if sequence is not None:
        set_timeline_sequence(envelope, sequence)
    try:
        return write_timeline_event(envelope)
    except Exception as exc:  # noqa: BLE001 — observability cannot change phase behavior
        log.debug("timeline: failed to close %s event %s", event_type, event, exc_info=True)
        _park(record_write_warning, component=f"timeline.{event_type}.finish", exc=exc)
        return None


def _opened_sequence(event: str, *, event_section: str) -> int | None:
    """The sequence an earlier :func:`open_event` took, from the event-level fragment.

    ``None`` when the event has not been opened yet or the spool cannot be read.
    """
    from ...session.sbd_v6 import timeline_sequence

    from .assembler import event_parts

    try:
        rows = event_parts((event_section,)).get(event_section) or []
    except Exception:  # noqa: BLE001 — a spool we cannot read is not a reason to skip the shell
        log.debug("timeline: cannot read %s to check whether %s is open", event_section, event, exc_info=True)
        return None
    for row in rows:
        if str(row.get(EVENT_ID_FIELD) or "") != str(event):
            continue
        sequence = timeline_sequence(row)
        if sequence is not None:
            return sequence
    return None


def residual_events(
    event_rows: Iterable[Mapping[str, Any]],
    *,
    event_type: str,
) -> list[ResidualEvent]:
    """Classify the events that fragments describe but the timeline does not."""
    from ...session.sbd_v6 import read_timeline_events, timeline_sequence

    from ...session.session_binding import bound_session_or_none

    session = bound_session_or_none()
    on_disk: dict[str, str] = {}
    if session is not None:
        for stored in read_timeline_events(session):
            if str(stored.get("type") or "") != str(event_type):
                continue
            stored_id = str(stored.get("id") or "")
            if stored_id:
                on_disk[stored_id] = str(stored.get("status") or "")

    residual: list[ResidualEvent] = []
    seen: set[str] = set()
    for row in event_rows:
        if not isinstance(row, Mapping):
            continue
        event = str(row.get(EVENT_ID_FIELD) or "")
        if not event or event in seen:
            continue
        seen.add(event)
        if on_disk.get(event, "") in TERMINAL_EVENT_STATUSES:
            continue
        sequence = timeline_sequence(row)
        residual.append(
            ResidualEvent(
                event_id=event,
                sequence=sequence,
                state=RESIDUAL_RUNNING if sequence is not None else RESIDUAL_NO_EVENT,
            )
        )
    return residual


def _park(record_warning: Any, *, component: str, exc: BaseException) -> None:
    """Persist a writer failure for the next export, best-effort."""
    from ...session.session_binding import bound_session_or_none

    session = bound_session_or_none()
    if session is None:
        log.warning("timeline: %s failed with no session bound, so it cannot be parked: %r", component, exc)
        return
    try:
        record_warning(session, component=component, exc=exc)
    except Exception:  # noqa: BLE001 — the warning sidecar is itself best-effort
        # The sidecar is what makes the parked failures above visible in the export, so losing it is the point at
        # which the original failure would otherwise go unreported entirely.
        log.warning(
            "timeline: cannot park the %s failure %r; it will not reach the export", component, exc, exc_info=True
        )
