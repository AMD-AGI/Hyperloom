# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The two timeline writes every event makes, and what a killed session leaves.

An event is written to the timeline exactly twice, however many facts it
collects in between. The opening write puts the shell there with
``status="running"``; the closing write assembles the rows and updates that
same entry in place. Facts recorded in between touch fragments only.

The opening write is not optional. Without it a live session shows no event at
all until the phase ends, and "a killed session can say which sub-step it
stopped on" is the capability the timeline was built for.

The sequence that write returns is stored on the event-level fragment, because
that fragment is the event's only durable identity. The closing write reuses it
to update one file instead of appending a second event, and its absence is what
separates the two residual states a killed session can leave:

* the shell was written and the closing write never ran -- the event is on disk
  as ``running`` and its sequence is recoverable from the fragment;
* the shell write itself failed, or rows were recorded before the event was
  opened -- there is no event and no sequence, so finalize has to allocate one.

Neither is guessed into a terminal status. The closing status is derived at
assembly from the evidence, so an event whose closing write never ran has
nothing that judged it; calling it ``succeeded`` because its fragments look
complete is exactly the inference this design exists to remove. Both become
:data:`EVENT_STATUS_INTERRUPTED`.
"""

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
    """One event a killed session left behind.

    Attributes:
        event_id (str): The event id the fragments are tagged with.
        sequence (int | None): The storage sequence to update in place, or
            ``None`` when no event was ever written and one must be allocated.
        state (str): :data:`RESIDUAL_RUNNING` or :data:`RESIDUAL_NO_EVENT`.
    """

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
    """Build the envelope every event type shares.

    Fields that mean the same thing for every type live here rather than being
    redeclared inside each ``ext``, because one semantic stored N times drifts.

    Args:
        event_type (str): The timeline event type, e.g. ``kernel``.
        event (str): The event id, which becomes the envelope's ``id``. Note
            the deliberate asymmetry with the rows: the envelope calls it
            ``id`` because it is this object's own identity, while a row calls
            it ``event_id`` because it is a reference to the event the row
            belongs to.
        status (str): The event status.
        kind (str): The type-specific sub-kind, when the type has one.
        start_time (str): ISO UTC timestamp the event opened.
        end_time (str): ISO UTC timestamp the event closed.
        ext (Mapping[str, Any] | None): The type-specific payload.

    Returns:
        dict[str, Any]: The envelope, carrying only the fields that were given.

    Raises:
        ValueError: If ``event`` is not a well-formed event id.
    """
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
    """Write the event shell and store its sequence on the event-level fragment.

    Both halves happen here so they cannot come apart: an event whose shell was
    written but whose sequence was not recorded would be updated by appending a
    second event at close, and one whose fragment was written but whose shell
    was not would be invisible until the phase ended.

    Opening an event twice returns the sequence the first call took rather than
    writing a second shell, so an event several actions share -- the rooflines
    one phase dispatched in one cycle, each of which opens the event it belongs
    to -- stays one entry on the timeline.

    Args:
        event_type (str): The timeline event type.
        event (str): The event id.
        event_section (str): The event-level section for this type, e.g.
            ``kernel_event``.
        producer (str): The producer label owning the fragment.
        kind (str): The type-specific sub-kind, when the type has one.
        start_time (str): ISO UTC timestamp the event opened.
        ext (Mapping[str, Any] | None): Whatever of ``ext`` is known already;
            the closing write replaces it with the assembled form.

    Returns:
        int | None: The storage sequence to close the event with, or ``None``
            when the write failed. A caller that gets ``None`` should carry on
            recording facts: the fragments still land, and finalize recovers
            the event from them.

    Raises:
        ValueError: If ``event`` is not a well-formed event id, or
            ``event_section`` is declared with a shape other than ``item``.
    """
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
    """Update the event in place with its assembled ``ext`` and final status.

    Args:
        event_type (str): The timeline event type.
        event (str): The event id.
        sequence (int | None): The sequence :func:`open_event` returned.
            ``None`` allocates a new event, which is what finalize does for
            fragments whose shell write never landed.
        status (str): The status assembly derived.
        ext (Mapping[str, Any]): The assembled type-specific payload.
        kind (str): The type-specific sub-kind, when the type has one.
        start_time (str): ISO UTC timestamp the event opened.
        end_time (str): ISO UTC timestamp the event closed.

    Returns:
        Path | None: The event file written, or ``None`` when the write failed
            and was parked for the next export.

    Raises:
        ValueError: If ``event`` is not a well-formed event id.
    """
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
    """Return the sequence an earlier :func:`open_event` took for this event.

    Args:
        event (str): The event id.
        event_section (str): The event-level section for this type.

    Returns:
        int | None: The sequence on the event-level fragment, or ``None`` when
            the event has not been opened yet or the spool cannot be read.
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
    """Classify the events that fragments describe but the timeline does not.

    Args:
        event_rows (Iterable[Mapping[str, Any]]): The event-level fragments of
            one type, as read back from the spool.
        event_type (str): The timeline event type those fragments belong to.

    Returns:
        list[ResidualEvent]: One entry per event id that still needs closing,
            in the order the rows were given. An event already on disk with a
            terminal status is not listed.
    """
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
    """Persist a writer failure for the next export, best-effort.

    Args:
        record_warning (Any): The ``record_write_warning`` callable.
        component (str): Dotted component name for the sidecar entry.
        exc (BaseException): The failure to record.
    """
    from ...session.session_binding import bound_session_or_none

    session = bound_session_or_none()
    if session is None:
        return
    try:
        record_warning(session, component=component, exc=exc)
    except Exception:  # noqa: BLE001 — the warning sidecar is itself best-effort
        log.debug("timeline: write-warning sidecar failed", exc_info=True)
