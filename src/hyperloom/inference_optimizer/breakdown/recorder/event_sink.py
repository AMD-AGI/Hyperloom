# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where an executor's rows land, decided by its caller rather than by itself.

Some executors are both dispatched on their own and called inline by a phase
that has a timeline event of its own -- roofline is the standing example. The
inline run belongs to the outer event and must not occupy a timeline entry of
its own, and the rule for deciding that is a property of the *phase*, not of
the action: a phase that owns a timeline event absorbs everything dispatched
during it, and only a phase with no event of its own leaves top-level events
behind.

Rather than teaching the executor to ask "am I inline right now", which makes
the number of callers into the executor's complexity, the event id is lifted
out of it. The core writes sections and rows; a sink supplies the event id.
Both wrappers hand it the same kind of sink, so the core has no branch at all.

Both modes write the *same* sections. There is no logical-to-physical section
mapping, because assembly selects rows by event id: rows tagged
``kernel_agent:3:kernel`` assemble into the kernel event and rows tagged
``prelude:0:roofline`` into the roofline event, from one section either way.
Which wire position a row ends up in -- ``forge.reprofile.run`` or
``actions[].profile.runs[]`` -- is the assembler's decision, and freezing it
into a section name would decide it too early.

A sink also closes off the way a row can be orphaned. The fragment key needs
the event id in front and the payload needs it as a field; the two protect
against different failures, and neither is optional. Going through
:meth:`EventSink.record` means a caller cannot supply one and forget the
other.

Nothing recorded through a sink can break the run being recorded, so a failed
row is logged and dropped rather than raised. It is logged at warning, not
debug: a row that did not land is a fact absent from the assembled event, and
downstream that is indistinguishable from a thing that never happened.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .event_ids import fragment_key, parse_event_id
from .event_rows import EVENT_ID_FIELD

__all__ = ["EventSink", "RecordSink", "make_sink"]

log = logging.getLogger(__name__)


class RecordSink(Protocol):
    """What an executor needs of the thing its rows are written through.

    Declared as a protocol so an executor's signature does not name the
    concrete sink, and a test can pass a recording double.
    """

    @property
    def event_id(self) -> str:
        """str: The event the rows written through this sink belong to."""

    def record(
        self,
        section: str,
        payload: Mapping[str, Any],
        *,
        row_type: str = "",
        natural_ids: str | Sequence[str] = (),
    ) -> Path | None:
        """Record one row into ``section``."""


class EventSink:
    """Writes rows into one event, whichever event that turns out to be."""

    def __init__(self, event: str, *, producer: str) -> None:
        """Bind a sink to one event id.

        Args:
            event (str): The event id every row written through this sink is
                tagged with -- the executor's own event when it was dispatched
                standalone, the enclosing phase's event when it was called
                inline.
            producer (str): The producer label owning the fragments, so a row
                replayed on behalf of a subprocess stays distinguishable from
                one the coordinator observed itself.

        Raises:
            ValueError: If ``event`` is not a well-formed event id.
        """
        parse_event_id(event)
        self._event_id = str(event)
        self._producer = str(producer)

    @property
    def event_id(self) -> str:
        """str: The event the rows written through this sink belong to."""
        return self._event_id

    @property
    def producer(self) -> str:
        """str: The producer label the fragments are written under."""
        return self._producer

    def record(
        self,
        section: str,
        payload: Mapping[str, Any],
        *,
        row_type: str = "",
        natural_ids: str | Sequence[str] = (),
    ) -> Path | None:
        """Record one row, keyed and tagged for this sink's event.

        Args:
            section (str): The section the row belongs to, registered in
                ``SECTION_SHAPES`` with shape ``item``.
            payload (Mapping[str, Any]): The fields this call knows. Only
                these are written: repeated calls on one key deep-merge, so a
                caller never reads a row back to change one field.
            row_type (str): The kind of row, an author-time constant. Omit it,
                along with ``natural_ids``, for the event-level fragment.
            natural_ids (str | Sequence[str]): The row's own identity within
                its section, one value or several in narrowing order.

        Returns:
            Path | None: The fragment written, or ``None`` when nothing was
                written. Recording never breaks the run that is being recorded,
                so every failure ends here rather than propagating -- but none
                of them is quiet. A row that did not land is a fact missing
                from the event, which reads downstream as a thing that did not
                happen, so each failure is logged at warning with the section,
                the key and the traceback.
        """
        from .recorder import get_recorder  # local: avoid an import cycle at module load

        key = ""
        try:
            declared = str(payload.get(EVENT_ID_FIELD) or "") if isinstance(payload, Mapping) else ""
            if declared and declared != self._event_id:
                # The core is meant to be ignorant of its event id, so a
                # payload naming one is a leak, not a value to trust.
                raise ValueError(
                    f"payload claims event {declared!r} but this sink writes {self._event_id!r}; "
                    "the event id is the sink's to decide, so the caller should not set it"
                )
            ids = (natural_ids,) if isinstance(natural_ids, str) else tuple(natural_ids)
            key = fragment_key(self._event_id, row_type, *ids)
            row = {EVENT_ID_FIELD: self._event_id, **dict(payload)}
            return get_recorder(producer=self._producer).record_upsert_item(section, row, key=key)
        except Exception:  # noqa: BLE001 — observability cannot change phase behavior
            log.warning(
                "recorder: dropped a %s row of event %s (key %s, producer %s); "
                "the assembled event will be missing this fact",
                section,
                self._event_id,
                key or "<unbuilt>",
                self._producer,
                exc_info=True,
            )
            return None


def make_sink(event: str, *, producer: str) -> EventSink:
    """Build the sink for one event.

    The standalone and inline wrappers of a shared executor both call this;
    they differ only in the event id they pass, which is the whole of the
    difference between the two modes.

    Args:
        event (str): The event id rows are tagged with.
        producer (str): The producer label owning the fragments.

    Returns:
        EventSink: A sink writing into ``event``.

    Raises:
        ValueError: If ``event`` is not a well-formed event id.
    """
    return EventSink(event, producer=producer)
