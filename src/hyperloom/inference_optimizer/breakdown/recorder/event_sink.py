# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where an executor's rows land, decided by its caller rather than by itself."""

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
    """What an executor needs of the thing its rows are written through."""

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
        """Bind a sink to one event id."""
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
        """Record one row, keyed and tagged for this sink's event."""
        from .recorder import get_recorder  # local: avoid an import cycle at module load

        key = ""
        try:
            declared = str(payload.get(EVENT_ID_FIELD) or "") if isinstance(payload, Mapping) else ""
            if declared and declared != self._event_id:
                # The core is meant to be ignorant of its event id, so a payload naming one is a leak, not a value to
                # trust.
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
    """Build the sink for one event."""
    return EventSink(event, producer=producer)
