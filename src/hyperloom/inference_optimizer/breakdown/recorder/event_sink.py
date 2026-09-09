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

    def append(self, section: str, payload: Mapping[str, Any]) -> Path | None:
        """Append one row to ``section``, with no identity of its own."""


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

    def has_row(
        self,
        section: str,
        *,
        row_type: str = "",
        natural_ids: str | Sequence[str] = (),
    ) -> bool:
        """Whether a row with this identity is already recorded on this event.

        For a caller holding a fact that belongs to a row some earlier stage
        recorded, and that reconstructed this sink's event id rather than being
        handed it. :meth:`record` would mint the row if the reconstruction were
        wrong, so asking first is what keeps a late verdict from inventing the
        thing it was meant to rule on.

        Args:
            section (str): The section the row belongs to.
            row_type (str): The kind of row, as passed to :meth:`record`.
            natural_ids (str | Sequence[str]): The row's own identity.

        Returns:
            bool: Whether the row exists. ``False`` when the question itself
                could not be answered, which keeps the caller's guard closed.
        """
        from .recorder import get_recorder  # local: avoid an import cycle at module load

        try:
            ids = (natural_ids,) if isinstance(natural_ids, str) else tuple(natural_ids)
            return get_recorder(producer=self._producer).item_fragment_exists(
                section,
                key=fragment_key(self._event_id, row_type, *ids),
            )
        except Exception:  # noqa: BLE001 — observability cannot change phase behavior
            log.warning(
                "recorder: could not tell whether event %s holds a %s row; treating it as absent",
                self._event_id,
                section,
                exc_info=True,
            )
            return False

    def append(self, section: str, payload: Mapping[str, Any]) -> Path | None:
        """Append one row to ``section``, tagged for this sink's event.

        For rows that are observations in time rather than entities: a plateau
        reading, a lifecycle step. They have nothing to be keyed by, and
        minting a key for them buys nothing and costs a whole failure mode --
        two rows that mint the same one silently become a single row. An
        appended row gets a write-unique fragment instead, so a second row can
        never land on a first, and a resumed leg needs no knowledge of what an
        earlier leg already wrote.

        The price is that the fragment carries no identity, so an appended row
        cannot be revised later. A fact that gets re-ruled -- a gate whose
        verdict resolves after the fact -- wants :meth:`record` instead.

        Args:
            section (str): The section the row belongs to, registered in
                ``SECTION_SHAPES`` with shape ``item``.
            payload (Mapping[str, Any]): The whole row. Unlike :meth:`record`
                there is no merging, so this is written as given.

        Returns:
            Path | None: The fragment written, or ``None`` when nothing was.
                As with :meth:`record`, recording never breaks the run being
                recorded, so a failure is logged at warning and ends here.
        """
        from .recorder import get_recorder  # local: avoid an import cycle at module load

        try:
            declared = str(payload.get(EVENT_ID_FIELD) or "") if isinstance(payload, Mapping) else ""
            if declared and declared != self._event_id:
                raise ValueError(
                    f"payload claims event {declared!r} but this sink writes {self._event_id!r}; "
                    "the event id is the sink's to decide, so the caller should not set it"
                )
            row = {EVENT_ID_FIELD: self._event_id, **dict(payload)}
            return get_recorder(producer=self._producer).record_item(section, row)
        except Exception:  # noqa: BLE001 — observability cannot change phase behavior
            log.warning(
                "recorder: dropped an appended %s row of event %s (producer %s); "
                "the assembled event will be missing this fact",
                section,
                self._event_id,
                self._producer,
                exc_info=True,
            )
            return None


def make_sink(event: str, *, producer: str) -> EventSink:
    """Build the sink for one event.

    The standalone and inline wrappers of a shared executor both call this;
    they differ only in the event id they pass, which is the whole of the
    difference between the two modes.

    Returns:
        EventSink: A sink writing into ``event``.

    Raises:
        ValueError: If ``event`` is not a well-formed event id.
    """
    return EventSink(event, producer=producer)
