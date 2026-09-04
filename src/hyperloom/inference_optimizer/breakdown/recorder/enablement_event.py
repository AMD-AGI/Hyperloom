# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``enablement`` event: the repair rounds a stuck baseline needed.

Enablement is the loop that runs when a (model, backend) combo cannot be
benchmarked at all: a specialist authors a patch, ``integrate_patch`` applies it
and re-benches, and the verdict either keeps it, advances to a deeper gap, or
reverts. Almost none of that survived into the breakdown. Only the patches of
rounds that were kept reached it, flattened and detached from the round that
produced them, so the question a later session asks first -- has this combo been
attempted, and how far did it get -- had no answer.

A reverted round records which deeper gap the boot reached and the server log
that says so, and a serial enablement is a sequence of such rounds by
construction: each clears gap #n and the next boot stops at gap #n+1.

Unlike the other event types, this recorder is not held across the work it
records. An enablement round is
dispatched on one coordinator tick and settles on another, in a process that may
not be the one that dispatched it, so the recorder is rebuilt at each call site
and every fact is a partial update to a row keyed by the round's task id.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .event_fields import (
    as_dict as _as_dict,
    as_list as _as_list,
    clip as _clip,
    failure_row as _failure_row,
    now_iso_seconds as _now_iso,
)
from .event_ids import event_id
from .event_rows import rows_for_event, sort_rows, wire_rows
from .event_sink import RecordSink
from .event_timeline import finish_event, open_event

log = logging.getLogger(__name__)

EVENT_TYPE = "enablement"
EVENT_KIND = "enablement"

#: The component segment of an enablement event id. The phase segment is the
#: phase the rounds were dispatched in, which is why it is a parameter.
EVENT_COMPONENT = "enablement"

PRODUCER = "orchestrator"

#: The event-level section, one fragment per event, holding the timeline
#: sequence the two writes share and the facts describing the effort rather
#: than a round.
SECTION_EVENT = "enablement_event"

#: One fragment per round, keyed by the specialist task id that authored it.
#: The rounds cannot live as a list inside the event-level fragment: a partial
#: update to a row nested there appends instead of merging, so the dispatch
#: write and the settle write would produce two half-rounds.
SECTION_ROUND = "enablement_round"

ROW_ROUND = "round"

#: The verdicts ``integrate_patch`` returns for a round. ``kept`` means the
#: combo now runs, ``advanced`` that the boot reached a deeper gap, ``reverted``
#: that it did not move.
STATUS_KEPT = "kept"
STATUS_ADVANCED = "advanced"
STATUS_REVERTED = "reverted"

# Grounding drops and switch problems are prose, one line per patch refused or
# per manifest entry rejected, and the manifest admits MAX_SWITCHES of them.
_MAX_ROUND_NOTES = 8

__all__ = [
    "EVENT_COMPONENT",
    "EVENT_KIND",
    "EVENT_TYPE",
    "PRODUCER",
    "ROW_ROUND",
    "SECTION_EVENT",
    "SECTION_ROUND",
    "STATUS_ADVANCED",
    "STATUS_KEPT",
    "STATUS_REVERTED",
    "EnablementEventRecorder",
    "assemble_enablement_ext",
    "assemble_enablement_rounds",
    "enablement_event_id",
    "make_enablement_recorder",
]


def enablement_event_id(phase: str, macro_cycle: Any) -> str:
    """Build the event id of the enablement effort of one phase and cycle.

    Args:
        phase (str): The coordinator phase the rounds were dispatched in.
        macro_cycle (Any): The macro cycle they were dispatched in.

    Returns:
        str: The event id, ``{phase}:{macro_cycle}:enablement``.

    Raises:
        ValueError: If either segment is malformed.
    """
    return event_id(phase, macro_cycle, EVENT_COMPONENT)


def _names(paths: Any) -> list[str]:
    """Project applied patch paths to their file names.

    Args:
        paths (Any): The recorded ``patches_applied`` paths.

    Returns:
        list[str]: The base names. Names and not paths: the recorded paths are
            workspace originals under ``runs/``, which the archive drops. A
            name matches the ``patch`` entry the same round recorded in
            ``files``, which is the fetchable form.
    """
    return [Path(str(path)).name for path in _as_list(paths) if str(path or "")]


def _artifacts(rows: Any) -> list[dict[str, str]]:
    """Project whole-file replacements to the target each one overwrote.

    Args:
        rows (Any): The recorded ``artifacts_applied`` rows.

    Returns:
        list[dict[str, str]]: One ``{"target"}`` entry per artifact. A row's
            ``source`` and ``backup`` are workspace paths the archive does not
            hold; the target is the position in the framework tree, which is
            what a replay needs and what outlives the session.
    """
    return [
        {"target": str(row.get("target") or "")}
        for row in _as_list(rows)
        if isinstance(row, Mapping) and str(row.get("target") or "")
    ]


def _effective_config(value: Any) -> dict[str, Any]:
    """Project the configuration delta the round's bench ran with.

    Args:
        value (Any): The recorded ``enablement_effective_config``.

    Returns:
        dict[str, Any]: The four layers enablement composes over the base
            config, which is archived as a file and referenced from ``files``.
    """
    config = _as_dict(value)
    return {
        "extra_server_args": str(config.get("extra_server_args") or ""),
        "extra_envs": {str(key): str(val) for key, val in _as_dict(config.get("extra_envs")).items()},
        "remove_args": [str(arg) for arg in _as_list(config.get("remove_args")) if str(arg or "")],
        "args_mode": str(config.get("args_mode") or ""),
    }


def _notes(rows: Any) -> dict[str, Any]:
    """Clip and bound a round's free-text notes.

    Args:
        rows (Any): The recorded note list.

    Returns:
        dict[str, Any]: ``{"count", "items"}``, where the count is what says
            the head is a head.
    """
    notes = [_clip(row) for row in _as_list(rows) if str(row or "")]
    return {"count": len(notes), "items": notes[:_MAX_ROUND_NOTES]}


class EnablementEventRecorder:
    """Records the enablement effort of one phase and cycle into its event.

    Holds a sink and nothing else; see :data:`SECTION_ROUND` for why.
    """

    def __init__(self, sink: RecordSink):
        """Bind a recorder to one enablement event.

        Args:
            sink (RecordSink): Where the rows go, which decides the event they
                belong to.
        """
        self._sink = sink

    @property
    def event_id(self) -> str:
        """str: The event the rows written through this recorder belong to."""
        return self._sink.event_id

    def begin(self, *, mode: str = "", origin: str = "") -> None:
        """Open the event and record what admitted the effort.

        Opening is idempotent, so the lane can call this as it dispatches each
        round rather than tracking whether the first one already did.

        Args:
            mode (str): The resolved ``--enablement`` mode admitting the lane.
            origin (str): ``eval`` when an accuracy failure triggered the
                effort, empty when a launch failure did.
        """
        open_event(
            event_type=EVENT_TYPE,
            event=self.event_id,
            event_section=SECTION_EVENT,
            producer=PRODUCER,
            kind=EVENT_KIND,
            start_time=_now_iso(),
        )
        self._sink.record(SECTION_EVENT, {"mode": str(mode or ""), "origin": str(origin or "")})

    def round_dispatched(self, *, task_id: str, failure_kind: str = "") -> None:
        """Record that a round was dispatched, and when.

        Args:
            task_id (str): The authoring specialist's task id, which is the
                round's identity for the rest of its life.
            failure_kind (str): The classified gap the round was aimed at.
        """
        self._record_round(task_id, {"started_at": _now_iso(), "failure_kind": str(failure_kind or "")})

    def round_settled(
        self,
        *,
        task_id: str,
        result: Mapping[str, Any] | None,
        files: list[dict[str, str]] | None = None,
    ) -> None:
        """Record how a round ended and what it left in the archive.

        Args:
            task_id (str): The round's specialist task id.
            result (Mapping[str, Any] | None): The ``integrate_patch`` result.
            files (list[dict[str, str]] | None): The deliverables the archive
                took, as ``snapshot_round`` reported them.
        """
        payload = _as_dict(result)
        self._record_round(
            task_id,
            {
                "status": str(payload.get("status") or STATUS_REVERTED),
                "ended_at": _now_iso(),
                "patches": _names(payload.get("patches_applied")),
                "artifacts": _artifacts(payload.get("artifacts_applied")),
                "setup_commands": [str(cmd) for cmd in _as_list(payload.get("setup_commands_applied")) if cmd],
                "after_signature": _as_dict(payload.get("after_signature")),
                "effective_config": _effective_config(payload.get("enablement_effective_config")),
                "framework_switch_problems": _notes(payload.get("framework_switch_problems")),
                "grounding_drops": _notes(payload.get("patches_dropped_by_grounding")),
                "files": [
                    {"path": str(entry.get("path") or ""), "role": str(entry.get("role") or "")}
                    for entry in (files or [])
                    if isinstance(entry, Mapping) and str(entry.get("path") or "")
                ],
            },
        )

    def finish(self, *, succeeded: bool, stop_reason: str = "") -> None:
        """Close the event on the verdict the effort reached.

        What the effort failed at is not passed in: it is the last round's
        residual signature, which assembly reads off the round itself.

        Args:
            succeeded (bool): Whether the combo ended up runnable.
            stop_reason (str): The run-level stop reason the effort set, which
                for a stalled enablement is what ended the session.
        """
        self._sink.record(
            SECTION_EVENT,
            {
                "succeeded": bool(succeeded),
                "stop_reason": str(stop_reason or ""),
                "end_time": _now_iso(),
            },
        )
        from ...session.sbd_v6 import timeline_sequence
        from .assembler import enablement_event_parts

        parts = enablement_event_parts()
        ext, derived = assemble_enablement_ext(parts, event=self.event_id)
        finish_event(
            event_type=EVENT_TYPE,
            event=self.event_id,
            # Off the fragment and not off this recorder: the tick that closes
            # the event is not the one that opened it.
            sequence=timeline_sequence(_event_row(parts, self.event_id)),
            status=derived,
            ext=ext,
            kind=EVENT_KIND,
            start_time=_event_field(parts, self.event_id, "start_time"),
            end_time=_event_field(parts, self.event_id, "end_time"),
        )

    def is_closed(self) -> bool:
        """Whether this event has already had its closing write.

        Returns:
            bool: ``True`` once :meth:`finish` has recorded an end.
        """
        from .assembler import enablement_event_parts

        return bool(_event_field(enablement_event_parts(), self.event_id, "end_time"))

    def _record_round(self, task_id: str, payload: Mapping[str, Any]) -> None:
        """Update one round's row.

        Args:
            task_id (str): The round's specialist task id.
            payload (Mapping[str, Any]): The fields this call knows.
        """
        identity = str(task_id or "").strip()
        if not identity:
            # Two rounds sharing an empty id would upsert into one row.
            return
        self._sink.record(
            SECTION_ROUND,
            {"specialist_task_id": identity, **dict(payload)},
            row_type=ROW_ROUND,
            natural_ids=identity,
        )


def _event_row(parts: Mapping[str, list[dict[str, Any]]], event: str) -> dict[str, Any]:
    """Return the event-level row of one event.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The enablement sections.
        event (str): The event id whose row to select.

    Returns:
        dict[str, Any]: The row, or an empty dict when the event has none.
    """
    rows = rows_for_event(parts.get(SECTION_EVENT) or [], event)
    return rows[0] if rows else {}


def _event_field(parts: Mapping[str, list[dict[str, Any]]], event: str, field: str) -> str:
    """Return one string field off the event-level row.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The enablement sections.
        event (str): The event id whose row to read.
        field (str): The field wanted.

    Returns:
        str: The value, or ``""`` when absent.
    """
    return str(_event_row(parts, event).get(field) or "")


def assemble_enablement_rounds(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> list[dict[str, Any]]:
    """Assemble every round belonging to one enablement event.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The enablement sections as
            read back from the spool.
        event (str): The event id whose rows to select.

    Returns:
        list[dict[str, Any]]: The rounds, ordered by when they were dispatched.
            A round still in flight carries its start and no verdict.
    """
    rows = sort_rows(
        rows_for_event(parts.get(SECTION_ROUND) or [], event),
        keys=("started_at", "specialist_task_id"),
    )
    return wire_rows(rows)


def assemble_enablement_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble one enablement event's ``ext`` out of its recorded rows.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The enablement sections as
            read back from the spool.
        event (str): The event id to assemble.

    Returns:
        tuple[dict[str, Any], str]: The ``ext`` payload and the status derived
            from it.
    """
    row = _event_row(parts, event)
    rounds = assemble_enablement_rounds(parts, event=event)
    return {
        "mode": str(row.get("mode") or ""),
        "origin": str(row.get("origin") or ""),
        "attempts": len(rounds),
        "failure_kind": _last_failure_kind(rounds),
        "failure": None if bool(row.get("succeeded")) else _failure(row, rounds),
        "rounds": rounds,
    }, _derived_status(row, rounds)


def _last_failure_kind(rounds: list[dict[str, Any]]) -> str:
    """Return the gap the effort was aimed at when it stopped.

    Args:
        rounds (list[dict[str, Any]]): The assembled rounds, in order.

    Returns:
        str: The last round's classified gap, or ``""`` when there is none. A
            serial enablement retargets every round, so the last one names
            where the effort actually stood.
    """
    for round_row in reversed(rounds):
        kind = str(round_row.get("failure_kind") or "")
        if kind:
            return kind
    return ""


def _failure(row: Mapping[str, Any], rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Build the failure block of an effort that did not end runnable.

    Args:
        row (Mapping[str, Any]): The event-level row.
        rounds (list[dict[str, Any]]): The assembled rounds, in order.

    Returns:
        dict[str, Any] | None: The failure block, or ``None`` while the effort
            has not reported a verdict. The residual signature comes from the
            last round: it describes the gap the boot still stops at, which is
            what a later session needs in order to decide whether to retry.
    """
    if not row.get("end_time"):
        return None
    residual = _as_dict(rounds[-1].get("after_signature")) if rounds else {}
    return {
        **_failure_row(
            phase=EVENT_TYPE,
            error_class=str(residual.get("kind") or ""),
            message=residual.get("raw_excerpt") or "",
        ),
        "offending_file": str(residual.get("offending_file") or ""),
        "stop_reason": str(row.get("stop_reason") or ""),
    }


def _derived_status(row: Mapping[str, Any], rounds: list[dict[str, Any]]) -> str:
    """Decide the status the event closes on.

    Args:
        row (Mapping[str, Any]): The event-level row.
        rounds (list[dict[str, Any]]): The assembled rounds.

    Returns:
        str: ``succeeded`` when the combo ended up runnable, ``skipped`` when
            the lane was admitted but never dispatched a round, ``running``
            until the effort reports a verdict, and ``failed`` otherwise --
            including the serial effort that advanced through several gaps and
            then ran out, whose partial progress is in the rounds and not in
            the status.
    """
    if bool(row.get("succeeded")):
        return "succeeded"
    if not rounds:
        return "skipped"
    if not row.get("end_time"):
        return "running"
    return "failed"


def make_enablement_recorder(sink: RecordSink | None) -> EnablementEventRecorder | None:
    """Build a recorder, or ``None`` when one cannot be constructed.

    An absent sink -- what a caller with no session bound has -- declines
    rather than raising.

    Args:
        sink (RecordSink | None): Where the rows go, or ``None`` to decline.

    Returns:
        EnablementEventRecorder | None: The recorder, or ``None``.
    """
    if sink is None:
        return None
    return EnablementEventRecorder(sink)
