# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``warm_start`` event: the T0 Recipe KB lookup, recorded live.

At session start T0 asks the KB whether anyone has already optimized this
workload, and either anchors on what it finds or starts cold.

Whether the lookup ran and what it found are two facts and two fields:
``status`` is the lookup's own outcome, ``match_status`` is what it found. The
split is load-bearing, because on a cold KB T0 stamps its own anchor row before
searching, then matches that row and demotes it to ``seed_only`` -- a status
that graded the finding would report every first-ever session as degraded.

Reads are recorded one row each, and only inside T0's own lookup window: the
recipe audit log also carries KB writes and the mid-session reads from
``_kb_amend_recipe``, and neither belongs to the anchor's tally.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

from .event_fields import (
    as_dict as _as_dict,
    bool_or_none as _bool_or_none,
    float_or_none as _float_or_none,
    now_iso_seconds as _now,
    text_or_none as _text_or_none,
)
from .event_ids import event_id
from .event_rows import rows_for_event, sort_rows, wire_rows
from .event_sink import RecordSink, make_sink
from .event_timeline import finish_event, open_event
from .recorder_warnings import note_failure

log = logging.getLogger(__name__)

EVENT_TYPE = "warm_start"
EVENT_KIND = "warm_start"

#: The component segment of a warm-start event id.
EVENT_COMPONENT = "warm_start"

#: The phase segment. A literal rather than a read of ``state.phase``: T0 runs
#: before the phase machine settles, and on a resume it re-runs from whatever
#: phase the session came back in, which would split one anchor into two.
EVENT_PHASE = "prelude"

PRODUCER = "orchestrator"

#: The event-level section, one fragment per event, holding the request, what
#: was matched, and the timeline sequence the open and close writes share.
SECTION_EVENT = "warm_start_event"

#: One row per KB read T0 made, in the order they were served.
SECTION_READ = "warm_start_read"

# What the lookup found, as T0's own ``warm_start_context`` states it.
# ``seed_only`` is a record that was found and cannot be executed, which is
# neither a match a reader can act on nor the absence of one.
MATCH_HIT = "hit"
MATCH_SEED_ONLY = "seed_only"
MATCH_MISS = "miss"

# The lookup's own outcome, and only that. Every finding -- a hit, a
# ``seed_only``, a miss -- is a lookup that ran and answered, so all three are
# ``succeeded`` and the finding is read off ``match_status``.
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"

#: Match statuses that mean the event should carry a ``matched`` block.
MATCHED_STATUSES = frozenset({MATCH_HIT, MATCH_SEED_ONLY})

#: The tier that means the match was on this exact workload identity rather
#: than a relaxed neighbourhood of it.
_EXACT_TIER = "exact"

#: The recorder whose lookup is in flight, for the audit hook to attribute reads
#: to. A context variable because the hook runs on whatever task the KB op was
#: issued from, and it holds the recorder rather than an id so a read arriving
#: after the event closed is dropped even if the variable outlived the lookup.
_ACTIVE: ContextVar[WarmStartEventRecorder | None] = ContextVar("warm_start_active", default=None)


def warm_start_event_id(macro_cycle: Any = 0) -> str:
    """Build the T0 lookup's event id, ``prelude:{macro_cycle}:warm_start``.

    ``macro_cycle`` is ``0`` for the anchor and non-zero only for a resumed
    session that re-anchors.

    Raises:
        ValueError: If either segment is malformed.
    """
    return event_id(EVENT_PHASE, macro_cycle, EVENT_COMPONENT)


def record_read(session_dir: Any, audit_event: Mapping[str, Any]) -> None:
    """Record one KB read served while a T0 lookup is in flight. Never raises.

    Called from the recipe audit hook, which sees writes as well as reads, and
    reads from every seam that consults the KB. Both are filtered here: a write
    is not a read, and a read served while no lookup is open belongs to whatever
    else was asking, which keeps ``_kb_amend_recipe``'s mid-session traffic out
    of the anchor's tally. A falsy ``session_dir`` is a no-op.
    """
    active = _ACTIVE.get()
    if not session_dir or active is None or not active.claims(session_dir):
        return
    if not isinstance(audit_event, Mapping) or str(audit_event.get("op") or "") != "read":
        return
    try:
        from .recorder import recorder_for

        recorder_for(session_dir, producer=PRODUCER).record_item(
            SECTION_READ,
            _read_row(active.event_id, active.next_read_ordinal(), audit_event),
        )
    except Exception as exc:  # noqa: BLE001 — a read's record must not cost the read
        note_failure(section="warm_start_event", error=exc, detail="record warm_start read failed")


def _read_row(event: str, ordinal: int, audit_event: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one audit read event into a wire row."""
    request = _as_dict(audit_event.get("request"))
    result = _as_dict(audit_event.get("result"))
    return {
        "event_id": event,
        # Service order, carried explicitly: the cascade issues its reads well
        # inside one second, and the order is what shows the exact-identity
        # probe missing before the degradation ladder was walked.
        "ordinal": ordinal,
        "ts": _now(),
        "method": str(audit_event.get("method") or ""),
        "resolution": str(audit_event.get("resolution") or ""),
        "remote": str(audit_event.get("remote") or ""),
        "backend": str(audit_event.get("backend") or ""),
        "hit": bool(audit_event.get("hit")),
        "candidates": int(audit_event.get("candidates") or 0),
        "requested_canonical_id": _text_or_none(request.get("canonical_id")),
        # Present only when the read returned a row; a miss has no result to
        # describe, and an empty one would read as a row with nothing in it.
        "matched_canonical_id": _text_or_none(result.get("canonical_id")),
        "exact": bool(result.get("exact")) if result else None,
        "best_throughput": _float_or_none(result.get("best_throughput")),
        "best_config_nonempty": bool(result.get("best_config_nonempty")) if result else None,
    }


class WarmStartEventRecorder:
    """Records one T0 Recipe KB lookup as it happens."""

    def __init__(
        self,
        sink: RecordSink,
        *,
        requested_canonical_id: str,
        scope: Mapping[str, Any] | None,
        start_time: str,
        session: Any = None,
    ):
        self._sink = sink
        self._session = str(session or "")
        self._sequence: int | None = None
        self._closed = False
        self._token: Token[WarmStartEventRecorder | None] | None = None
        self._reads = 0
        self._start_time = start_time
        self._t0 = time.monotonic()
        self._request = {
            # The identity T0 actually queried. Its hardware dimension is
            # topology-aware and resolved from the runtime environment, so
            # rebuilding it at export could disagree with what the run asked.
            "canonical_id": str(requested_canonical_id or ""),
            "scope": dict(scope or {}),
        }
        # Recorded onto the fragment as well as onto the open shell, because
        # assembly rebuilds ``ext`` from the fragment and would otherwise close
        # the event having dropped the identity the open shell carried.
        self._sink.record(SECTION_EVENT, {"request": dict(self._request)})

    @property
    def event_id(self) -> str:
        """str: The event every row this recorder writes is tagged with."""
        return self._sink.event_id

    @property
    def closed(self) -> bool:
        """bool: Whether the lookup has settled."""
        return self._closed

    def next_read_ordinal(self) -> int:
        """Claim the next read's position in service order."""
        self._reads += 1
        return self._reads

    def claims(self, session_dir: Any) -> bool:
        """Whether this recorder is the one a read in ``session_dir`` belongs to.

        A lookup that raised before settling leaves its window open, so the
        session it was opened in is checked rather than assumed: a stale window
        must not attribute a later session's reads to an anchor that never
        finished.
        """
        if self._closed:
            return False
        if not self._session:
            return True
        try:
            return Path(session_dir).resolve() == Path(self._session).resolve()
        except (OSError, TypeError, ValueError):  # noqa: BLE001 — unusable path
            return False

    def begin(self) -> None:
        """Put the event on the timeline before the KB is consulted.

        The request rides on the open shell so a session killed during its
        lookup is readable as a lookup of a named identity rather than as an
        anonymous event that never finished. Opening also claims the reads the
        KB is about to serve: the event's own open interval is the attribution
        window, so a lookup that dies without settling keeps its reads, which
        is what says how far it got.
        """
        self._sequence = open_event(
            event_type=EVENT_TYPE,
            event=self.event_id,
            event_section=SECTION_EVENT,
            producer=PRODUCER,
            kind=EVENT_KIND,
            start_time=self._start_time,
            ext={"request": dict(self._request)},
        )
        self._token = _ACTIVE.set(self)

    def finish(
        self,
        *,
        match_status: str,
        matched: Mapping[str, Any] | None = None,
        error: str = "",
    ) -> None:
        """Settle the lookup on one of ``hit`` / ``seed_only`` / ``miss``.

        ``error`` carries the exception class name when the lookup itself
        failed, which is not the same as having looked and found nothing.
        """
        if self._closed:
            return
        self._closed = True
        self._release()
        end_time = _now()
        found = str(match_status or "").strip().lower()
        status = _status_for(found, error=error)
        payload: dict[str, Any] = {
            "status": status,
            "match_status": found,
            "end_time": end_time,
            "duration_sec": round(time.monotonic() - self._t0, 3),
        }
        if matched:
            payload["matched"] = dict(matched)
        if error:
            payload["failure"] = {"error_class": str(error)}
        self._sink.record(SECTION_EVENT, payload)

        from .assembler import warm_start_event_parts

        ext, derived = assemble_warm_start_ext(warm_start_event_parts(self.event_id), event=self.event_id)
        finish_event(
            event_type=EVENT_TYPE,
            event=self.event_id,
            sequence=self._sequence,
            status=derived or status,
            ext=ext,
            kind=EVENT_KIND,
            start_time=self._start_time,
            end_time=end_time,
        )

    def _release(self) -> None:
        """Stop claiming reads. Tolerant of a token set on another context."""
        token, self._token = self._token, None
        if token is None:
            return
        try:
            _ACTIVE.reset(token)
        except ValueError:  # noqa: BLE001 — settled from a different context
            _ACTIVE.set(None)


def _status_for(match_status: str, *, error: str = "") -> str:
    """The lookup's own outcome, which is independent of what it found.

    A miss is ``succeeded``: the KB was asked and it answered.
    """
    return STATUS_FAILED if error else STATUS_SUCCEEDED


def make_warm_start_recorder(
    *,
    macro_cycle: Any = 0,
    requested_canonical_id: str,
    scope: Mapping[str, Any] | None = None,
    start_time: str = "",
) -> WarmStartEventRecorder | None:
    """Open the T0 lookup's event, or ``None`` when it cannot be recorded.

    ``None`` means no session is bound or the open write failed; ``start_time``
    defaults to now. Recording is best-effort: the anchor outranks its record.
    """
    try:
        from ...session.session_binding import bound_session

        recorder = WarmStartEventRecorder(
            make_sink(warm_start_event_id(macro_cycle), producer=PRODUCER),
            requested_canonical_id=requested_canonical_id,
            scope=scope,
            start_time=start_time or _now(),
            session=bound_session(),
        )
        recorder.begin()
        return recorder
    except Exception as exc:  # noqa: BLE001
        note_failure(section="warm_start_event", error=exc, detail="open warm_start event failed")
        return None


def assemble_warm_start_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble one warm-start event's ``ext`` out of its recorded rows.

    ``parts`` is the spool's sections keyed by name, and rows of every event
    other than ``event`` are ignored. The returned status is empty when no
    write has closed the event, which leaves the caller's own reading standing.
    """
    event_rows = rows_for_event(parts.get(SECTION_EVENT) or [], event)
    header = event_rows[0] if event_rows else {}
    reads = wire_rows(
        sort_rows(rows_for_event(parts.get(SECTION_READ) or [], event), keys=("ordinal", "ts")),
        drop=("event_id", "ordinal"),
    )
    ext: dict[str, Any] = {
        "request": _as_dict(header.get("request")),
        "match_status": str(header.get("match_status") or ""),
        "matched": _as_dict(header.get("matched")) or None,
        "reads": _reads_block(reads),
    }
    failure = _as_dict(header.get("failure"))
    if failure:
        ext["failure"] = failure
    return ext, str(header.get("status") or "")


def _reads_block(reads: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Summarize T0's reads, keeping every row it summarizes.

    The tallies are what a reader scans first, but they are derived from rows
    that are themselves published, so a number that looks wrong can be checked
    against the reads it came from.
    """
    if not reads:
        return None
    by_resolution: dict[str, int] = {}
    by_method: dict[str, int] = {}
    for row in reads:
        resolution = str(row.get("resolution") or "unknown")
        by_resolution[resolution] = by_resolution.get(resolution, 0) + 1
        method = str(row.get("method") or "unknown")
        by_method[method] = by_method.get(method, 0) + 1
    return {
        "count": len(reads),
        "hits": sum(1 for row in reads if row.get("hit")),
        "by_resolution": by_resolution,
        "by_method": by_method,
        "rows": reads,
    }


def matched_block(
    *,
    tier: str,
    confidence: Any,
    source: str,
    canonical_id: str,
    recipe: Mapping[str, Any] | None,
    expected_gain_pct: Any = None,
    lessons: Any = None,
    pitfalls: Any = None,
) -> dict[str, Any]:
    """Build the ``matched`` block from the record T0 anchored on."""
    row = _as_dict(recipe)
    tier_text = str(tier or "").strip()
    block: dict[str, Any] = {
        "match_type": _EXACT_TIER if tier_text.lower() == _EXACT_TIER else "degraded",
        "tier": tier_text,
        "confidence": _float_or_none(confidence),
        "source": str(source or ""),
        "canonical_id": str(canonical_id or ""),
        "optimized_throughput": _float_or_none(row.get("best_throughput")),
        "validated_gain_pct": _float_or_none(row.get("validated_gain_pct")),
        "expected_gain_pct": _float_or_none(expected_gain_pct),
        "replayable": _bool_or_none(row.get("replayable")),
        "replay_disabled_reason": _text_or_none(row.get("replay_disabled_reason")),
        "replay_material_available": _bool_or_none(row.get("replay_material_available")),
        "view_source": _text_or_none(row.get("view_source")),
        "experience": {
            "lessons_count": len(lessons) if isinstance(lessons, list) else 0,
            "pitfalls_count": len(pitfalls) if isinstance(pitfalls, list) else 0,
        },
    }
    shape = _as_dict(row.get("workload_shape"))
    if shape:
        block["scope"] = {key: shape.get(key) for key in ("tp", "conc", "isl", "osl")}
    origin = _origin(row)
    if origin:
        block["origin"] = origin
    return block


def _origin(recipe: Mapping[str, Any]) -> dict[str, Any] | None:
    """Which session wrote the matched record, and what it gained."""
    provenance = _as_dict(recipe.get("provenance"))
    session_id = _text_or_none(recipe.get("remote_session_id")) or _text_or_none(provenance.get("session_id"))
    sessions = [row for row in (recipe.get("sessions") or []) if isinstance(row, Mapping)]
    gain = _float_or_none(sessions[0].get("gain_pct")) if sessions else None
    if not session_id and gain is None:
        return None
    return {"session_id": session_id, "gain_pct": gain}


__all__ = [
    "EVENT_COMPONENT",
    "EVENT_KIND",
    "EVENT_PHASE",
    "EVENT_TYPE",
    "MATCHED_STATUSES",
    "MATCH_HIT",
    "MATCH_MISS",
    "MATCH_SEED_ONLY",
    "PRODUCER",
    "SECTION_EVENT",
    "SECTION_READ",
    "STATUS_FAILED",
    "STATUS_SUCCEEDED",
    "WarmStartEventRecorder",
    "assemble_warm_start_ext",
    "make_warm_start_recorder",
    "matched_block",
    "record_read",
    "warm_start_event_id",
]
