# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX workload timeline, read from aiperf's own per-request export.

KV occupancy is engine-wide: it says the pool filled, never which work filled it. Answering "what was running when it
filled" needs the workload's own timeline, and aiperf already writes one. ``profile_export.jsonl`` carries one record
per request with epoch-nanosecond stamps and the identifiers that reconstruct the hierarchy above it -- session,
conversation, turn index, request id -- at the default ``--export-level records``. Nothing has to be asked of aiperf
and no flag has to be added; the file was simply never read.

This module turns those records into two things:

* an event stream (``phase`` / ``trajectory`` / ``turn`` / ``request`` start and end, plus ``first_token``), written
  beside the KV artifact so a consumer can join the two on time;
* an exact in-flight count per KV scrape window, folded into the KV rows themselves, so the common question -- which
  trajectories were live when the pool spiked -- is answerable without the join.

The same rule as everywhere else in this collection path: nothing here may raise, and nothing missing becomes zero. A
round without the export is a round whose KV metrics are still worth having.
"""

from __future__ import annotations

import heapq
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger(__name__)


__all__ = [
    "TIMELINE_ARTIFACT_NAME",
    "RequestRecord",
    "build_events",
    "correlate_rows",
    "find_profile_export",
    "parse_profile_export",
    "write_timeline",
]

#: Artifact written beside ``kv_metrics.json``.
TIMELINE_ARTIFACT_NAME = "agentx_timeline.jsonl"

#: Where aiperf's per-request export lands, relative to the round's own directory. Mirrors how the progress-API address
#: and the aiperf log are resolved, since an AgentX round may nest its benchmark one level down.
_EXPORT_RELPATHS = ("aiperf_artifacts/profile_export.jsonl", "*/aiperf_artifacts/profile_export.jsonl")

#: Cap on request-level events written out. A three-hour round at high concurrency produces well over a hundred
#: thousand requests, and three events each would outweigh every other artifact in the session bundle combined. Over
#: the cap the request layer is stride-sampled; the phase, trajectory and turn layers are always complete, and the
#: in-flight counts folded into the KV rows are computed from every record before any sampling.
_MAX_REQUEST_EVENTS = 60000

#: Cap on trajectory ids recorded against one KV row. Concurrency is tens, not thousands, so this only guards against a
#: pathological run; the count beside it is never capped.
_MAX_ROW_TRAJECTORIES = 32


@dataclass(frozen=True)
class RequestRecord:
    """One request from aiperf's export, reduced to what a timeline needs.

    Times are epoch seconds. aiperf reports nanoseconds, which is the same clock the progress API stamps phases on, so
    the two line up without conversion beyond scale.
    """

    request_id: str
    trajectory_id: str
    conversation_id: str
    turn_index: int
    start: float
    end: float
    first_token: float | None
    phase: str
    ok: bool

    def overlaps(self, window_start: float, window_end: float) -> bool:
        """Whether this request was in flight at any point in a window."""
        return self.start <= window_end and self.end >= window_start


def find_profile_export(workspace: Any) -> Path | None:
    """Locate aiperf's per-request export under a round's directory."""
    try:
        root = Path(workspace)
        for pattern in _EXPORT_RELPATHS:
            for candidate in sorted(root.glob(pattern)):
                if candidate.is_file():
                    return candidate
    except OSError:
        return None
    return None


def _seconds(raw: Any) -> float | None:
    """Epoch nanoseconds as epoch seconds, or ``None`` when unusable."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        return None
    return float(raw) / 1e9


def _metric(metrics: Any, name: str) -> float | None:
    """One metric's value out of aiperf's ``{"value": x, "unit": u}`` wrapper."""
    if not isinstance(metrics, dict):
        return None
    entry = metrics.get(name)
    if isinstance(entry, dict):
        entry = entry.get("value")
    if isinstance(entry, bool) or not isinstance(entry, (int, float)):
        return None
    return float(entry)


def _record_from(payload: dict[str, Any]) -> RequestRecord | None:
    """Build one record, or ``None`` when the line cannot be used."""
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    start = _seconds(metadata.get("request_start_ns"))
    end = _seconds(metadata.get("request_end_ns"))
    if start is None or end is None or end < start:
        return None
    # The live session is the trajectory: it is what stays constant across the turns of one multi-turn conversation.
    # ``conversation_id`` names the dataset entry being replayed, which can recur across trajectories, so it is carried
    # alongside rather than used as the identity.
    trajectory = metadata.get("x_correlation_id") or metadata.get("conversation_id") or ""
    turn = metadata.get("turn_index")
    ttft_ms = _metric(payload.get("metrics"), "time_to_first_token")
    # Derived from TTFT rather than from ``request_ack_ns``: the ack is when the server accepted the request, which is
    # not the same instant as the first token, and conflating them would misplace exactly the event a reader would use
    # to line a decode-phase KV rise up against.
    first_token = start + ttft_ms / 1000.0 if ttft_ms is not None else None
    return RequestRecord(
        request_id=str(metadata.get("x_request_id") or ""),
        trajectory_id=str(trajectory),
        conversation_id=str(metadata.get("conversation_id") or ""),
        turn_index=int(turn) if isinstance(turn, int) and not isinstance(turn, bool) else 0,
        start=start,
        end=end,
        first_token=first_token,
        phase=str(metadata.get("benchmark_phase") or ""),
        ok=payload.get("error") in (None, {}),
    )


def parse_profile_export(path: Path) -> list[RequestRecord]:
    """Read every usable record from aiperf's JSONL export, sorted by start time.

    Line-by-line and lenient: the export is written while the round is ending, and one truncated last line must not
    cost the timeline everything before it.
    """
    records: list[RequestRecord] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                if isinstance(payload, dict):
                    record = _record_from(payload)
                    if record is not None:
                        records.append(record)
    except OSError as exc:
        log.debug("agentx_timeline: could not read %s (%s)", path, exc)
        return []
    records.sort(key=lambda r: r.start)
    return records


def _span_events(
    records: Iterable[RequestRecord],
    kind: str,
    key: Any,
    fields: Any,
) -> list[dict[str, Any]]:
    """Start/end events for a grouping of requests, spanning first start to last end."""
    spans: dict[Any, list[float]] = {}
    identity: dict[Any, dict[str, Any]] = {}
    for record in records:
        group = key(record)
        bounds = spans.get(group)
        if bounds is None:
            spans[group] = [record.start, record.end]
            identity[group] = fields(record)
        else:
            bounds[0] = min(bounds[0], record.start)
            bounds[1] = max(bounds[1], record.end)
    events: list[dict[str, Any]] = []
    for group, (start, end) in spans.items():
        events.append({"event": f"{kind}_start", "ts": round(start, 6), **identity[group]})
        events.append({"event": f"{kind}_end", "ts": round(end, 6), **identity[group]})
    return events


def build_events(
    records: list[RequestRecord],
    phase_bounds: dict[str, tuple[float, float | None]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Turn records into a time-ordered event stream, with a summary of what it covers.

    Phase events come from the caller's authoritative bounds -- aiperf's progress API -- rather than from the records,
    whose ``benchmark_phase`` currently reads ``profiling`` for everything it exports.
    """
    events: list[dict[str, Any]] = []
    for phase, (start, end) in (phase_bounds or {}).items():
        events.append({"event": "phase_start", "ts": round(start, 6), "phase": phase})
        if end is not None:
            events.append({"event": "phase_end", "ts": round(end, 6), "phase": phase})

    events.extend(
        _span_events(
            records,
            "trajectory",
            lambda r: r.trajectory_id,
            lambda r: {"trajectory_id": r.trajectory_id, "conversation_id": r.conversation_id},
        )
    )
    events.extend(
        _span_events(
            records,
            "turn",
            lambda r: (r.trajectory_id, r.turn_index),
            lambda r: {"trajectory_id": r.trajectory_id, "turn_index": r.turn_index},
        )
    )

    kept = records
    stride = 1
    if len(records) > _MAX_REQUEST_EVENTS:
        stride = -(-len(records) // _MAX_REQUEST_EVENTS)
        kept = records[::stride]
    for record in kept:
        common = {
            "request_id": record.request_id,
            "trajectory_id": record.trajectory_id,
            "turn_index": record.turn_index,
        }
        events.append({"event": "request_start", "ts": round(record.start, 6), **common})
        if record.first_token is not None:
            events.append({"event": "first_token", "ts": round(record.first_token, 6), **common})
        events.append({"event": "request_end", "ts": round(record.end, 6), "ok": record.ok, **common})

    events.sort(key=lambda e: e["ts"])
    summary = {
        "requests": len(records),
        "trajectories": len({r.trajectory_id for r in records}),
        "turns": len({(r.trajectory_id, r.turn_index) for r in records}),
        "events": len(events),
        # Stated rather than implied: a reader counting request_start events must be able to tell a sampled stream from
        # a complete one, since the two support very different conclusions.
        "request_events_complete": stride == 1,
        "request_event_stride": stride,
    }
    return events, summary


def write_timeline(path: Path, events: Iterable[dict[str, Any]]) -> bool:
    """Write the event stream as JSONL. Returns whether it landed.

    Owner-only, matching the KV artifact it sits beside: ``atomic_write_text`` masks every written payload to ``0o700``
    on purpose, and a sibling artifact from the same collector should not be the one file in the directory that is
    world-readable.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        os.chmod(path, 0o600)
    except (OSError, TypeError) as exc:
        log.debug("agentx_timeline: could not write %s (%s)", path, exc)
        return False
    return True


def _windows(rows: list[dict[str, Any]]) -> Iterator[tuple[int, float, float]]:
    """Each row's scrape window as ``(index, start, end)``, skipping rows without one."""
    for index, row in enumerate(rows):
        start = row.get("scrape_start_unix")
        end = row.get("scrape_end_unix", start)
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            yield index, float(start), float(end)


def correlate_rows(rows: list[dict[str, Any]], records: list[RequestRecord]) -> int:
    """Fold the workload in flight during each scrape into that scrape's row.

    A sweep rather than a scan per row: at a few thousand rows and a few hundred thousand records the naive form is
    hundreds of millions of comparisons, inside the path that finishes a benchmark round.

    Counts are exact -- computed from every record, before the event stream is ever sampled. Returns how many rows were
    annotated.
    """
    if not rows or not records:
        return 0
    windows = sorted(_windows(rows), key=lambda w: w[1])
    if not windows:
        return 0
    ordered = sorted(records, key=lambda r: r.start)

    annotated = 0
    cursor = 0
    # Requests still open, keyed by end time so the earliest to finish leaves first.
    active: list[tuple[float, int]] = []
    for index, start, end in windows:
        while cursor < len(ordered) and ordered[cursor].start <= end:
            record = ordered[cursor]
            heapq.heappush(active, (record.end, cursor))
            cursor += 1
        while active and active[0][0] < start:
            heapq.heappop(active)
        in_flight = [ordered[i] for _, i in active if ordered[i].overlaps(start, end)]
        if not in_flight:
            continue
        trajectories = sorted({r.trajectory_id for r in in_flight if r.trajectory_id})
        workload = rows[index].setdefault("workload", {})
        workload["in_flight"] = {
            "requests": len(in_flight),
            "trajectories": len(trajectories),
            "turns": len({(r.trajectory_id, r.turn_index) for r in in_flight}),
            # Ids, not just a count: "which trajectory was live when the pool spiked" is the question this exists for,
            # and a count cannot answer it.
            "trajectory_ids": trajectories[:_MAX_ROW_TRAJECTORIES],
        }
        annotated += 1
    return annotated
