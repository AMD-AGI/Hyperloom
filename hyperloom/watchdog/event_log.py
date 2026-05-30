"""Event log — append-only JSONL event stream for the session.

All significant events (benchmarks, gate results, agent failures, config changes,
server crashes) are appended here. The watchdog scanner reads from this file.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Event:
    """A single event in the event log."""

    event_id: str
    source: str
    event_type: str
    severity: str  # "info" | "warning" | "error" | "critical"
    timestamp: str
    promising: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _event_log_path(session_dir: str) -> Path:
    return Path(session_dir) / "event_log.jsonl"


def append_event(
    session_dir: str,
    source: str,
    event_type: str,
    severity: str = "info",
    promising: bool = False,
    details: dict[str, Any] | None = None,
) -> str:
    """Append an event to the session event log. Returns the event_id."""
    event_id = str(uuid.uuid4())[:8]
    event = {
        "event_id": event_id,
        "source": source,
        "type": event_type,
        "severity": severity,
        "promising": promising,
        "details": details or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
    }

    path = _event_log_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")

    return event_id


def read_events(session_dir: str, limit: int = 0) -> list[dict[str, Any]]:
    """Read all events from the log. If limit > 0, return only the last N."""
    path = _event_log_path(session_dir)
    if not path.exists():
        return []

    events = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if limit > 0:
        return events[-limit:]
    return events


def read_new_events(session_dir: str, after_offset: int) -> tuple[list[dict], int]:
    """Read events added after byte offset. Returns (new_events, new_offset).

    Use this for incremental polling:
        offset = 0
        while True:
            events, offset = read_new_events(session_dir, offset)
            process(events)
            sleep(poll_interval)
    """
    path = _event_log_path(session_dir)
    if not path.exists():
        return [], 0

    file_size = path.stat().st_size
    if file_size <= after_offset:
        return [], after_offset

    new_events = []
    with open(path) as f:
        f.seek(after_offset)
        for line in f:
            line = line.strip()
            if line:
                try:
                    new_events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return new_events, file_size


def event_count(session_dir: str) -> int:
    """Return total number of events in the log."""
    path = _event_log_path(session_dir)
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())
