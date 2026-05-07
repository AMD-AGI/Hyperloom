"""Pydantic data models shared across services and the HTTP layer.

The repository layer translates these to/from asyncpg rows (raw dicts);
the API layer exposes them directly. Keeping the dataclass set thin
avoids accidental coupling between persistence and transport schemas.
"""

from __future__ import annotations

from .events import (
    EventEnvelope,
    EventKind,
    KVAssignmentChange,
    parse_event_envelope,
)
from .pods import PodAssignment, PodAssignmentSource, PodRef, PodRole
from .sessions import Session, SessionState

__all__ = [
    "EventEnvelope",
    "EventKind",
    "KVAssignmentChange",
    "PodAssignment",
    "PodAssignmentSource",
    "PodRef",
    "PodRole",
    "Session",
    "SessionState",
    "parse_event_envelope",
]
