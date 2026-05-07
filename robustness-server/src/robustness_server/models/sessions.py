"""Session-level data models.

A ``Session`` row tracks the lifecycle of one Claw session as observed
by this server. State transitions are computed from NATS events:

* ``ACTIVE`` once any event for the session has been seen and no
  terminal event (complete/failed/aborted) has arrived;
* ``COMPLETED`` after a successful terminal event;
* ``FAILED`` after a failed terminal event;
* ``ABORTED`` for explicit cancel/abort terminals.

The reconciler does not invent state — it only updates ``last_event_at``
and ``t_end`` based on events the consumer feeds in.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SessionState(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class Session(BaseModel):
    """Canonical session view returned by the API and stored in PG."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    user_id: str | None = None
    plugin_id: str | None = None
    t_start: datetime
    t_end: datetime | None = None
    final_state: SessionState | None = None
    last_event_at: datetime
