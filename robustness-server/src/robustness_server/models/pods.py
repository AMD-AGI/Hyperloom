"""Pod identity and session ↔ pod assignment models.

``PodAssignment`` represents one row in ``session_pod_assignment``. The
table is append-mostly: on assignment open we INSERT a row with
``t_end=NULL``; on close we UPDATE ``t_end``. This shape preserves the
full timeline of a session's pod set without losing intermediate
states.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PodRole(StrEnum):
    BRAIN = "brain"
    HANDS_GPU = "hands_gpu"
    HANDS_CPU = "hands_cpu"
    OTHER = "other"


class PodAssignmentSource(StrEnum):
    """Which signal first observed the (session, pod) pairing.

    Stored on each row so debugging multi-source races (e.g., NATS event
    arrived seconds before the workload poller propagated the labels)
    is cheap.
    """

    NATS_KV = "nats_kv"
    NATS_EVENT = "nats_event"
    WORKLOAD_RECONCILE = "workload_reconcile"


class PodRef(BaseModel):
    """Minimal pod identity; matches the Robust pod-metrics endpoint.

    ``pod_uid`` is best-effort: NATS events sometimes omit it. The
    Robust pod-metrics catalogue keys on ``(namespace, name)``, not
    UID, so missing UID is non-fatal for queries.
    """

    model_config = ConfigDict(extra="forbid")

    namespace: str = Field(min_length=1)
    name: str = Field(min_length=1)
    pod_uid: str | None = None


class PodAssignment(BaseModel):
    """One row in ``session_pod_assignment``.

    ``assignment_id`` is set by Postgres on INSERT and surfaced back so
    callers (notably the KV watcher's close path) can target an exact
    row without needing a composite key.
    """

    model_config = ConfigDict(extra="forbid")

    assignment_id: int | None = None
    session_id: str = Field(min_length=1)
    pod: PodRef
    role: PodRole
    source: PodAssignmentSource
    t_start: datetime
    t_end: datetime | None = None
    last_seen_at: datetime
