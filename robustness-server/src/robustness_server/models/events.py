"""NATS event models and parsing.

Claw's ``NatsEmitter`` publishes envelopes on subject ``events.<sid>``
(JetStream stream ``PRIMUS_CLAW_EVENTS``). The exact payload shape is
defined by Claw and intentionally treated as untyped JSON outside the
small set of canonical fields we depend on:

``{
    "type":      "<event-kind>",
    "sessionId": "<session-id>",
    "timestamp": "<ISO-8601 UTC>",
    "podName":   "<pod-name>"        # optional
    "namespace": "<pod-namespace>"   # optional
    "pluginId":  "<plugin>"          # optional
    ...                              # passthrough
}``

The full body is stored in ``session_events.body`` so future analyses
can mine fields we don't model up front.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventKind(StrEnum):
    """Canonical event kinds we reason about explicitly.

    Anything outside this set is still persisted, but does not move the
    session state machine. Adding a new terminal kind is a one-line
    change in ``TERMINAL_KINDS``.
    """

    SESSION_START = "session_start"
    SESSION_PROGRESS = "session_progress"
    EXEC_COMPLETE = "exec_complete"
    EXEC_FAILED = "exec_failed"
    SANDBOX_CREATE = "sandbox_create"
    SANDBOX_STATUS = "sandbox_status"
    SANDBOX_DELETE = "sandbox_delete"
    OTHER = "other"


# Event kinds whose arrival closes a session window. Keep in sync with
# the Claw event taxonomy; surfacing them here in one place lets the
# session router decide ``final_state`` declaratively.
TERMINAL_KINDS: dict[EventKind, str] = {
    EventKind.EXEC_COMPLETE: "completed",
    EventKind.EXEC_FAILED: "failed",
}


class EventEnvelope(BaseModel):
    """Decoded NATS event we persist + react to.

    ``body`` holds the full original JSON so the audit log is lossless.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    subject: str
    kind: EventKind
    raw_type: str
    occurred_at: datetime
    received_at: datetime
    pod_name: str | None = None
    pod_namespace: str | None = None
    plugin_id: str | None = None
    body: dict[str, Any]


class KVAssignmentChange(BaseModel):
    """Synthesised event emitted by the BRAIN_REGISTRY watcher.

    The KV bucket carries one key per session (``lock.<sessionId>``)
    whose value names the brain pod currently servicing that session.
    PUT/DELETE/expire all collapse to this one shape; consumers branch
    on ``opening``.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    pod_name: str
    pod_namespace: str
    opening: bool
    observed_at: datetime


def _coerce_dt(value: Any) -> datetime:
    """Permissive ISO-8601 / epoch parser.

    Accepts the four shapes Claw has been seen to publish: aware
    ISO-8601 strings, naive ISO-8601 (assumed UTC), int/float epoch
    seconds, and ``datetime`` instances forwarded by tests. Falls back
    to "now (UTC)" rather than raising so a malformed timestamp does
    not lose the entire event — the audit row still gets persisted.
    """

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(tz=timezone.utc)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return datetime.now(tz=timezone.utc)


def _normalise_kind(raw: str) -> EventKind:
    candidate = raw.strip().lower()
    for kind in EventKind:
        if kind.value == candidate:
            return kind
    return EventKind.OTHER


def parse_event_envelope(
    *,
    subject: str,
    body: dict[str, Any],
    received_at: datetime | None = None,
) -> EventEnvelope | None:
    """Build an ``EventEnvelope`` from a NATS message.

    Returns ``None`` when the body is missing the minimum identifiers
    we need (``sessionId`` / ``type``). Caller is expected to log such
    drops; we don't raise so the consumer keeps draining the stream.
    """

    session_id = (
        body.get("sessionId")
        or body.get("session_id")
        or _session_from_subject(subject)
    )
    raw_type = body.get("type") or body.get("event") or body.get("kind")
    if not session_id or not raw_type:
        return None

    occurred_at = _coerce_dt(
        body.get("timestamp") or body.get("ts") or body.get("time")
    )
    received_at = received_at or datetime.now(tz=timezone.utc)

    pod_name = body.get("podName") or body.get("pod") or body.get("pod_name")
    pod_namespace = (
        body.get("namespace")
        or body.get("podNamespace")
        or body.get("pod_namespace")
    )
    plugin_id = body.get("pluginId") or body.get("plugin_id")

    return EventEnvelope(
        session_id=str(session_id),
        subject=subject,
        kind=_normalise_kind(str(raw_type)),
        raw_type=str(raw_type),
        occurred_at=occurred_at,
        received_at=received_at,
        pod_name=str(pod_name) if pod_name else None,
        pod_namespace=str(pod_namespace) if pod_namespace else None,
        plugin_id=str(plugin_id) if plugin_id else None,
        body=body,
    )


def _session_from_subject(subject: str) -> str | None:
    """Best-effort fallback when the body forgot ``sessionId``.

    Subject convention is ``events.<sessionId>``; if Claw ever extends
    it (``events.<sid>.<topic>``) the first dotted segment after
    ``events.`` is still the session id.
    """

    if not subject.startswith("events."):
        return None
    rest = subject[len("events.") :]
    head, _, _ = rest.partition(".")
    return head or None
