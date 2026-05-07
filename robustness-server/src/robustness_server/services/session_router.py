"""Single orchestrator that translates ingest signals into store writes.

Both the NATS event consumer and the KV watcher route everything they
see through this object, so the policy that decides "open vs close
assignment / mark session terminal" lives in exactly one place. The
class itself is pure async glue against the repositories — easy to
unit-test by swapping in fake repos.

Policy summary:

* Every observed event upserts the session row (idempotent) and
  appends to ``session_events``. If the event carries ``podName``,
  the (session, pod, role) assignment is opened with role inferred
  from the event kind / payload.
* Terminal events close *all* open assignments for the session and
  stamp ``t_end`` / ``final_state``.
* KV PUT opens a brain assignment; KV DELETE/expire closes it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping

from ..models import (
    EventEnvelope,
    EventKind,
    KVAssignmentChange,
    PodAssignmentSource,
    PodRef,
    PodRole,
    SessionState,
)
from ..models.events import TERMINAL_KINDS
from ..store import AssignmentsRepository, EventsRepository, SessionsRepository

logger = logging.getLogger(__name__)


class SessionRouter:
    """Single orchestrator for ingest signals.

    Holds repository references but no other state — safe to share
    across the consumer and the watcher.
    """

    def __init__(
        self,
        *,
        sessions: SessionsRepository,
        assignments: AssignmentsRepository,
        events: EventsRepository,
        default_namespace: str = "default",
    ) -> None:
        self._sessions = sessions
        self._assignments = assignments
        self._events = events
        self._default_namespace = default_namespace

    async def ingest_event(self, event: EventEnvelope) -> None:
        """Persist + react to one decoded NATS event.

        The session upsert and event append are unconditional. Pod
        assignment effects depend on the event kind / payload — see
        ``_apply_event_side_effects``.
        """

        await self._sessions.upsert_observation(
            session_id=event.session_id,
            occurred_at=event.occurred_at,
            user_id=_pick(event.body, "userId", "user_id"),
            plugin_id=event.plugin_id,
        )
        try:
            await self._events.append(event)
        except Exception:
            # The audit log is best-effort: a duplicate or malformed
            # row should not stop downstream side effects, but we do
            # want it surfaced for ops.
            logger.exception(
                "events repo append failed (session=%s subject=%s)",
                event.session_id,
                event.subject,
            )
        await self._apply_event_side_effects(event)

    async def ingest_kv_change(self, change: KVAssignmentChange) -> None:
        """React to a brain-pod KV assignment change.

        KV PUT opens a brain assignment; DELETE/expire closes it. We
        do not synthesise a session row here — the brain pod may be
        registered before any event lands, and creating an empty
        session would muddy the ``last_event_at`` ordering used by
        ``list_recent``.
        """

        pod = PodRef(
            namespace=change.pod_namespace or self._default_namespace,
            name=change.pod_name,
        )
        if change.opening:
            await self._assignments.open_assignment(
                session_id=change.session_id,
                pod=pod,
                role=PodRole.BRAIN,
                source=PodAssignmentSource.NATS_KV,
                observed_at=change.observed_at,
            )
            return
        await self._assignments.close_assignment(
            session_id=change.session_id,
            pod=pod,
            role=PodRole.BRAIN,
            closed_at=change.observed_at,
        )

    async def _apply_event_side_effects(self, event: EventEnvelope) -> None:
        if event.kind in TERMINAL_KINDS:
            await self._sessions.mark_terminal(
                session_id=event.session_id,
                terminal_at=event.occurred_at,
                final_state=SessionState(TERMINAL_KINDS[event.kind]),
            )
            await self._assignments.close_all_for_session(
                session_id=event.session_id,
                closed_at=event.occurred_at,
            )
            return

        if event.pod_name:
            pod = PodRef(
                namespace=event.pod_namespace or self._default_namespace,
                name=event.pod_name,
            )
            role = _infer_role(event)
            if event.kind == EventKind.SANDBOX_DELETE:
                await self._assignments.close_assignment(
                    session_id=event.session_id,
                    pod=pod,
                    role=role,
                    closed_at=event.occurred_at,
                )
                return
            await self._assignments.open_assignment(
                session_id=event.session_id,
                pod=pod,
                role=role,
                source=PodAssignmentSource.NATS_EVENT,
                observed_at=event.occurred_at,
            )


def _infer_role(event: EventEnvelope) -> PodRole:
    """Best-effort role inference from the event payload.

    The Claw event taxonomy carries an explicit ``role`` field on
    sandbox lifecycle messages; we honour it when present and fall
    back to GPU-allocation-based heuristics otherwise. Unknown roles
    are persisted as ``OTHER`` so the data does not get lost.
    """

    raw = (
        _pick(event.body, "role", "podRole")
        or _pick(event.body, "component")
    )
    if raw:
        try:
            return PodRole(str(raw).lower())
        except ValueError:
            pass
    if _pick(event.body, "gpuCount", "gpu_count"):
        return PodRole.HANDS_GPU
    if event.kind in (EventKind.SANDBOX_CREATE, EventKind.SANDBOX_STATUS):
        return PodRole.HANDS_CPU
    return PodRole.OTHER


def _pick(body: Mapping[str, Any], *keys: str) -> Any:
    """Pick the first non-empty value from ``body`` for any of ``keys``."""

    for key in keys:
        value = body.get(key)
        if value not in (None, "", 0, False):
            return value
        if isinstance(value, (int, float)) and value != 0:
            return value
    return None


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)
