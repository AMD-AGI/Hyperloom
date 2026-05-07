"""Tests for the SessionRouter orchestrator.

Each test wires fake repositories that record calls. We exercise:

* every observation upserts the session + appends the event;
* sandbox_create / sandbox_status open a hands assignment with the
  inferred role;
* sandbox_delete closes the hands assignment;
* a terminal event closes all open assignments and stamps the
  session terminal state;
* KV PUT / DELETE drive brain assignments and never touch sessions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from robustness_server.models import (
    EventKind,
    KVAssignmentChange,
    PodAssignmentSource,
    PodRef,
    PodRole,
    SessionState,
    parse_event_envelope,
)
from robustness_server.services import SessionRouter


class FakeSessions:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.terminals: list[dict[str, Any]] = []

    async def upsert_observation(self, **kwargs: Any) -> None:
        self.upserts.append(kwargs)

    async def mark_terminal(self, **kwargs: Any) -> None:
        self.terminals.append(kwargs)


class FakeAssignments:
    def __init__(self) -> None:
        self.opens: list[dict[str, Any]] = []
        self.closes: list[dict[str, Any]] = []
        self.session_closes: list[dict[str, Any]] = []
        self._next_id = 1

    async def open_assignment(self, **kwargs: Any) -> int:
        self.opens.append(kwargs)
        assignment_id = self._next_id
        self._next_id += 1
        return assignment_id

    async def close_assignment(self, **kwargs: Any) -> int:
        self.closes.append(kwargs)
        return 1

    async def close_all_for_session(self, **kwargs: Any) -> int:
        self.session_closes.append(kwargs)
        return 1


class FakeEvents:
    def __init__(self) -> None:
        self.appended: list[Any] = []

    async def append(self, event: Any) -> int:
        self.appended.append(event)
        return len(self.appended)


@pytest.fixture
def router_setup() -> tuple[SessionRouter, FakeSessions, FakeAssignments, FakeEvents]:
    sessions = FakeSessions()
    assignments = FakeAssignments()
    events = FakeEvents()
    router = SessionRouter(
        sessions=sessions,
        assignments=assignments,
        events=events,
    )
    return router, sessions, assignments, events


def _event(**body: Any):
    """Helper: build an EventEnvelope from a body dict."""

    base = {"sessionId": "sess", "type": "session_start", "timestamp": "2026-04-28T12:00:00Z"}
    base.update(body)
    return parse_event_envelope(subject="events.sess", body=base)


@pytest.mark.asyncio
async def test_event_persists_session_and_audit(router_setup) -> None:
    router, sessions, assignments, events = router_setup
    event = _event(type="session_start", userId="alice", pluginId="p")
    await router.ingest_event(event)
    assert len(sessions.upserts) == 1
    assert sessions.upserts[0]["session_id"] == "sess"
    assert sessions.upserts[0]["user_id"] == "alice"
    assert sessions.upserts[0]["plugin_id"] == "p"
    assert events.appended == [event]
    assert assignments.opens == []  # no podName, no assignment


@pytest.mark.asyncio
async def test_sandbox_create_opens_hands_assignment(router_setup) -> None:
    router, _, assignments, _ = router_setup
    event = _event(
        type="sandbox_create",
        podName="hands-1",
        podNamespace="claw",
        gpuCount=4,
    )
    await router.ingest_event(event)
    assert len(assignments.opens) == 1
    opened = assignments.opens[0]
    assert opened["session_id"] == "sess"
    assert opened["role"] == PodRole.HANDS_GPU
    assert opened["source"] == PodAssignmentSource.NATS_EVENT
    assert opened["pod"] == PodRef(namespace="claw", name="hands-1")


@pytest.mark.asyncio
async def test_sandbox_status_without_gpu_defaults_to_hands_cpu(router_setup) -> None:
    router, _, assignments, _ = router_setup
    event = _event(type="sandbox_status", podName="hands-cpu", podNamespace="claw")
    await router.ingest_event(event)
    assert assignments.opens[0]["role"] == PodRole.HANDS_CPU


@pytest.mark.asyncio
async def test_explicit_role_label_wins(router_setup) -> None:
    router, _, assignments, _ = router_setup
    event = _event(
        type="sandbox_create",
        podName="custom",
        podNamespace="claw",
        role="brain",
    )
    await router.ingest_event(event)
    assert assignments.opens[0]["role"] == PodRole.BRAIN


@pytest.mark.asyncio
async def test_sandbox_delete_closes_assignment(router_setup) -> None:
    router, _, assignments, _ = router_setup
    event = _event(
        type="sandbox_delete",
        podName="hands-1",
        podNamespace="claw",
    )
    await router.ingest_event(event)
    assert assignments.opens == []
    assert len(assignments.closes) == 1
    assert assignments.closes[0]["session_id"] == "sess"


@pytest.mark.asyncio
async def test_terminal_event_marks_session_and_closes_all(router_setup) -> None:
    router, sessions, assignments, _ = router_setup
    event = _event(type="exec_complete", podName="brain-1", podNamespace="claw")
    await router.ingest_event(event)
    assert sessions.terminals == [
        {
            "session_id": "sess",
            "terminal_at": event.occurred_at,
            "final_state": SessionState.COMPLETED,
        }
    ]
    assert assignments.session_closes == [
        {"session_id": "sess", "closed_at": event.occurred_at}
    ]
    assert assignments.opens == []  # terminal short-circuits before opening


@pytest.mark.asyncio
async def test_failed_event_uses_failed_state(router_setup) -> None:
    router, sessions, _, _ = router_setup
    event = _event(type="exec_failed")
    await router.ingest_event(event)
    assert sessions.terminals[0]["final_state"] == SessionState.FAILED


@pytest.mark.asyncio
async def test_kv_put_opens_brain_assignment(router_setup) -> None:
    router, sessions, assignments, events = router_setup
    change = KVAssignmentChange(
        session_id="sess",
        pod_name="brain-1",
        pod_namespace="claw",
        opening=True,
        observed_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )
    await router.ingest_kv_change(change)
    assert sessions.upserts == []  # KV is not a session observation
    assert events.appended == []
    assert len(assignments.opens) == 1
    op = assignments.opens[0]
    assert op["role"] == PodRole.BRAIN
    assert op["source"] == PodAssignmentSource.NATS_KV


@pytest.mark.asyncio
async def test_kv_delete_closes_brain_assignment(router_setup) -> None:
    router, _, assignments, _ = router_setup
    change = KVAssignmentChange(
        session_id="sess",
        pod_name="brain-1",
        pod_namespace="claw",
        opening=False,
        observed_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )
    await router.ingest_kv_change(change)
    assert assignments.opens == []
    assert len(assignments.closes) == 1
    assert assignments.closes[0]["role"] == PodRole.BRAIN


@pytest.mark.asyncio
async def test_event_audit_failure_does_not_block_side_effects(
    router_setup,
) -> None:
    router, sessions, assignments, events = router_setup

    async def boom(_event: Any) -> int:
        raise RuntimeError("disk full")

    events.append = boom  # type: ignore[assignment]
    event = _event(type="sandbox_create", podName="hands-1", podNamespace="claw")
    await router.ingest_event(event)
    assert len(sessions.upserts) == 1
    assert len(assignments.opens) == 1
