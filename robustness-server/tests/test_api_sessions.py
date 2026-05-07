"""HTTP smoke tests for the session-centric endpoints.

We monkeypatch the repository factories on the ``api.sessions`` module
so the endpoints run against in-memory stubs. ``api_client`` (from
``conftest``) gives us a TestClient with the lifespan disabled.

These tests verify wiring (404 → not found, query params → repo
arguments, robust-api error → 502) without booting a real DB or NATS.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

import robustness_server.api.sessions as sessions_module
from robustness_server.models import (
    PodAssignment,
    PodAssignmentSource,
    PodRef,
    PodRole,
    Session,
)
from robustness_server.services import RobustAPIError


class StubSessions:
    def __init__(self, sessions: dict[str, Session] | None = None) -> None:
        self._sessions = sessions or {}
        self.list_calls: list[int] = []

    async def list_recent(self, *, limit: int):
        self.list_calls.append(limit)
        return list(self._sessions.values())[:limit]

    async def get(self, session_id: str):
        return self._sessions.get(session_id)


class StubAssignments:
    def __init__(self, rows: list[PodAssignment] | None = None) -> None:
        self.rows = rows or []
        self.list_calls: list[dict[str, Any]] = []

    async def list_for_session(self, **kwargs: Any):
        self.list_calls.append(kwargs)
        return list(self.rows)


class StubEvents:
    def __init__(self) -> None:
        self.list_calls: list[dict[str, Any]] = []

    async def list_for_session(self, **kwargs: Any):
        self.list_calls.append(kwargs)
        return [{"event_id": 1, "session_id": kwargs.get("session_id"), "body": {}}]


class StubRobust:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.batch_calls: list[Any] = []
        self.list_calls: list[Any] = []

    async def fetch_batch(self, request):
        self.batch_calls.append(request)
        if self.fail:
            raise RobustAPIError("downstream offline")
        from robustness_server.services.robust_client import (
            PodMetricsResponse,
            _PodMetricsData,
        )

        return PodMetricsResponse(data=_PodMetricsData(pods=[]))

    async def list_categories_for_pod(self, **kwargs: Any):
        self.list_calls.append(kwargs)
        return [{"name": "cpu_usage_cores", "category": "cpu"}]


def _install_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_client,
    sessions: StubSessions | None = None,
    assignments: StubAssignments | None = None,
    events: StubEvents | None = None,
    robust: StubRobust | None = None,
) -> tuple[StubSessions, StubAssignments, StubEvents, StubRobust]:
    sessions = sessions or StubSessions()
    assignments = assignments or StubAssignments()
    events = events or StubEvents()
    robust = robust or StubRobust()
    # FastAPI resolves Depends(...) by *function identity* captured at
    # router build time, so module-level monkeypatching is invisible
    # to the running app. Use the official override surface instead.
    overrides = api_client.app.dependency_overrides
    overrides[sessions_module._sessions_repo] = lambda: sessions
    overrides[sessions_module._assignments_repo] = lambda: assignments
    overrides[sessions_module._events_repo] = lambda: events
    overrides[sessions_module._robust_client] = lambda: robust
    api_client.app.state.robust_client = robust
    return sessions, assignments, events, robust


def test_list_sessions_returns_recent(monkeypatch, api_client) -> None:
    s = Session(
        session_id="s1",
        t_start=datetime(2026, 4, 28, tzinfo=timezone.utc),
        last_event_at=datetime(2026, 4, 28, 12, tzinfo=timezone.utc),
    )
    sessions, *_ = _install_stubs(
        monkeypatch,
        api_client=api_client,
        sessions=StubSessions({"s1": s}),
    )
    resp = api_client.get("/api/v1/sessions?limit=5")
    assert resp.status_code == 200
    assert resp.json()[0]["session_id"] == "s1"
    assert sessions.list_calls == [5]


def test_get_session_returns_404_when_missing(monkeypatch, api_client) -> None:
    _install_stubs(monkeypatch, api_client=api_client)
    resp = api_client.get("/api/v1/sessions/missing")
    assert resp.status_code == 404


def test_list_pods_passes_window(monkeypatch, api_client) -> None:
    assignment = PodAssignment(
        session_id="s1",
        pod=PodRef(namespace="ns", name="p1"),
        role=PodRole.HANDS_GPU,
        source=PodAssignmentSource.NATS_EVENT,
        t_start=datetime(2026, 4, 28, tzinfo=timezone.utc),
        last_seen_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )
    _, assignments, _, _ = _install_stubs(
        monkeypatch,
        api_client=api_client,
        assignments=StubAssignments([assignment]),
    )
    resp = api_client.get(
        "/api/v1/sessions/s1/pods?start=2026-04-28T00:00:00Z&end=2026-04-29T00:00:00Z"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["pod"]["name"] == "p1"
    call = assignments.list_calls[0]
    assert call["window_start"] is not None
    assert call["window_end"] is not None


def test_metrics_endpoint_returns_404_when_session_missing(
    monkeypatch, api_client
) -> None:
    _install_stubs(monkeypatch, api_client=api_client)
    resp = api_client.get(
        "/api/v1/sessions/missing/metrics?start=1&end=2"
    )
    assert resp.status_code == 404


def test_metrics_endpoint_resolves_pods_and_calls_robust(
    monkeypatch, api_client
) -> None:
    s = Session(
        session_id="s1",
        t_start=datetime(2026, 4, 28, tzinfo=timezone.utc),
        last_event_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )
    assignment = PodAssignment(
        session_id="s1",
        pod=PodRef(namespace="ns", name="p1"),
        role=PodRole.HANDS_GPU,
        source=PodAssignmentSource.NATS_EVENT,
        t_start=datetime(2026, 4, 28, tzinfo=timezone.utc),
        last_seen_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )
    _, _, _, robust = _install_stubs(
        monkeypatch,
        api_client=api_client,
        sessions=StubSessions({"s1": s}),
        assignments=StubAssignments([assignment, assignment]),  # duplicate
    )
    resp = api_client.get(
        "/api/v1/sessions/s1/metrics?start=100&end=200&categories=gpu,cpu&step=15"
    )
    assert resp.status_code == 200
    assert len(robust.batch_calls) == 1
    request = robust.batch_calls[0]
    # duplicates collapsed
    assert len(request.pods) == 1
    assert request.categories == ["gpu", "cpu"]
    assert request.step == "15"


def test_metrics_endpoint_502s_on_robust_error(monkeypatch, api_client) -> None:
    s = Session(
        session_id="s1",
        t_start=datetime(2026, 4, 28, tzinfo=timezone.utc),
        last_event_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )
    assignment = PodAssignment(
        session_id="s1",
        pod=PodRef(namespace="ns", name="p1"),
        role=PodRole.HANDS_GPU,
        source=PodAssignmentSource.NATS_EVENT,
        t_start=datetime(2026, 4, 28, tzinfo=timezone.utc),
        last_seen_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )
    _install_stubs(
        monkeypatch,
        api_client=api_client,
        sessions=StubSessions({"s1": s}),
        assignments=StubAssignments([assignment]),
        robust=StubRobust(fail=True),
    )
    resp = api_client.get(
        "/api/v1/sessions/s1/metrics?start=100&end=200"
    )
    assert resp.status_code == 502


def test_summary_calls_per_pod_list_endpoint(monkeypatch, api_client) -> None:
    s = Session(
        session_id="s1",
        t_start=datetime(2026, 4, 28, tzinfo=timezone.utc),
        last_event_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )
    a1 = PodAssignment(
        session_id="s1",
        pod=PodRef(namespace="ns", name="p1"),
        role=PodRole.HANDS_GPU,
        source=PodAssignmentSource.NATS_EVENT,
        t_start=datetime(2026, 4, 28, tzinfo=timezone.utc),
        last_seen_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )
    a2 = PodAssignment(
        session_id="s1",
        pod=PodRef(namespace="ns", name="brain"),
        role=PodRole.BRAIN,
        source=PodAssignmentSource.NATS_KV,
        t_start=datetime(2026, 4, 28, tzinfo=timezone.utc),
        last_seen_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
    )
    _, _, _, robust = _install_stubs(
        monkeypatch,
        api_client=api_client,
        sessions=StubSessions({"s1": s}),
        assignments=StubAssignments([a1, a2]),
    )
    resp = api_client.get(
        "/api/v1/sessions/s1/summary?start=100&end=200"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session"]["session_id"] == "s1"
    assert {pod["pod"]["name"] for pod in body["pods"]} == {"p1", "brain"}
    # one /list call per assignment
    assert len(robust.list_calls) == 2
