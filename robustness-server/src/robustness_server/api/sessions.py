"""Session-centric query endpoints.

Translate ``session_id`` into the materialised pod set + metrics view
the rest of the platform consumes. The endpoints are intentionally
thin — domain logic lives in the repositories and services; routers
only marshal HTTP shapes and delegate.

Routes:

* ``GET /api/v1/sessions``                       — recently active
* ``GET /api/v1/sessions/{session_id}``          — session metadata
* ``GET /api/v1/sessions/{session_id}/pods``     — pods in window
* ``GET /api/v1/sessions/{session_id}/events``   — raw audit events
* ``GET /api/v1/sessions/{session_id}/metrics``  — proxy to robust-api
* ``GET /api/v1/sessions/{session_id}/summary``  — pods + metric catalog
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..models import PodAssignment, PodRef, Session
from ..services import (
    PodMetricsRequest,
    RobustAPIClient,
    RobustAPIError,
)
from ..store import (
    AssignmentsRepository,
    EventsRepository,
    SessionsRepository,
    get_database,
)

router = APIRouter(prefix="/api/v1", tags=["sessions"])


def _sessions_repo() -> SessionsRepository:
    return SessionsRepository(get_database().pool)


def _assignments_repo() -> AssignmentsRepository:
    return AssignmentsRepository(get_database().pool)


def _events_repo() -> EventsRepository:
    return EventsRepository(get_database().pool)


def _robust_client(request: Request) -> RobustAPIClient:
    """Pull the shared robust-api client off ``app.state``.

    Stored as a singleton because ``httpx.AsyncClient`` keeps an
    HTTP/2 connection pool — recreating per-request would defeat
    keep-alive.
    """

    client = getattr(request.app.state, "robust_client", None)
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="robust-api client is not initialised",
        )
    return client


@router.get("/sessions", response_model=list[Session])
async def list_sessions(
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    repo: Annotated[SessionsRepository, Depends(_sessions_repo)] = ...,
) -> list[Session]:
    """Most recently active sessions, newest first."""

    return await repo.list_recent(limit=limit)


@router.get("/sessions/{session_id}", response_model=Session)
async def get_session(
    session_id: str,
    repo: Annotated[SessionsRepository, Depends(_sessions_repo)] = ...,
) -> Session:
    session = await repo.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session


@router.get("/sessions/{session_id}/pods", response_model=list[PodAssignment])
async def list_session_pods(
    session_id: str,
    start: Annotated[datetime | None, Query()] = None,
    end: Annotated[datetime | None, Query()] = None,
    repo: Annotated[AssignmentsRepository, Depends(_assignments_repo)] = ...,
) -> list[PodAssignment]:
    """All pod assignments overlapping the requested window.

    ``start`` / ``end`` are optional; missing bounds collapse to
    "all-time" so dashboards can ask for a session's full pod set
    without computing a window.
    """

    return await repo.list_for_session(
        session_id=session_id,
        window_start=start,
        window_end=end,
    )


@router.get("/sessions/{session_id}/events")
async def list_session_events(
    session_id: str,
    start: Annotated[datetime | None, Query()] = None,
    end: Annotated[datetime | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=2000)] = 500,
    repo: Annotated[EventsRepository, Depends(_events_repo)] = ...,
) -> dict[str, Any]:
    rows = await repo.list_for_session(
        session_id=session_id,
        window_start=start,
        window_end=end,
        limit=limit,
    )
    return {"events": rows}


@router.get("/sessions/{session_id}/metrics")
async def get_session_metrics(
    session_id: str,
    start: Annotated[str, Query()],
    end: Annotated[str, Query()],
    categories: Annotated[str | None, Query()] = None,
    step: Annotated[str | None, Query()] = None,
    assignments_repo: Annotated[
        AssignmentsRepository, Depends(_assignments_repo)
    ] = ...,
    sessions_repo: Annotated[
        SessionsRepository, Depends(_sessions_repo)
    ] = ...,
    client: Annotated[RobustAPIClient, Depends(_robust_client)] = ...,
) -> dict[str, Any]:
    """Resolve pods → batch-call robust-api → return per-pod series.

    ``start`` / ``end`` are passed through to robust-api as Unix
    seconds (the format robust-api's ``query_range`` expects), so this
    endpoint is a metric-pass-through with session-scoped pod
    resolution.
    """

    if await sessions_repo.get(session_id) is None:
        raise HTTPException(status_code=404, detail="session not found")

    window_start, window_end = _parse_unix_window(start, end)
    pod_refs = await _resolve_pod_refs(
        repo=assignments_repo,
        session_id=session_id,
        window_start=window_start,
        window_end=window_end,
    )

    request = PodMetricsRequest(
        pods=pod_refs,
        categories=_split_csv(categories),
        start=start,
        end=end,
        step=step,
    )
    try:
        response = await client.fetch_batch(request)
    except RobustAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return response.model_dump()


@router.get("/sessions/{session_id}/summary")
async def get_session_summary(
    session_id: str,
    start: Annotated[str, Query()],
    end: Annotated[str, Query()],
    sessions_repo: Annotated[
        SessionsRepository, Depends(_sessions_repo)
    ] = ...,
    assignments_repo: Annotated[
        AssignmentsRepository, Depends(_assignments_repo)
    ] = ...,
    client: Annotated[RobustAPIClient, Depends(_robust_client)] = ...,
) -> dict[str, Any]:
    """Compose session metadata + pods + per-pod available categories.

    Calls the robust-api ``/list`` endpoint per pod so the response
    advertises which categories actually have data — saves the UI
    from probing each metric itself.
    """

    session = await sessions_repo.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    window_start, window_end = _parse_unix_window(start, end)
    assignments = await assignments_repo.list_for_session(
        session_id=session_id,
        window_start=window_start,
        window_end=window_end,
    )

    pods_payload: list[dict[str, Any]] = []
    for assignment in assignments:
        try:
            available = await client.list_categories_for_pod(
                pod=assignment.pod,
                categories=[],
                start=start,
                end=end,
            )
        except RobustAPIError:
            available = []
        pods_payload.append(
            {
                "assignment_id": assignment.assignment_id,
                "pod": assignment.pod.model_dump(),
                "role": assignment.role.value,
                "source": assignment.source.value,
                "t_start": assignment.t_start,
                "t_end": assignment.t_end,
                "available_metrics": available,
            }
        )

    return {
        "session": session.model_dump(),
        "pods": pods_payload,
        "window": {"start": start, "end": end},
    }


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_unix_window(start: str, end: str) -> tuple[datetime, datetime]:
    """Convert robust-api Unix-second strings into UTC datetimes.

    Used to filter ``session_pod_assignment`` rows on the same window
    that will be applied to metrics. Falls back to "all-time" bounds
    on parse failure so a malformed query still returns *something*
    instead of 500.
    """

    try:
        s_dt = datetime.fromtimestamp(int(start), tz=timezone.utc)
    except (TypeError, ValueError):
        s_dt = datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        e_dt = datetime.fromtimestamp(int(end), tz=timezone.utc)
    except (TypeError, ValueError):
        e_dt = datetime.now(tz=timezone.utc)
    return s_dt, e_dt


async def _resolve_pod_refs(
    *,
    repo: AssignmentsRepository,
    session_id: str,
    window_start: datetime,
    window_end: datetime,
) -> list[PodRef]:
    """Distinct pod refs assigned to ``session_id`` within the window.

    Same pod can appear in multiple assignment rows (open/close
    cycles); we collapse to a unique ``(namespace, name)`` set so the
    batch metrics call is not duplicated.
    """

    rows = await repo.list_for_session(
        session_id=session_id,
        window_start=window_start,
        window_end=window_end,
    )
    seen: set[tuple[str, str]] = set()
    out: list[PodRef] = []
    for row in rows:
        key = (row.pod.namespace, row.pod.name)
        if key in seen:
            continue
        seen.add(key)
        out.append(row.pod)
    return out
