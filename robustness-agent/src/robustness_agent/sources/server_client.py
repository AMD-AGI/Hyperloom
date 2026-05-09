"""robustness-server client + Source adapter.

The client wraps a small subset of the robustness-server REST API used
by the M1 reactor:

* ``GET  /healthz``
* ``GET  /api/v1/sessions``
* ``GET  /api/v1/sessions/{session_id}``
* ``GET  /api/v1/sessions/{session_id}/pods``
* ``GET  /api/v1/sessions/{session_id}/events``
* ``GET  /api/v1/sessions/{session_id}/metrics``
* ``GET  /api/v1/sessions/{session_id}/summary``

Networking errors (timeout / connect refused / 5xx) are translated to
:class:`SourceUnavailable` so :class:`DegradeRouter` can count failures
and degrade to the local fallback. 4xx responses are returned to the
caller as parsed JSON because they typically encode "no data for this
session" rather than an upstream outage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from .base import Source, SourceData, SourceUnavailable


log = logging.getLogger(__name__)


@dataclass
class _MetricsWindow:
    """Optional override for the ``/metrics`` and ``/summary`` window.

    The robustness-server requires explicit ``start`` / ``end`` Unix
    seconds. The reactor decides which window to ask for; M1 defaults
    to "last 5 minutes" computed from the context clock.
    """

    start_unix: int
    end_unix: int


class RobustnessServerClient:
    """HTTP client for the subset of robustness-server we use in M1."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be non-empty")
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_s)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- low-level GET ---------------------------------------------------

    async def _get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        try:
            resp = await self._client.get(path, params=params)
        except httpx.TimeoutException as exc:
            raise SourceUnavailable(f"GET {path}: timeout") from exc
        except httpx.RequestError as exc:
            raise SourceUnavailable(f"GET {path}: {type(exc).__name__}: {exc}") from exc
        if resp.status_code >= 500:
            raise SourceUnavailable(
                f"GET {path}: upstream {resp.status_code}"
            )
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            return None
        try:
            return resp.json()
        except ValueError as exc:
            raise SourceUnavailable(f"GET {path}: invalid json") from exc

    # -- public methods --------------------------------------------------

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/healthz")
        except httpx.RequestError:
            return False
        return resp.status_code == 200

    async def list_sessions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        body = await self._get_json("/api/v1/sessions", params={"limit": limit})
        if isinstance(body, list):
            return body
        return []

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        body = await self._get_json(f"/api/v1/sessions/{session_id}")
        return body if isinstance(body, dict) else None

    async def list_session_pods(
        self,
        session_id: str,
        *,
        start_unix: int | None = None,
        end_unix: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if start_unix is not None:
            params["start"] = _to_iso(start_unix)
        if end_unix is not None:
            params["end"] = _to_iso(end_unix)
        body = await self._get_json(
            f"/api/v1/sessions/{session_id}/pods",
            params=params or None,
        )
        if isinstance(body, list):
            return body
        return []

    async def list_session_events(
        self,
        session_id: str,
        *,
        start_unix: int | None = None,
        end_unix: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if start_unix is not None:
            params["start"] = _to_iso(start_unix)
        if end_unix is not None:
            params["end"] = _to_iso(end_unix)
        body = await self._get_json(
            f"/api/v1/sessions/{session_id}/events",
            params=params,
        )
        if isinstance(body, dict):
            events = body.get("events")
            if isinstance(events, list):
                return events
        return []

    async def get_session_summary(
        self,
        session_id: str,
        window: _MetricsWindow,
    ) -> dict[str, Any]:
        body = await self._get_json(
            f"/api/v1/sessions/{session_id}/summary",
            params={"start": str(window.start_unix), "end": str(window.end_unix)},
        )
        return body if isinstance(body, dict) else {}

    async def get_session_metrics(
        self,
        session_id: str,
        window: _MetricsWindow,
        *,
        categories: list[str] | None = None,
        step: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "start": str(window.start_unix),
            "end": str(window.end_unix),
        }
        if categories:
            params["categories"] = ",".join(categories)
        if step:
            params["step"] = step
        body = await self._get_json(
            f"/api/v1/sessions/{session_id}/metrics",
            params=params,
        )
        return body if isinstance(body, dict) else {}


def _to_iso(unix_seconds: int) -> str:
    """Format Unix seconds as ISO-8601 UTC for ``start`` / ``end`` query args."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(int(unix_seconds), tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Source adapter
# ---------------------------------------------------------------------------

class RobustnessServerSource:
    """Adapter wrapping :class:`RobustnessServerClient` as a :class:`Source`.

    Per tick we fetch ``pods`` + ``events`` + ``summary`` for the
    session referenced by ``ctx.shared_state.session_id``. ``metrics``
    is intentionally skipped in M1 because cluster-physical proxies
    arrive in M2; the reactor still has plenty to chew on with pod
    phase + recent events.
    """

    name = "robustness-server"

    def __init__(
        self,
        client: RobustnessServerClient,
        *,
        metrics_window_s: int = 300,
        events_limit: int = 200,
    ) -> None:
        self._client = client
        self._metrics_window_s = max(60, int(metrics_window_s))
        self._events_limit = max(1, int(events_limit))

    async def fetch(self, ctx: Any) -> SourceData:
        session_id = _extract_session_id(ctx)
        if not session_id:
            raise SourceUnavailable("no session_id in reactor context")

        now_unix = int(getattr(ctx, "now_unix", 0)) or 0
        window = _MetricsWindow(
            start_unix=now_unix - self._metrics_window_s if now_unix else 0,
            end_unix=now_unix or 0,
        )

        pods = await self._client.list_session_pods(
            session_id,
            start_unix=window.start_unix or None,
            end_unix=window.end_unix or None,
        )
        events = await self._client.list_session_events(
            session_id,
            start_unix=window.start_unix or None,
            end_unix=window.end_unix or None,
            limit=self._events_limit,
        )
        summary: dict[str, Any] = {}
        if window.start_unix and window.end_unix:
            summary = await self._client.get_session_summary(session_id, window)

        return SourceData(
            session_pods=pods,
            session_events=events,
            session_summary=summary,
            sources_used=[self.name],
        )


def _extract_session_id(ctx: Any) -> str:
    shared = getattr(ctx, "shared_state", None)
    return getattr(shared, "session_id", "") or ""


__all__ = [
    "RobustnessServerClient",
    "RobustnessServerSource",
]
