# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""robustness-server client + Source adapter.

The client wraps a small subset of the robustness-server REST API
(``/healthz`` and the ``/api/v1/sessions/...`` pods/events/metrics/
summary endpoints). Networking errors (timeout / connect refused / 5xx)
are translated to :class:`SourceUnavailable` so :class:`DegradeRouter`
counts failures and degrades to the local fallback. 4xx responses are
returned as parsed JSON since they usually mean "no data for this
session" rather than an upstream outage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from .base import SourceData, SourceUnavailable
from .cluster_decoder import decode_gpu_snapshot, merge_gpu_snapshots


log = logging.getLogger(__name__)


@dataclass
class _MetricsWindow:
    """Explicit ``start`` / ``end`` Unix-second window for ``/metrics``
    and ``/summary``. M1 defaults to the last 5 minutes."""

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
        """Build a client for the robustness-server REST subset.

        Args:
            base_url (str): Base URL of the robustness-server; trailing
                slash is stripped. Must be non-empty.
            timeout_s (float): Per-request timeout in seconds, applied
                only when this client owns its ``httpx`` client.
            client (httpx.AsyncClient | None): Optional pre-built async
                client to reuse; when ``None`` a new one is created and
                owned by this instance.

        Raises:
            ValueError: If ``base_url`` is empty.
        """
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
        """Normalised base URL (without a trailing slash).

        Returns:
            str: The base URL the client issues requests against.
        """
        return self._base_url

    async def aclose(self) -> None:
        """Close the underlying ``httpx`` client when this owns it.

        No-op when an external client was injected, since its lifecycle
        belongs to the caller.
        """
        if self._owns_client:
            await self._client.aclose()

    # -- low-level GET ---------------------------------------------------

    async def _get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """Issue a GET and decode JSON, mapping transport errors.

        Timeouts, transport errors, 5xx responses and invalid JSON are
        translated to :class:`SourceUnavailable` so the DegradeRouter
        can count them. 404 / other 4xx responses return ``None`` since
        they usually mean "no data for this session" rather than outage.

        Args:
            path (str): Request path appended to the base URL.
            params (dict[str, Any] | None): Optional query parameters.

        Returns:
            Any: The decoded JSON body, or ``None`` for 4xx responses.

        Raises:
            SourceUnavailable: On timeout, transport error, 5xx status,
                or undecodable JSON.
        """
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
        """Probe ``GET /healthz`` for server liveness.

        Returns:
            bool: ``True`` when the server responds 200; ``False`` on a
            non-200 status or any transport error.
        """
        try:
            resp = await self._client.get("/healthz")
        except httpx.RequestError:
            return False
        return resp.status_code == 200

    async def list_sessions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """List known sessions via ``GET /api/v1/sessions``.

        Args:
            limit (int): Maximum number of sessions to request.

        Returns:
            list[dict[str, Any]]: The session rows, or ``[]`` when the
            response is not a list.
        """
        body = await self._get_json("/api/v1/sessions", params={"limit": limit})
        if isinstance(body, list):
            return body
        return []

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Fetch one session via ``GET /api/v1/sessions/{id}``.

        Args:
            session_id (str): Identifier of the session to fetch.

        Returns:
            dict[str, Any] | None: The session object, or ``None`` when
            absent / not a dict.
        """
        body = await self._get_json(f"/api/v1/sessions/{session_id}")
        return body if isinstance(body, dict) else None

    async def list_session_pods(
        self,
        session_id: str,
        *,
        start_unix: int | None = None,
        end_unix: int | None = None,
    ) -> list[dict[str, Any]]:
        """List a session's pods via ``GET .../{id}/pods``.

        Args:
            session_id (str): Identifier of the session.
            start_unix (int | None): Optional window start in Unix
                seconds; converted to ISO-8601 for the ``start`` query.
            end_unix (int | None): Optional window end in Unix seconds;
                converted to ISO-8601 for the ``end`` query.

        Returns:
            list[dict[str, Any]]: The pod rows, or ``[]`` when the
            response is not a list.
        """
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
        """List a session's events via ``GET .../{id}/events``.

        Args:
            session_id (str): Identifier of the session.
            start_unix (int | None): Optional window start in Unix
                seconds; converted to ISO-8601 for the ``start`` query.
            end_unix (int | None): Optional window end in Unix seconds;
                converted to ISO-8601 for the ``end`` query.
            limit (int): Maximum number of events to request.

        Returns:
            list[dict[str, Any]]: The ``events`` array from the
            response, or ``[]`` when absent.
        """
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
        """Fetch a session summary via ``GET .../{id}/summary``.

        Args:
            session_id (str): Identifier of the session.
            window (_MetricsWindow): Explicit ``start`` / ``end`` Unix
                second bounds for the summary.

        Returns:
            dict[str, Any]: The summary object, or ``{}`` when the
            response is not a dict.
        """
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
        """Fetch session metrics via ``GET .../{id}/metrics``.

        Args:
            session_id (str): Identifier of the session.
            window (_MetricsWindow): Explicit ``start`` / ``end`` Unix
                second bounds for the query.
            categories (list[str] | None): Optional metric categories;
                joined comma-separated into the ``categories`` query.
            step (str | None): Optional sampling step passed through.

        Returns:
            dict[str, Any]: The metrics object, or ``{}`` when the
            response is not a dict.
        """
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

    # -- cluster-physical proxies ---------------------------------------

    async def get_cluster_pod_metrics(
        self,
        namespace: str,
        name: str,
        window: _MetricsWindow,
        *,
        categories: list[str] | None = None,
        step: str | None = None,
    ) -> dict[str, Any]:
        """GET ``/api/v1/cluster/pods/{ns}/{name}/metrics``.

        Single-pod metrics; the response shape mirrors
        ``get_session_metrics`` (``{"data": {"pods": [...]}}``).

        Args:
            namespace (str): Kubernetes namespace of the pod.
            name (str): Pod name.
            window (_MetricsWindow): Explicit ``start`` / ``end`` Unix
                second bounds for the query.
            categories (list[str] | None): Optional metric categories;
                joined comma-separated into the ``categories`` query.
            step (str | None): Optional sampling step passed through.

        Returns:
            dict[str, Any]: The metrics object, or ``{}`` when the
            response is not a dict.
        """

        params: dict[str, Any] = {
            "start": str(window.start_unix),
            "end": str(window.end_unix),
        }
        if categories:
            params["categories"] = ",".join(categories)
        if step:
            params["step"] = step
        body = await self._get_json(
            f"/api/v1/cluster/pods/{namespace}/{name}/metrics",
            params=params,
        )
        return body if isinstance(body, dict) else {}

    async def list_cluster_pod_metric_categories(
        self,
        namespace: str,
        name: str,
        window: _MetricsWindow,
        *,
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """GET ``/api/v1/cluster/pods/{ns}/{name}/metrics/list``.

        Args:
            namespace (str): Kubernetes namespace of the pod.
            name (str): Pod name.
            window (_MetricsWindow): Explicit ``start`` / ``end`` Unix
                second bounds for the query.
            categories (list[str] | None): Optional metric categories;
                joined comma-separated into the ``categories`` query.

        Returns:
            list[dict[str, Any]]: The ``available`` array verbatim so
            callers can pick which categories to query, or ``[]`` when
            absent.
        """

        params: dict[str, Any] = {
            "start": str(window.start_unix),
            "end": str(window.end_unix),
        }
        if categories:
            params["categories"] = ",".join(categories)
        body = await self._get_json(
            f"/api/v1/cluster/pods/{namespace}/{name}/metrics/list",
            params=params,
        )
        if isinstance(body, dict):
            available = body.get("available")
            if isinstance(available, list):
                return available
        return []

    async def get_cluster_workload_hierarchy(
        self,
        workload_id: str,
    ) -> dict[str, Any]:
        """GET ``/api/v1/cluster/workloads/{id}/hierarchy``.

        Args:
            workload_id (str): Identifier of the workload whose pod /
                container hierarchy is requested.

        Returns:
            dict[str, Any]: The hierarchy object, or ``{}`` when the
            response is not a dict.
        """

        body = await self._get_json(
            f"/api/v1/cluster/workloads/{workload_id}/hierarchy",
        )
        return body if isinstance(body, dict) else {}

    async def list_cluster_faults(
        self,
        *,
        since: str | None = None,
        node: str | None = None,
        phase: str | None = None,
        page_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """GET ``/api/v1/cluster/faults``.

        Flattens dict / list responses so callers don't branch on
        shape; the array is already paginated upstream.

        Args:
            since (str | None): Optional lower time bound passed through
                as the ``since`` query.
            node (str | None): Optional node filter.
            phase (str | None): Optional fault-phase filter.
            page_size (int | None): Optional page size; stringified into
                the ``page_size`` query.

        Returns:
            list[dict[str, Any]]: The ``faults`` array, or ``[]`` when
            absent.
        """

        params: dict[str, Any] = {}
        if since:
            params["since"] = since
        if node:
            params["node"] = node
        if phase:
            params["phase"] = phase
        if page_size is not None:
            params["page_size"] = str(page_size)
        body = await self._get_json(
            "/api/v1/cluster/faults",
            params=params or None,
        )
        if isinstance(body, dict):
            faults = body.get("faults")
            if isinstance(faults, list):
                return faults
        if isinstance(body, list):
            return body
        return []


def _to_iso(unix_seconds: int) -> str:
    """Format Unix seconds as ISO-8601 UTC for ``start`` / ``end`` query args.

    Args:
        unix_seconds (int): The timestamp in Unix seconds.

    Returns:
        str: The ISO-8601 UTC string for the given instant.
    """
    from datetime import datetime, timezone

    return datetime.fromtimestamp(int(unix_seconds), tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Source adapter
# ---------------------------------------------------------------------------

class RobustnessServerSource:
    """Adapter wrapping :class:`RobustnessServerClient` as a :class:`Source`.

    Per tick fetches session-scoped data (``pods`` + ``events`` +
    ``summary``) for ``ctx.shared_state.session_id`` plus cluster
    ``cluster_faults``. Cluster fetches are best-effort: a 4xx / 5xx
    does not invalidate the session snapshot, but a transport failure
    does (server unreachable → DegradeRouter switches to local probe).
    """

    name = "robustness-server"

    def __init__(
        self,
        client: RobustnessServerClient,
        *,
        metrics_window_s: int = 300,
        events_limit: int = 200,
        faults_lookback_s: int = 300,
        faults_page_size: int = 50,
        enable_cluster_faults: bool = True,
        enable_cluster_pod_metrics: bool = False,
        pod_metrics_categories: tuple[str, ...] = ("gpu",),
        max_pods_per_tick: int = 16,
        workload_uid: str = "",
    ) -> None:
        """Configure the source adapter's per-tick fetch behaviour.

        Args:
            client (RobustnessServerClient): The HTTP client used for
                all fetches.
            metrics_window_s (int): Look-back window for session
                metrics / summary; clamped to at least 60 seconds.
            events_limit (int): Max events fetched per tick; clamped to
                at least 1.
            faults_lookback_s (int): Look-back for cluster faults;
                clamped to at least 0.
            faults_page_size (int): Faults page size; clamped to the
                1..500 range.
            enable_cluster_faults (bool): Whether to fetch cluster
                faults each tick.
            enable_cluster_pod_metrics (bool): Whether to fan out
                per-pod cluster GPU metrics; off by default due to cost.
            pod_metrics_categories (tuple[str, ...]): Metric categories
                requested when pod metrics are enabled.
            max_pods_per_tick (int): Upper bound on pods queried per
                tick; clamped to at least 1.
        """
        self._client = client
        self._metrics_window_s = max(60, int(metrics_window_s))
        self._events_limit = max(1, int(events_limit))
        self._faults_lookback_s = max(0, int(faults_lookback_s))
        self._faults_page_size = max(1, min(500, int(faults_page_size)))
        self._enable_cluster_faults = bool(enable_cluster_faults)
        # Off by default: fans out one HTTP call per pod per tick. Enabling
        # makes signals/local_health.py prefer server-decoded GPU data over
        # LocalProbe rocm-smi.
        self._enable_cluster_pod_metrics = bool(enable_cluster_pod_metrics)
        self._pod_metrics_categories = tuple(pod_metrics_categories)
        self._max_pods_per_tick = max(1, int(max_pods_per_tick))
        # ``workload_uid`` opts into hierarchy-based pod discovery so
        # multi-node RayJobs reconcile the full pod set (head + workers).
        # Empty string keeps the legacy ``list_session_pods`` path.
        self._workload_uid = (workload_uid or "").strip()

    async def fetch(self, ctx: Any) -> SourceData:
        """Collect one tick of session- and cluster-scoped data.

        Fetches pods, events and (when the window is set) a summary for
        the context's session, plus best-effort cluster faults and
        optional per-pod GPU metrics. Transport failures on the cluster
        endpoints re-raise so the DegradeRouter can degrade.

        Args:
            ctx (Any): The reactor context; must expose a session id via
                ``ctx.shared_state.session_id`` and may carry
                ``ctx.now_unix``.

        Returns:
            SourceData: The assembled snapshot for the tick.

        Raises:
            SourceUnavailable: When no session id is present, or a
                cluster fetch hits a transport / 5xx failure.
        """
        session_id = _extract_session_id(ctx)
        if not session_id:
            raise SourceUnavailable("no session_id in reactor context")

        now_unix = int(getattr(ctx, "now_unix", 0)) or 0
        window = _MetricsWindow(
            start_unix=now_unix - self._metrics_window_s if now_unix else 0,
            end_unix=now_unix or 0,
        )

        session_pods = await self._client.list_session_pods(
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

        # When workload_uid is configured, merge cluster-hierarchy pods with
        # the session view so the fan-out covers Ray workers the session has
        # not yet seen (consistent multi-node view for downstream signals).
        hierarchy_pods: list[dict[str, Any]] = []
        if self._workload_uid:
            try:
                hierarchy = await self._client.get_cluster_workload_hierarchy(
                    self._workload_uid,
                )
            except SourceUnavailable:
                raise
            hierarchy_pods = _extract_hierarchy_pods(hierarchy)
        merged_pods = _merge_pods(session_pods, hierarchy_pods)

        cluster_faults: list[dict[str, Any]] = []
        if self._enable_cluster_faults:
            since = (
                str(now_unix - self._faults_lookback_s)
                if now_unix and self._faults_lookback_s
                else None
            )
            try:
                cluster_faults = await self._client.list_cluster_faults(
                    since=since,
                    page_size=self._faults_page_size,
                )
            except SourceUnavailable:
                # Transport-level failure (timeouts / 5xx wrapped by
                # _get_json): re-raise so DegradeRouter counts it.
                raise

        local_gpu: dict[str, Any] = {}
        if (
            self._enable_cluster_pod_metrics
            and merged_pods
            and window.start_unix
            and window.end_unix
        ):
            local_gpu = await self._fetch_cluster_pod_metrics(merged_pods, window)

        return SourceData(
            session_pods=merged_pods,
            session_events=events,
            session_summary=summary,
            cluster_faults=cluster_faults,
            local_gpu=local_gpu,
            sources_used=[self.name],
        )

    async def _fetch_cluster_pod_metrics(
        self,
        pods: list[dict[str, Any]],
        window: _MetricsWindow,
    ) -> dict[str, Any]:
        """Fan out cluster pod metrics across the session's pods.

        Decodes each per-pod response into the LocalProbe ``local_gpu``
        schema and merges them into one snapshot. A 5xx / transport
        failure on any pod re-raises :class:`SourceUnavailable` so the
        DegradeRouter degrades.
        """

        refs = _unique_pod_refs(pods)
        if not refs:
            return {}
        if len(refs) > self._max_pods_per_tick:
            refs = refs[: self._max_pods_per_tick]

        snapshots: list[dict[str, Any]] = []
        for ns, name in refs:
            try:
                metrics = await self._client.get_cluster_pod_metrics(
                    ns,
                    name,
                    window,
                    categories=list(self._pod_metrics_categories),
                )
            except SourceUnavailable:
                raise
            decoded = decode_gpu_snapshot(metrics)
            if decoded:
                snapshots.append(decoded)
        return merge_gpu_snapshots(snapshots)


def _extract_session_id(ctx: Any) -> str:
    """Pull the session id out of a reactor context.

    Args:
        ctx (Any): The reactor context, expected to expose
            ``shared_state.session_id``.

    Returns:
        str: The session id, or ``""`` when it is missing / falsy.
    """
    shared = getattr(ctx, "shared_state", None)
    return getattr(shared, "session_id", "") or ""


def _unique_pod_refs(pods: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Distinct (namespace, name) tuples from session_pods.

    Rows carry the pod under ``pod.namespace`` / ``pod.name``; a pod may
    recur across open/close cycles so we collapse to the unique set to
    avoid duplicating the cluster-metrics fan-out.
    """

    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for entry in pods or []:
        if not isinstance(entry, dict):
            continue
        pod = entry.get("pod") if isinstance(entry.get("pod"), dict) else entry
        ns = str(pod.get("namespace") or "")
        name = str(pod.get("name") or "")
        if not ns or not name:
            continue
        key = (ns, name)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _extract_hierarchy_pods(
    hierarchy: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Pluck pod rows from a workload hierarchy response.

    Accepts ``pods`` (documented), ``children`` / ``items`` (mirrors),
    and a single ``pod`` (degraded response) so an upstream schema nudge
    does not silently disable multi-node fan-out.
    """

    if not isinstance(hierarchy, dict):
        return []
    for key in ("pods", "children", "items"):
        rows = hierarchy.get(key)
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    pod = hierarchy.get("pod")
    if isinstance(pod, dict):
        return [pod]
    return []


def _merge_pods(
    session_pods: list[dict[str, Any]],
    extra_pods: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine session_pods with hierarchy-derived pods, deduping by ref.

    Hierarchy rows are wrapped in the session-pod envelope
    (``{"pod": {...}}``) so downstream consumers see a uniform shape.
    Session entries win on conflicts (richer phase / role metadata).
    """

    out: list[dict[str, Any]] = list(session_pods or [])
    seen: set[tuple[str, str]] = set()
    for entry in out:
        if not isinstance(entry, dict):
            continue
        pod = entry.get("pod") if isinstance(entry.get("pod"), dict) else entry
        ns = str(pod.get("namespace") or "")
        name = str(pod.get("name") or "")
        if ns and name:
            seen.add((ns, name))

    for raw in extra_pods or []:
        pod = raw.get("pod") if isinstance(raw.get("pod"), dict) else raw
        if not isinstance(pod, dict):
            continue
        ns = str(pod.get("namespace") or "")
        name = str(pod.get("name") or "")
        if not ns or not name:
            continue
        key = (ns, name)
        if key in seen:
            continue
        seen.add(key)
        out.append({"pod": {"namespace": ns, "name": name}, "source": "hierarchy"})
    return out


__all__ = [
    "RobustnessServerClient",
    "RobustnessServerSource",
]
