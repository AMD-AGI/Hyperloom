"""HTTP client for the Robust pod-metrics endpoint group.

Wraps the catalogue surface defined in
``Primus-Robust-Internal/tools/guardian/robust-api/pkg/endpoints_pod_metrics.go``.
The query API translates a session resolution result into one
``batch`` call here so per-session views stay 1 RTT regardless of the
session's pod count.

Network details (timeouts, retries, error propagation) are confined to
this module so the rest of the service treats Robust as a normal
async function.
"""

from __future__ import annotations

import logging
from typing import Iterable

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ..config import Settings
from ..models import PodRef

logger = logging.getLogger(__name__)


class RobustAPIError(RuntimeError):
    """Raised on non-200 responses from robust-api.

    The query API surfaces this as a 502 so callers can distinguish
    "we couldn't reach Robust" from "Robust returned no data".
    """


class PodMetricsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pods: list[PodRef]
    categories: list[str] = Field(default_factory=list)
    start: str
    end: str
    step: str | None = None


class _PodMetricSample(BaseModel):
    timestamp: int
    value: float


class _PodMetricSeries(BaseModel):
    labels: dict[str, str]
    values: list[_PodMetricSample]


class _PodMetricResult(BaseModel):
    name: str
    category: str
    unit: str
    series: list[_PodMetricSeries]


class _PodMetricsForPod(BaseModel):
    namespace: str
    name: str
    results: list[_PodMetricResult]


class _PodMetricsData(BaseModel):
    pods: list[_PodMetricsForPod]


class PodMetricsResponse(BaseModel):
    """Decoded body of ``POST /api/v1/pod-metrics/batch``.

    The shape mirrors robust-api so the query API can pass it through
    to clients without a per-field rewrite.
    """

    model_config = ConfigDict(extra="ignore")

    data: _PodMetricsData


class RobustAPIClient:
    """Thin async client around the robust-api pod-metrics surface."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = settings.robust_api_url.rstrip("/")
        self._timeout = settings.robust_api_timeout_seconds
        self._owns_client = client is None
        # ``client`` is injectable so tests can swap in pytest-httpx /
        # httpx.MockTransport. Production code constructs the default
        # client and closes it on shutdown.
        self._client = client or httpx.AsyncClient(timeout=self._timeout)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch_batch(self, request: PodMetricsRequest) -> PodMetricsResponse:
        """POST to ``/api/v1/pod-metrics/batch``.

        Returns an empty ``data.pods`` list when no pod was supplied
        (cheap shortcut to avoid an HTTP call for empty sessions).
        """

        if not request.pods:
            return PodMetricsResponse(data=_PodMetricsData(pods=[]))

        url = f"{self._base_url}/api/v1/pod-metrics/batch"
        payload = {
            "pods": [
                {"namespace": p.namespace, "name": p.name} for p in request.pods
            ],
            "categories": list(request.categories),
            "start": request.start,
            "end": request.end,
        }
        if request.step:
            payload["step"] = request.step
        try:
            resp = await self._client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise RobustAPIError(f"robust-api transport error: {exc}") from exc
        if resp.status_code >= 400:
            raise RobustAPIError(
                f"robust-api returned {resp.status_code}: {resp.text}"
            )
        return PodMetricsResponse.model_validate(resp.json())

    async def list_workloads(
        self,
        *,
        state: str | None = None,
        namespace: str | None = None,
        kind: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        """GET ``/api/v1/workloads`` with optional filters.

        The robust-api response is a flat array of workload objects; we
        return it verbatim so the caller can pick off whichever fields
        it needs (typically ``uid``/``id``, ``namespace``, ``labels``,
        ``state``).
        """

        url = f"{self._base_url}/api/v1/workloads"
        params: dict[str, str] = {"limit": str(limit)}
        if state:
            params["state"] = state
        if namespace:
            params["namespace"] = namespace
        if kind:
            params["kind"] = kind
        try:
            resp = await self._client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise RobustAPIError(f"robust-api transport error: {exc}") from exc
        if resp.status_code >= 400:
            raise RobustAPIError(
                f"robust-api returned {resp.status_code}: {resp.text}"
            )
        body = resp.json()
        if isinstance(body, list):
            return body
        if isinstance(body, dict):
            for key in ("data", "items", "results"):
                value = body.get(key)
                if isinstance(value, list):
                    return value
        return []

    async def get_workload_hierarchy(
        self,
        *,
        workload_id: str,
    ) -> dict[str, object]:
        """GET ``/api/v1/workloads/{workload_id}/hierarchy``.

        Used by the workload reconciler to enumerate pods belonging to
        a workload without round-tripping through Kubernetes.
        """

        url = (
            f"{self._base_url}/api/v1/workloads/"
            f"{workload_id}/hierarchy"
        )
        try:
            resp = await self._client.get(url)
        except httpx.HTTPError as exc:
            raise RobustAPIError(f"robust-api transport error: {exc}") from exc
        if resp.status_code >= 400:
            raise RobustAPIError(
                f"robust-api returned {resp.status_code}: {resp.text}"
            )
        body = resp.json()
        if isinstance(body, dict):
            return body
        return {"pods": []}

    async def list_categories_for_pod(
        self,
        *,
        pod: PodRef,
        categories: Iterable[str],
        start: str,
        end: str,
    ) -> list[dict[str, str]]:
        """GET ``/api/v1/pod-metrics/{ns}/{name}/list``.

        Returns the ``available`` array verbatim. Used by the
        ``/summary`` endpoint to discover which categories are
        meaningful per pod (e.g., brain pods skip ``gpu``).
        """

        url = (
            f"{self._base_url}/api/v1/pod-metrics/"
            f"{pod.namespace}/{pod.name}/list"
        )
        params: dict[str, str] = {"start": start, "end": end}
        cats = ",".join(c for c in categories if c)
        if cats:
            params["categories"] = cats
        try:
            resp = await self._client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise RobustAPIError(f"robust-api transport error: {exc}") from exc
        if resp.status_code >= 400:
            raise RobustAPIError(
                f"robust-api returned {resp.status_code}: {resp.text}"
            )
        body = resp.json()
        return list(body.get("available", []))
