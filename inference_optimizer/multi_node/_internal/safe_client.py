# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Thin SaFE REST client used by the rayjob CLIs.

Endpoints (under ``/api/v1/workloads``): POST create, GET ``{id}`` (full
state incl. ``.pods``), GET ``{id}/service``, POST ``{id}/stop``
(idempotent), DELETE ``{id}``. The path id is the workload id from create.
Uses :mod:`httpx` (already a dep) to avoid a new sandbox dependency.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .log import warn

# 60s read upper bound so a slow/hung SaFE call doesn't wedge the CLI poll loop.
_HTTPX_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=30.0)


class SafeApiError(RuntimeError):
    """Raised when SaFE returns an unexpected status or a network call fails."""

    def __init__(self, status: int | None, body: str, *, endpoint: str) -> None:
        # Truncate body to keep the agent's stderr legible.
        snippet = body[:500] + ("..." if len(body) > 500 else "")
        super().__init__(f"SaFE {endpoint} -> status={status} body={snippet}")
        self.status = status
        self.body = body
        self.endpoint = endpoint


def _strip_trailing_slash(url: str) -> str:
    return url.rstrip("/")


class SafeClient:
    """Single-instance SaFE REST client. Stateless, thread-unsafe."""

    def __init__(self, base_url: str, api_key: str) -> None:
        if not base_url:
            raise ValueError("SAFE_API_URL is empty")
        if not api_key:
            raise ValueError("SAFE_API_KEY is empty")
        self._base = _strip_trailing_slash(base_url)
        self._api_key = api_key
        self._client = httpx.Client(
            timeout=_HTTPX_TIMEOUT,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self) -> "SafeClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _url(self, path: str) -> str:
        # path always begins with "/api/v1/..."
        return f"{self._base}{path}"

    def _decode(self, resp: httpx.Response, endpoint: str) -> Any:
        try:
            return resp.json()
        except json.JSONDecodeError as e:
            raise SafeApiError(resp.status_code, resp.text, endpoint=endpoint) from e

    def create_workload(self, body: dict) -> str:
        """POST /api/v1/workloads; returns the workload_id (non-2xx raises :class:`SafeApiError`)."""
        endpoint = "POST /api/v1/workloads"
        resp = self._client.post(self._url("/api/v1/workloads"), json=body)
        if not (200 <= resp.status_code < 300):
            raise SafeApiError(resp.status_code, resp.text, endpoint=endpoint)
        data = self._decode(resp, endpoint)
        # Tolerate both the bare {"workloadId": ...} and handle()-wrapped shapes.
        wid = (
            data.get("workloadId")
            or (data.get("data") or {}).get("workloadId")
            or data.get("workload_id")
        )
        if not wid:
            raise SafeApiError(resp.status_code, resp.text, endpoint=endpoint)
        return wid

    def get_workload(self, workload_id: str) -> dict:
        """GET /api/v1/workloads/{workload_id}. Returns the full GetWorkloadResponse."""
        endpoint = f"GET /api/v1/workloads/{workload_id}"
        resp = self._client.get(self._url(f"/api/v1/workloads/{workload_id}"))
        if not (200 <= resp.status_code < 300):
            raise SafeApiError(resp.status_code, resp.text, endpoint=endpoint)
        data = self._decode(resp, endpoint)
        # Tolerate a {"data": ...} wrapper.
        if isinstance(data, dict) and "data" in data and "phase" not in data:
            return data["data"]
        return data

    def get_workload_service(self, workload_id: str) -> dict:
        """GET /api/v1/workloads/{workload_id}/service. Returns GetWorkloadServiceResponse."""
        endpoint = f"GET /api/v1/workloads/{workload_id}/service"
        resp = self._client.get(self._url(f"/api/v1/workloads/{workload_id}/service"))
        if not (200 <= resp.status_code < 300):
            raise SafeApiError(resp.status_code, resp.text, endpoint=endpoint)
        data = self._decode(resp, endpoint)
        if isinstance(data, dict) and "data" in data and "clusterIp" not in data:
            return data["data"]
        return data

    def stop_workload(self, workload_id: str) -> None:
        """POST /api/v1/workloads/{workload_id}/stop; idempotent (404/409 treated as success)."""
        endpoint = f"POST /api/v1/workloads/{workload_id}/stop"
        resp = self._client.post(self._url(f"/api/v1/workloads/{workload_id}/stop"))
        if resp.status_code in (200, 204, 404, 409):
            if resp.status_code in (404, 409):
                warn(f"{endpoint} -> {resp.status_code}; treating as success")
            return
        raise SafeApiError(resp.status_code, resp.text, endpoint=endpoint)

    def delete_workload(self, workload_id: str) -> None:
        """DELETE /api/v1/workloads/{workload_id}; idempotent (404 treated as success)."""
        endpoint = f"DELETE /api/v1/workloads/{workload_id}"
        resp = self._client.delete(self._url(f"/api/v1/workloads/{workload_id}"))
        if resp.status_code in (200, 204, 404):
            if resp.status_code == 404:
                warn(f"{endpoint} -> 404; treating as success")
            return
        raise SafeApiError(resp.status_code, resp.text, endpoint=endpoint)


def from_env() -> SafeClient:
    """Construct a SafeClient from SAFE_API_URL + SAFE_API_KEY env vars; clean RuntimeError when missing."""
    base = (os.environ.get("SAFE_API_URL") or "").strip()
    key = (os.environ.get("SAFE_API_KEY") or "").strip()
    missing = [name for name, val in (("SAFE_API_URL", base), ("SAFE_API_KEY", key)) if not val]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". The rayjob CLI is invoked from inside a Claw sandbox where "
            "SAFE_API_URL and SAFE_API_KEY must be exported by Brain at "
            "sandbox start. If running locally for debugging, export both "
            "before calling."
        )
    return SafeClient(base_url=base, api_key=key)
