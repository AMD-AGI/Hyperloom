# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tiny Ray Dashboard REST client, running inside the sandbox."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import Any

import httpx

from .log import warn

# Ray Dashboard's fixed HTTP control port on the head pod.
RAY_DASHBOARD_PORT = 8265

# 30s read so a multi-MB log fetch doesn't wedge the poll cadence.
_HTTPX_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)


def _wrap_for_dash(body: str) -> str:
    """Base64-wrap a bash body so it survives /bin/sh (dash), which Ray Dashboard uses to run entrypoints."""
    import base64 as _b64

    encoded = _b64.b64encode(body.encode()).decode()
    return f"echo {encoded} | base64 -d | bash"


class RayDashboardError(RuntimeError):
    """Raised when the Ray dashboard returns an unexpected status."""

    def __init__(self, status: int | None, body: str, *, endpoint: str) -> None:
        """Initialize the error with response context."""
        snippet = body[:500] + ("..." if len(body) > 500 else "")
        super().__init__(f"Ray dashboard {endpoint} -> status={status} body={snippet}")
        self.status = status
        self.body = body
        self.endpoint = endpoint


def dashboard_url(head_pod_ip: str) -> str:
    """Build the Ray Dashboard base URL for a given head pod IP."""
    if not head_pod_ip:
        raise ValueError("head_pod_ip is empty; cannot build Ray Dashboard URL")
    return f"http://{head_pod_ip}:{RAY_DASHBOARD_PORT}"


class RayDashboardClient:
    """Stateless HTTP wrapper for ``/api/jobs/`` on a single head pod IP."""

    def __init__(self, head_pod_ip: str, *, token: str | None = None) -> None:
        """Create a client bound to one head pod's dashboard."""
        self._base = dashboard_url(head_pod_ip)
        headers: dict[str, str] = {"Accept": "application/json"}
        tok = (token or "").strip()
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
        self._client = httpx.Client(
            timeout=_HTTPX_TIMEOUT,
            headers=headers,
        )

    def close(self) -> None:
        """Close the underlying HTTP client, ignoring any errors."""
        with suppress(Exception):
            self._client.close()

    def __enter__(self) -> "RayDashboardClient":
        """Enter the context manager."""
        return self

    def __exit__(self, *exc) -> None:
        """Close the HTTP client on context-manager exit."""
        self.close()

    def _decode(self, resp: httpx.Response, endpoint: str) -> Any:
        """Decode a JSON response, wrapping decode errors."""
        try:
            return resp.json()
        except json.JSONDecodeError as e:
            raise RayDashboardError(resp.status_code, resp.text, endpoint=endpoint) from e

    def submit_job(self, entrypoint: str, *, runtime_env: dict | None = None) -> str:
        """POST /api/jobs/; returns the ``submission_id`` immediately (entrypoint runs async, poll :meth:`get_job`)."""
        if not entrypoint:
            raise ValueError("entrypoint is empty")
        entrypoint = _wrap_for_dash(entrypoint)
        endpoint = "POST /api/jobs/"
        payload: dict[str, Any] = {"entrypoint": entrypoint}
        if runtime_env:
            payload["runtime_env"] = runtime_env
        resp = self._client.post(f"{self._base}/api/jobs/", json=payload)
        if resp.status_code != 200:
            raise RayDashboardError(resp.status_code, resp.text, endpoint=endpoint)
        data = self._decode(resp, endpoint)
        sub = data.get("submission_id") or data.get("job_id")
        if not sub:
            raise RayDashboardError(resp.status_code, resp.text, endpoint=endpoint)
        return sub

    def get_job(self, submission_id: str) -> dict:
        """GET /api/jobs/{submission_id}; response carries ``status`` (PENDING/RUNNING/SUCCEEDED/FAILED/STOPPED) + ``message``."""
        if not submission_id:
            raise ValueError("submission_id is empty")
        endpoint = f"GET /api/jobs/{submission_id}"
        resp = self._client.get(f"{self._base}/api/jobs/{submission_id}")
        if resp.status_code != 200:
            raise RayDashboardError(resp.status_code, resp.text, endpoint=endpoint)
        return self._decode(resp, endpoint)

    def get_job_logs(self, submission_id: str) -> str:
        """GET /api/jobs/{submission_id}/logs."""
        if not submission_id:
            raise ValueError("submission_id is empty")
        endpoint = f"GET /api/jobs/{submission_id}/logs"
        resp = self._client.get(f"{self._base}/api/jobs/{submission_id}/logs")
        if resp.status_code != 200:
            warn(f"{endpoint} -> {resp.status_code}; returning empty log")
            return ""
        try:
            data = resp.json()
            # Ray may wrap as {"logs": "..."} or return plain text.
            return data.get("logs", "") if isinstance(data, dict) else str(data)
        except json.JSONDecodeError:
            return resp.text
