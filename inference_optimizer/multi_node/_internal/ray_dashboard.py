"""Tiny Ray Dashboard REST client.

This module runs INSIDE THE SANDBOX. Per ADDENDUM-02, multi_node
orchestration code MUST NOT ``import ray`` or call
``ray.init(address="ray://...")`` to control the **inference** RayJob: the
Python client is sensitive to version skew with the cluster image.
Dashboard REST is HTTP/JSON and version-tolerant, so it is the supported
sandbox→RayJob control channel. The sandbox may still have ``ray``
installed for unrelated work (e.g. a local ``ray start --head``); this
module stays HTTP-only regardless.

ADDENDUM-02 says NOTHING about code that runs INSIDE the RayJob pods
themselves (head/worker). Those pods ARE the ray cluster — ``ray.init()``
without an address is the standard way for in-pod scripts to attach to
the local GCS, and that path is fine for entrypoints we submit via
``submit_job()``. Just keep the rule straight: sandbox = HTTP only;
RayJob pod entrypoint = whatever the framework needs.

The dashboard listens on a single fixed port inside the head pod. The
port is always 8265 — there is no configuration knob and the
orchestrator agent does not need to know about it. We hard-code it here.

Reachable from the sandbox via the head pod's POD IP (NOT host IP):
``http://<head_pod_ip>:8265``. The pod IP is fetched via SaFE's
GetWorkload .pods array (PodIp field).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .log import warn

# Hard-coded; see module docstring.
RAY_DASHBOARD_PORT = 8265

# Job-submission HTTP is short-lived (returns immediately with a
# submission_id), but log fetches can be 1-10 MB. 30s read keeps a single
# call from wedging the CLI's poll cadence.
_HTTPX_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)



def _wrap_for_dash(body: str) -> str:
    """Wrap a bash script body so it survives /bin/sh (dash) execution.

    Ray Dashboard /api/jobs/ executes entrypoints via /bin/sh, which on many
    images is dash. Dash doesn't support ``set -o pipefail`` and other
    bash-isms. This helper base64-encodes the body and runs it through bash.
    """
    import base64 as _b64
    encoded = _b64.b64encode(body.encode()).decode()
    return f'echo {encoded} | base64 -d | bash'

class RayDashboardError(RuntimeError):
    """Raised when the Ray dashboard returns an unexpected status."""

    def __init__(self, status: int | None, body: str, *, endpoint: str) -> None:
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

    def __init__(self, head_pod_ip: str) -> None:
        self._base = dashboard_url(head_pod_ip)
        self._client = httpx.Client(
            timeout=_HTTPX_TIMEOUT,
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self) -> "RayDashboardClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _decode(self, resp: httpx.Response, endpoint: str) -> Any:
        try:
            return resp.json()
        except json.JSONDecodeError as e:
            raise RayDashboardError(resp.status_code, resp.text, endpoint=endpoint) from e

    def submit_job(self, entrypoint: str, *, runtime_env: dict | None = None) -> str:
        """POST /api/jobs/.

        Returns the dashboard's ``submission_id`` immediately; the entrypoint
        runs asynchronously inside the cluster. The orchestrator polls
        :meth:`get_job` until status is terminal.
        """
        if not entrypoint:
            raise ValueError("entrypoint is empty")
        # Wrap for dash compatibility (Ray Dashboard uses /bin/sh).
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
        """GET /api/jobs/{submission_id}.

        Response includes ``status`` ∈ {PENDING, RUNNING, SUCCEEDED, FAILED, STOPPED}
        and ``message``. Used by poll loops.
        """
        if not submission_id:
            raise ValueError("submission_id is empty")
        endpoint = f"GET /api/jobs/{submission_id}"
        resp = self._client.get(f"{self._base}/api/jobs/{submission_id}")
        if resp.status_code != 200:
            raise RayDashboardError(resp.status_code, resp.text, endpoint=endpoint)
        return self._decode(resp, endpoint)

    def get_job_logs(self, submission_id: str) -> str:
        """GET /api/jobs/{submission_id}/logs. Returns plain text logs."""
        if not submission_id:
            raise ValueError("submission_id is empty")
        endpoint = f"GET /api/jobs/{submission_id}/logs"
        resp = self._client.get(f"{self._base}/api/jobs/{submission_id}/logs")
        if resp.status_code != 200:
            warn(f"{endpoint} -> {resp.status_code}; returning empty log")
            return ""
        try:
            data = resp.json()
            # Newer Ray versions wrap as {"logs": "..."}; older return text.
            return data.get("logs", "") if isinstance(data, dict) else str(data)
        except json.JSONDecodeError:
            return resp.text
