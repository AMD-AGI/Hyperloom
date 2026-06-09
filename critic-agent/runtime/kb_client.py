# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""KB client interface + minimal HTTP transport.

The Critic uses only 4 KB endpoints (contract §4):

* ``POST /api/kb/list``
* ``POST /api/kb/upsert``
* ``POST /api/kb/batch_insert``
* ``POST /api/kb/edges/add``

Exposed as methods on the :class:`KBClient` protocol with two
implementations: :class:`HTTPKBClient` (urllib-based, retry + exponential
backoff; avoids ``httpx`` so it installs in minimal Codex containers) and
:class:`InMemoryKBClient` (pure-Python, for tests / dry-runs).
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from .errors import (
    KBConflictError,
    KBNotFoundError,
    KBTransportError,
    KBValidationError,
)
from .metrics import (
    CRITIC_KB_WRITE_DURATION_SECONDS,
    CRITIC_KB_WRITE_TOTAL,
    get_registry,
)


DEFAULT_TIMEOUT_MS = 10_000
DEFAULT_RETRY_MAX = 3
DEFAULT_BACKOFF_BASE = 1.0  # seconds; 1, 2, 4 ...


class KBClient(Protocol):
    """Protocol shared between ``HTTPKBClient`` and ``InMemoryKBClient``."""

    def list(
        self,
        *,
        scope_filter: dict[str, Any],
        kind: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        sort_by: str = "updated_at_desc",
        include_deleted: bool = False,
    ) -> dict[str, Any]: ...

    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def batch_insert(
        self,
        items: list[dict[str, Any]],
        *,
        on_conflict: str = "upsert",
    ) -> dict[str, Any]: ...

    def add_edges(self, edges: list[dict[str, Any]]) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
class HTTPKBClient:
    """Minimal HTTP wrapper over ``/api/kb/*``.

    Retry policy mirrors contract §6: exponential backoff on 429 / 5xx /
    network errors up to ``retry_max`` times. 4xx errors raise
    :class:`KBValidationError` immediately so the caller can dead-letter.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        retry_max: int = DEFAULT_RETRY_MAX,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        sleep_fn=time.sleep,
    ):
        if not base_url:
            raise ValueError("HTTPKBClient: base_url is required")
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("KB_SERVICE_TOKEN") or ""
        self.timeout_s = float(timeout_ms) / 1000.0
        self.retry_max = retry_max
        self.backoff_base = backoff_base
        self._sleep = sleep_fn

    # ------------------------------------------------------------------
    # Public API — one method per endpoint.
    # ------------------------------------------------------------------
    def list(
        self,
        *,
        scope_filter: dict[str, Any],
        kind: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        sort_by: str = "updated_at_desc",
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "scope_filter": scope_filter,
            "limit": limit,
            "sort_by": sort_by,
            "include_deleted": include_deleted,
        }
        if kind is not None:
            body["kind"] = kind
        if metadata_filter is not None:
            body["metadata_filter"] = metadata_filter
        return self._request("/api/kb/list", body)

    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("/api/kb/upsert", payload)

    def batch_insert(
        self,
        items: list[dict[str, Any]],
        *,
        on_conflict: str = "upsert",
    ) -> dict[str, Any]:
        return self._request(
            "/api/kb/batch_insert",
            {"items": items, "on_conflict": on_conflict},
        )

    def add_edges(self, edges: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("/api/kb/edges/add", {"edges": edges})

    # ------------------------------------------------------------------
    # Internal — request / retry
    # ------------------------------------------------------------------
    def _request(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        registry = get_registry()
        endpoint_label = path.split("/")[-1]
        last_error: Exception | None = None
        attempt = 0
        start = time.time()
        while attempt <= self.retry_max:
            attempt += 1
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    payload = resp.read().decode("utf-8") or "{}"
                    body_obj = json.loads(payload) if payload else {}
                    registry.counter(CRITIC_KB_WRITE_TOTAL).inc({
                        "endpoint": endpoint_label,
                        "status": str(resp.status),
                    })
                    registry.histogram(CRITIC_KB_WRITE_DURATION_SECONDS).observe(
                        time.time() - start, {"endpoint": endpoint_label}
                    )
                    return body_obj
            except urllib.error.HTTPError as exc:
                # 4xx → don't retry; let caller dead-letter.
                status = getattr(exc, "code", 0)
                err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                registry.counter(CRITIC_KB_WRITE_TOTAL).inc({
                    "endpoint": endpoint_label, "status": str(status),
                })
                if status == 404:
                    raise KBNotFoundError(f"{path} 404: {err_body}") from exc
                if status == 409:
                    raise KBConflictError(f"{path} 409: {err_body}") from exc
                if 400 <= status < 500 and status != 429:
                    raise KBValidationError(f"{path} {status}: {err_body}") from exc
                last_error = exc
            except urllib.error.URLError as exc:
                last_error = exc
            if attempt > self.retry_max:
                break
            self._sleep(self._backoff_for(attempt))
        raise KBTransportError(
            f"{path}: failed after {self.retry_max} retries — last_error={last_error!r}"
        )

    def _backoff_for(self, attempt: int) -> float:
        # Exponential with optional jitter to avoid thundering herd.
        base = self.backoff_base * (2 ** (attempt - 1))
        return base * (0.9 + 0.2 * random.random())


__all__ = [
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_RETRY_MAX",
    "DEFAULT_TIMEOUT_MS",
    "HTTPKBClient",
    "KBClient",
]
