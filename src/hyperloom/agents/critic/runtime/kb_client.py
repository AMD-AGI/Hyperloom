# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KB client interface + minimal HTTP transport."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


from .errors import (
    KBConflictError,
    KBNotFoundError,
    KBTransportError,
    KBValidationError,
)
from hyperloom.common.url_safety import require_http_url as _require_http_url


DEFAULT_TIMEOUT_MS = 10_000
DEFAULT_RETRY_MAX = 3
DEFAULT_BACKOFF_BASE = 1.0  # seconds


@dataclass(frozen=True)
class KBListQuery:
    """Which entries one ``POST /api/kb/list`` reads, how many and in what order."""

    scope_filter: dict[str, Any]
    kind: str | None = None
    metadata_filter: dict[str, Any] | None = None
    limit: int = 10
    sort_by: str = "updated_at_desc"
    include_deleted: bool = False


class KBClient(Protocol):
    """Protocol shared between ``HTTPKBClient`` and ``InMemoryKBClient``."""

    def list(self, query: KBListQuery) -> dict[str, Any]:
        """Read entries matching a scope via ``POST /api/kb/list``."""

    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Insert or update a single entry via ``POST /api/kb/upsert``."""


# ---------------------------------------------------------------------------
class HTTPKBClient:
    """Minimal HTTP wrapper over ``/api/kb/*``."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        retry_max: int = DEFAULT_RETRY_MAX,
        sleep_fn=time.sleep,
    ):
        """Configure the HTTP transport."""
        if not base_url:
            raise ValueError("HTTPKBClient: base_url is required")
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("KB_SERVICE_TOKEN") or ""
        self.timeout_s = float(timeout_ms) / 1000.0
        self.retry_max = retry_max
        self._sleep = sleep_fn

    # Public API — one method per endpoint.
    def list(self, query: KBListQuery) -> dict[str, Any]:
        """Read entries matching a scope via ``POST /api/kb/list``."""
        body: dict[str, Any] = {
            "scope_filter": query.scope_filter,
            "limit": query.limit,
            "sort_by": query.sort_by,
            "include_deleted": query.include_deleted,
        }
        if query.kind is not None:
            body["kind"] = query.kind
        if query.metadata_filter is not None:
            body["metadata_filter"] = query.metadata_filter
        return self._request("/api/kb/list", body)

    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Insert or update a single entry via ``POST /api/kb/upsert``."""
        return self._request("/api/kb/upsert", payload)

    # Internal — request / retry
    def _request(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST ``body`` to ``path`` with retry and backoff."""
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        last_error: Exception | None = None
        attempt = 0
        while attempt <= self.retry_max:
            attempt += 1
            _require_http_url(url)
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # nosec B310 - URL scheme checked above.
                    payload = resp.read().decode("utf-8") or "{}"
                    return json.loads(payload)
            except urllib.error.HTTPError as exc:
                # 4xx → don't retry.
                status = getattr(exc, "code", 0)
                err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
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
        raise KBTransportError(f"{path}: failed after {self.retry_max} retries — last_error={last_error!r}")

    def _backoff_for(self, attempt: int) -> float:
        """Compute the sleep duration before the next retry."""
        base = DEFAULT_BACKOFF_BASE * (2 ** (attempt - 1))
        return base * (0.9 + 0.2 * random.random())


__all__ = [
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_RETRY_MAX",
    "DEFAULT_TIMEOUT_MS",
    "HTTPKBClient",
    "KBClient",
    "KBListQuery",
]
