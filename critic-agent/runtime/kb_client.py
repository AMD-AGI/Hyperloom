# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""KB client interface + minimal HTTP transport.

The Critic uses only 4 KB endpoints (contract §4):

* ``POST /api/kb/list``
* ``POST /api/kb/upsert``
* ``POST /api/kb/batch_insert``
* ``POST /api/kb/edges/add``

We expose them as plain methods on :class:`KBClient` (a protocol) and ship
two implementations:

* :class:`HTTPKBClient` — thin urllib-based transport with retry +
  exponential backoff. We deliberately avoid pulling in ``httpx`` so the
  runtime stays installable in minimal Codex containers.
* :class:`InMemoryKBClient` (in ``in_memory_kb_client.py``) — same surface,
  pure Python state, used by tests and dry-runs.
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
    ) -> dict[str, Any]:
        """Read entries matching a scope via ``POST /api/kb/list``.

        Args:
            scope_filter (dict[str, Any]): Scope dimensions to match against.
            kind (str | None): Optional entry kind filter.
            metadata_filter (dict[str, Any] | None): Optional metadata filter.
            limit (int): Maximum number of entries to return.
            sort_by (str): Server-side sort key (e.g. ``updated_at_desc``).
            include_deleted (bool): Whether soft-deleted entries are returned.

        Returns:
            dict[str, Any]: The decoded JSON response body.
        """
        ...

    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Insert or update a single entry via ``POST /api/kb/upsert``.

        Args:
            payload (dict[str, Any]): The entry body to upsert.

        Returns:
            dict[str, Any]: The decoded JSON response body.
        """
        ...

    def batch_insert(
        self,
        items: list[dict[str, Any]],
        *,
        on_conflict: str = "upsert",
    ) -> dict[str, Any]:
        """Insert many entries via ``POST /api/kb/batch_insert``.

        Args:
            items (list[dict[str, Any]]): Entry bodies to insert.
            on_conflict (str): Conflict resolution strategy (e.g. ``upsert``).

        Returns:
            dict[str, Any]: The decoded JSON response body.
        """
        ...

    def add_edges(self, edges: list[dict[str, Any]]) -> dict[str, Any]:
        """Create graph edges via ``POST /api/kb/edges/add``.

        Args:
            edges (list[dict[str, Any]]): Edge definitions to add.

        Returns:
            dict[str, Any]: The decoded JSON response body.
        """
        ...


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
        """Configure the HTTP transport.

        Args:
            base_url (str): KB service base URL; trailing slash is stripped.
            token (str | None): Bearer token; falls back to the
                ``KB_SERVICE_TOKEN`` environment variable when omitted.
            timeout_ms (int): Per-request timeout in milliseconds.
            retry_max (int): Maximum number of retries on 429/5xx/network
                errors.
            backoff_base (float): Base seconds for exponential backoff.
            sleep_fn (Callable[[float], None]): Sleep function, injectable for
                tests.

        Raises:
            ValueError: If ``base_url`` is empty.
        """
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
        """Read entries matching a scope via ``POST /api/kb/list``.

        Args:
            scope_filter (dict[str, Any]): Scope dimensions to match against.
            kind (str | None): Optional entry kind filter; omitted when
                ``None``.
            metadata_filter (dict[str, Any] | None): Optional metadata filter;
                omitted when ``None``.
            limit (int): Maximum number of entries to return.
            sort_by (str): Server-side sort key (e.g. ``updated_at_desc``).
            include_deleted (bool): Whether soft-deleted entries are returned.

        Returns:
            dict[str, Any]: The decoded JSON response body.
        """
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
        """Insert or update a single entry via ``POST /api/kb/upsert``.

        Args:
            payload (dict[str, Any]): The entry body to upsert.

        Returns:
            dict[str, Any]: The decoded JSON response body.
        """
        return self._request("/api/kb/upsert", payload)

    def batch_insert(
        self,
        items: list[dict[str, Any]],
        *,
        on_conflict: str = "upsert",
    ) -> dict[str, Any]:
        """Insert many entries via ``POST /api/kb/batch_insert``.

        Args:
            items (list[dict[str, Any]]): Entry bodies to insert.
            on_conflict (str): Conflict resolution strategy (e.g. ``upsert``).

        Returns:
            dict[str, Any]: The decoded JSON response body.
        """
        return self._request(
            "/api/kb/batch_insert",
            {"items": items, "on_conflict": on_conflict},
        )

    def add_edges(self, edges: list[dict[str, Any]]) -> dict[str, Any]:
        """Create graph edges via ``POST /api/kb/edges/add``.

        Args:
            edges (list[dict[str, Any]]): Edge definitions to add.

        Returns:
            dict[str, Any]: The decoded JSON response body.
        """
        return self._request("/api/kb/edges/add", {"edges": edges})

    # ------------------------------------------------------------------
    # Internal — request / retry
    # ------------------------------------------------------------------
    def _request(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST ``body`` to ``path`` with retry, backoff, and metrics.

        Retries on 429/5xx/network errors up to ``retry_max`` times. Records
        write counter and duration metrics on every attempt.

        Args:
            path (str): Endpoint path appended to ``base_url`` (e.g.
                ``/api/kb/upsert``).
            body (dict[str, Any]): JSON-serialisable request body.

        Returns:
            dict[str, Any]: The decoded JSON response body (``{}`` if empty).

        Raises:
            KBNotFoundError: On HTTP 404.
            KBConflictError: On HTTP 409.
            KBValidationError: On non-retryable 4xx (excluding 429).
            KBTransportError: When all retries are exhausted.
        """
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
        """Compute the sleep duration before the next retry.

        Args:
            attempt (int): 1-based attempt number that just failed.

        Returns:
            float: Seconds to sleep — exponential in ``attempt`` with jitter
            to avoid a thundering herd.
        """
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
