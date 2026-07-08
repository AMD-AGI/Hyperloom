# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Read-only Cortex KB client: maps the Critic ``list`` read onto cortex ``/v1``.

The Critic was written against the legacy scoped-article contract
(``POST /api/kb/{list,upsert,batch_insert,edges/add}``) served by
``claw-memory-service``. The cortex ``kb-service`` deprecated that surface
and exposes a graph paradigm instead (``/v1/points`` + ``/v1/edges``).

Against cortex the Critic is a **read-only consumer of priors**: it queries
existing knowledge to inform a review, but never writes back. This client
therefore only implements the read path (:meth:`list` ->
``POST /v1/points/query``); the write methods of the
:class:`~runtime.kb_client.KBClient` protocol are intentionally rejected so
an accidental write fails loudly instead of mutating the graph. Deployments
using this client should run with ``KB_WRITE_ENABLED=false``.

Read mapping
------------
* The Critic ``kind`` is namespaced as ``critic_{kind}`` on the wire, matching
  how Critic-authored priors are tagged in ``kb_points``.
* ``scope_filter`` becomes a JSONB-containment ``attrs_filter`` on the
  reserved ``_critic_scope`` key (cortex matches with ``@>``), which both
  filters by scope and restricts results to Critic-authored rows.
* The reserved ``_critic_*`` attrs keys are projected back into the legacy
  scoped-article entry shape so callers see no difference from the in-memory
  / HTTP clients.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any

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

# Reserved attrs keys Critic-authored points carry; used to project a cortex
# point back into the scoped-article entry shape on read.
_ATTR_SCOPE = "_critic_scope"
_ATTR_KIND = "_critic_kind"
_ATTR_SLUG = "_critic_slug"
_ATTR_IMPORTANCE = "_critic_importance"
_ATTR_SUMMARY = "_critic_summary"
_ATTR_METADATA = "_critic_metadata"
_ATTR_UPDATED_AT = "_critic_updated_at"

_KIND_PREFIX = "critic_"

_READ_ONLY_MSG = (
    "CortexKBClient is read-only; the Critic must not write to the cortex "
    "graph KB (run with KB_WRITE_ENABLED=false)"
)


def _normalise_value(value: Any) -> str:
    """Normalise a scope value to a trimmed, lower-cased string (G-3 parity)."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalise_scope(scope: dict[str, Any]) -> dict[str, str]:
    """Normalise every value in a scope dict."""
    return {k: _normalise_value(v) for k, v in (scope or {}).items()}


class CortexKBClient:
    """Read-only scoped-article-over-graph adapter for cortex ``kb-service``.

    Implements the :class:`~runtime.kb_client.KBClient` protocol so it is a
    drop-in replacement on the read path. Write methods raise
    :class:`KBValidationError` because the Critic only reads from cortex.
    Retry / backoff / error mapping mirror the HTTP client (contract §6).
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
        """Configure the cortex transport.

        Args:
            base_url (str): cortex ``kb-service`` base URL (typically
                ``$CORTEX_KB_URL``); trailing slash is stripped. The ``/v1``
                prefix is appended per endpoint.
            token (str | None): Bearer token; falls back to
                ``KB_SERVICE_TOKEN`` when omitted.
            timeout_ms (int): Per-request timeout in milliseconds.
            retry_max (int): Maximum retries on 429/5xx/network errors.
            backoff_base (float): Base seconds for exponential backoff.
            sleep_fn (Callable[[float], None]): Sleep function (injectable).

        Raises:
            ValueError: If ``base_url`` is empty.
        """
        if not base_url:
            raise ValueError("CortexKBClient: base_url is required")
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("KB_SERVICE_TOKEN") or ""
        self.timeout_s = float(timeout_ms) / 1000.0
        self.retry_max = retry_max
        self.backoff_base = backoff_base
        self._sleep = sleep_fn

    # ------------------------------------------------------------------
    # KBClient protocol — list (the only supported operation)
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
        """Read Critic priors via ``POST /v1/points/query``.

        ``scope_filter`` is matched with JSONB containment on
        ``attrs._critic_scope``; the presence of that key also restricts the
        result to Critic-authored rows. Cortex has no server-side
        ``sort_by``, so newest-first ordering is applied client-side using the
        stored ``_critic_updated_at`` stamp.

        Returns:
            dict[str, Any]: ``{"entries": [...], "count": n}`` with each entry
            reconstructed into the legacy scoped-article shape.
        """
        attrs_filter: dict[str, Any] = {_ATTR_SCOPE: _normalise_scope(scope_filter)}
        if metadata_filter:
            # Best-effort: nest the metadata filter under the reserved key so
            # containment still applies; cortex matches the subset with ``@>``.
            attrs_filter[_ATTR_METADATA] = metadata_filter
        body: dict[str, Any] = {
            "attrs_filter": attrs_filter,
            "neighbor_preview": False,
            # Pull at least the requested page so client-side sort + limit is
            # faithful even though cortex orders by id, not updated_at.
            "limit": max(int(limit), 1) if limit else 50,
        }
        if kind is not None:
            body["kind"] = _KIND_PREFIX + kind

        response = self._post("/v1/points/query", body, endpoint_label="list")
        points = response.get("points") or []
        entries = [self._point_to_entry(p) for p in points]
        entries.sort(
            key=lambda e: e.get("updated_at") or 0.0,
            reverse=(sort_by == "updated_at_desc"),
        )
        if limit:
            entries = entries[: int(limit)]
        return {"entries": entries, "count": len(entries)}

    # ------------------------------------------------------------------
    # KBClient protocol — writes (rejected: Critic is read-only on cortex)
    # ------------------------------------------------------------------
    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Reject writes — the Critic is read-only against cortex.

        Raises:
            KBValidationError: Always.
        """
        raise KBValidationError(_READ_ONLY_MSG)

    def batch_insert(
        self,
        items: list[dict[str, Any]],
        *,
        on_conflict: str = "upsert",
    ) -> dict[str, Any]:
        """Reject writes — the Critic is read-only against cortex.

        Raises:
            KBValidationError: Always.
        """
        raise KBValidationError(_READ_ONLY_MSG)

    def add_edges(self, edges: list[dict[str, Any]]) -> dict[str, Any]:
        """Reject writes — the Critic is read-only against cortex.

        Raises:
            KBValidationError: Always.
        """
        raise KBValidationError(_READ_ONLY_MSG)

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_kind(cortex_kind: str | None) -> str:
        kind = cortex_kind or ""
        return kind[len(_KIND_PREFIX):] if kind.startswith(_KIND_PREFIX) else kind

    def _point_to_entry(self, point: dict[str, Any]) -> dict[str, Any]:
        """Reconstruct the scoped-article entry shape from a cortex PointDto."""
        attrs = point.get("attrs") or {}
        return {
            "id": point.get("id"),
            "canonical_id": point.get("canonical_id"),
            "scope": attrs.get(_ATTR_SCOPE) or {},
            "kind": attrs.get(_ATTR_KIND) or self._strip_kind(point.get("kind")),
            "slug": attrs.get(_ATTR_SLUG) or "",
            "importance": attrs.get(_ATTR_IMPORTANCE),
            "summary": attrs.get(_ATTR_SUMMARY) or "",
            "metadata": attrs.get(_ATTR_METADATA) or {},
            "edges": {},
            "created_at": point.get("created_at"),
            "updated_at": attrs.get(_ATTR_UPDATED_AT),
            "deleted": not point.get("is_active", True),
        }

    # ------------------------------------------------------------------
    # Transport — POST with retry / backoff / error mapping
    # ------------------------------------------------------------------
    def _post(
        self, path: str, body: dict[str, Any], *, endpoint_label: str
    ) -> dict[str, Any]:
        """POST ``body`` to ``path`` with retry, backoff, and metrics.

        Mirrors :meth:`runtime.kb_client.HTTPKBClient._request`: retries on
        429/5xx/network errors up to ``retry_max`` times; non-retryable 4xx
        raise the matching typed error so the caller can degrade gracefully.

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
        last_error: Exception | None = None
        attempt = 0
        start = time.time()
        while attempt <= self.retry_max:
            attempt += 1
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8") or "{}"
                    body_obj = json.loads(raw) if raw else {}
                    registry.counter(CRITIC_KB_WRITE_TOTAL).inc(
                        {"endpoint": endpoint_label, "status": str(resp.status)}
                    )
                    registry.histogram(CRITIC_KB_WRITE_DURATION_SECONDS).observe(
                        time.time() - start, {"endpoint": endpoint_label}
                    )
                    return body_obj
            except urllib.error.HTTPError as exc:
                status = getattr(exc, "code", 0)
                err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                registry.counter(CRITIC_KB_WRITE_TOTAL).inc(
                    {"endpoint": endpoint_label, "status": str(status)}
                )
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
        """Exponential backoff with jitter to avoid a thundering herd."""
        base = self.backoff_base * (2 ** (attempt - 1))
        return base * (0.9 + 0.2 * random.random())


__all__ = ["CortexKBClient"]
