# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Cortex KB client: maps the Critic 4-endpoint contract onto cortex ``/v1``.

The Critic was written against the legacy scoped-article contract
(``POST /api/kb/{list,upsert,batch_insert,edges/add}``) served by
``claw-memory-service``. The cortex ``kb-service`` deprecated that surface
and exposes a graph paradigm instead (``/v1/points`` + ``/v1/edges``). This
client lets the Critic talk to cortex without changing any caller: it
implements the same :class:`~runtime.kb_client.KBClient` protocol and
translates each scoped-article operation into the matching cortex request.

Mapping rules
-------------
* The Critic's ``(scope, kind, slug)`` composite key is folded into a
  deterministic cortex ``canonical_id`` (``critic.{kind}.{slug}.{hash}``),
  which is globally unique in ``kb_points`` and therefore idempotent on
  re-upsert.
* The Critic ``kind`` is namespaced as ``critic_{kind}`` on the wire so it
  never collides with cortex's registered ``pitfall`` / ``lesson`` schemas
  (those enforce strict attrs validation on propose); namespaced kinds take
  the pass-through validation path.
* All Critic-specific fields (scope, slug, importance, summary, metadata)
  live under reserved ``_critic_*`` keys inside ``kb_points.attrs`` so reads
  can faithfully reconstruct the original scoped-article shape.
* ``scope_filter`` becomes a JSONB-containment ``attrs_filter`` on
  ``_critic_scope`` (cortex matches with ``@>``), which both filters by scope
  and restricts results to Critic-authored rows.

Writes carry the cortex-mandatory ``authority`` / ``evidence_refs`` /
``provenance`` envelope, synthesised from the Critic payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
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

# Cortex requires a write authority on every propose / ingest. Critic priors
# are reported observations, not authoritative ground truth.
_CRITIC_AUTHORITY = "EXPERIENTIAL"

# Reserved attrs keys used to round-trip the scoped-article shape.
_ATTR_SCOPE = "_critic_scope"
_ATTR_KIND = "_critic_kind"
_ATTR_SLUG = "_critic_slug"
_ATTR_IMPORTANCE = "_critic_importance"
_ATTR_SUMMARY = "_critic_summary"
_ATTR_METADATA = "_critic_metadata"
_ATTR_UPDATED_AT = "_critic_updated_at"

_KIND_PREFIX = "critic_"


def _normalise_value(value: Any) -> str:
    """Normalise a scope value to a trimmed, lower-cased string (G-3 parity)."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalise_scope(scope: dict[str, Any]) -> dict[str, str]:
    """Normalise every value in a scope dict."""
    return {k: _normalise_value(v) for k, v in (scope or {}).items()}


def _scope_key(scope: dict[str, str]) -> str:
    """Deterministic ``k=v|k=v`` string with keys sorted (stable hash input)."""
    return "|".join(f"{k}={scope[k]}" for k in sorted(scope.keys()))


def _canonical_id(scope: dict[str, str], kind: str, slug: str) -> str:
    """Fold ``(scope, kind, slug)`` into a stable cortex canonical_id."""
    digest = hashlib.sha256(_scope_key(scope).encode("utf-8")).hexdigest()[:12]
    return f"critic.{kind}.{slug}.{digest}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CortexKBClient:
    """Scoped-article-over-graph adapter for the cortex ``kb-service``.

    Implements the :class:`~runtime.kb_client.KBClient` protocol so it is a
    drop-in replacement for :class:`~runtime.kb_client.HTTPKBClient`. Retry /
    backoff / error mapping mirror the HTTP client (contract §6).
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
            base_url (str): cortex ``kb-service`` base URL (e.g.
                ``http://kb-service.primus-cortex:8080``); trailing slash is
                stripped. The ``/v1`` prefix is appended per endpoint.
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
    # KBClient protocol — list
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
            # Pull a wider page so client-side sort + limit is faithful.
            "limit": max(int(limit) if limit else 50, int(limit) or 1),
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
    # KBClient protocol — upsert
    # ------------------------------------------------------------------
    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Upsert a single Critic entry via ``POST /v1/points/propose``.

        Returns:
            dict[str, Any]: ``{"row": <entry>, "created": bool,
            "cortex": <propose response>}``.
        """
        request_body = self._build_point_request(payload)
        response = self._post(
            "/v1/points/propose", request_body, endpoint_label="upsert"
        )
        entry = self._payload_to_entry(payload, point_id=response.get("point_id"))
        return {
            "row": entry,
            "created": response.get("status") == "auto_accepted",
            "cortex": response,
        }

    # ------------------------------------------------------------------
    # KBClient protocol — batch_insert
    # ------------------------------------------------------------------
    def batch_insert(
        self,
        items: list[dict[str, Any]],
        *,
        on_conflict: str = "upsert",
    ) -> dict[str, Any]:
        """Batch-upsert Critic entries via ``POST /v1/bulk/ingest``.

        cortex bulk ingest is keyed by ``canonical_id`` and always merges on
        conflict, so ``on_conflict`` is accepted for protocol compatibility
        but only ``upsert`` semantics are available server-side. Any per-item
        rejection raises :class:`KBValidationError` so the whole batch is
        dead-lettered for replay.

        Returns:
            dict[str, Any]: ``{"results": [...], "count": n, "cortex": ...}``.

        Raises:
            KBValidationError: If ``on_conflict`` is invalid or cortex rejects
                one or more points.
        """
        if on_conflict not in ("upsert", "error"):
            raise KBValidationError(
                f"batch_insert: on_conflict must be upsert|error, got {on_conflict!r}"
            )
        body = {
            "points": [self._build_point_request(item) for item in items],
            "pipeline_id": "critic-kb",
            "batch_id": uuid.uuid4().hex,
        }
        response = self._post("/v1/bulk/ingest", body, endpoint_label="batch_insert")
        rejected = ((response.get("rejected") or {}).get("points")) or []
        if rejected:
            raise KBValidationError(f"batch_insert: cortex rejected points: {rejected!r}")
        results = [
            self._payload_to_entry(item, point_id=pid)
            for item, pid in zip(
                items, ((response.get("accepted") or {}).get("points")) or []
            )
        ]
        return {"results": results, "count": len(results), "cortex": response}

    # ------------------------------------------------------------------
    # KBClient protocol — add_edges
    # ------------------------------------------------------------------
    def add_edges(self, edges: list[dict[str, Any]]) -> dict[str, Any]:
        """Create ``contradicts`` edges via ``POST /v1/edges/negate``.

        cortex replaced the scoped-article ``contradicts`` auto-mirror with a
        directed ``negation`` edge. Endpoints are cortex integer point ids
        (as returned by :meth:`upsert`). Non-``contradicts`` kinds and
        non-integer endpoints are rejected.

        Returns:
            dict[str, Any]: ``{"added": [...], "cortex": [...]}``.

        Raises:
            KBValidationError: If an edge is malformed or of an unsupported
                kind.
        """
        added: list[dict[str, Any]] = []
        raw: list[dict[str, Any]] = []
        for edge in edges:
            kind = edge.get("kind")
            src = edge.get("from_id")
            dst = edge.get("to_id")
            if kind != "contradicts":
                raise KBValidationError(
                    f"add_edges: only 'contradicts' is supported on cortex, got {kind!r}"
                )
            try:
                from_point = int(src)
                to_point = int(dst)
            except (TypeError, ValueError) as exc:
                raise KBValidationError(
                    f"add_edges: cortex requires integer point ids, got {src!r}->{dst!r}"
                ) from exc
            body = {
                "from_point": from_point,
                "to_point": to_point,
                "reason": "critic_contradiction",
                "authority": _CRITIC_AUTHORITY,
                "evidence_refs": self._synthetic_evidence({}),
                "provenance": self._provenance(),
            }
            response = self._post("/v1/edges/negate", body, endpoint_label="edges/add")
            added.append({"from_id": from_point, "to_id": to_point, "kind": "negation"})
            raw.append(response)
        return {"added": added, "cortex": raw}

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------
    def _build_point_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Translate a scoped-article upsert payload into a cortex point."""
        for field in ("scope", "kind", "slug", "importance"):
            if field not in payload:
                raise KBValidationError(f"upsert: missing field {field!r}")
        scope = _normalise_scope(payload["scope"])
        kind = str(payload["kind"])
        slug = str(payload["slug"])
        metadata = payload.get("metadata") or {}
        attrs = {
            _ATTR_SCOPE: scope,
            _ATTR_KIND: kind,
            _ATTR_SLUG: slug,
            _ATTR_IMPORTANCE: float(payload["importance"]),
            _ATTR_SUMMARY: payload.get("summary", ""),
            _ATTR_METADATA: metadata,
            _ATTR_UPDATED_AT: time.time(),
        }
        return {
            "canonical_id": _canonical_id(scope, kind, slug),
            "kind": _KIND_PREFIX + kind,
            "attrs": attrs,
            "authority": _CRITIC_AUTHORITY,
            "evidence_refs": self._synthetic_evidence(metadata),
            "provenance": self._provenance(),
        }

    @staticmethod
    def _synthetic_evidence(metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """cortex requires a non-empty evidence list; synthesise a log ref."""
        source = (metadata or {}).get("source_session") or "critic-agent"
        return [
            {
                "kind": "log",
                "ref": f"critic-session:{source}",
                "note": "auto-generated by critic-agent KB write",
            }
        ]

    @staticmethod
    def _provenance() -> dict[str, Any]:
        return {
            "source": "agent_observation",
            "generator": "critic-agent",
            "generated_at": _now_iso(),
        }

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

    def _payload_to_entry(
        self, payload: dict[str, Any], *, point_id: Any = None
    ) -> dict[str, Any]:
        """Echo an upsert/batch payload back as a scoped-article entry."""
        scope = _normalise_scope(payload.get("scope") or {})
        kind = str(payload.get("kind") or "")
        slug = str(payload.get("slug") or "")
        return {
            "id": point_id,
            "canonical_id": _canonical_id(scope, kind, slug) if slug else None,
            "scope": scope,
            "kind": kind,
            "slug": slug,
            "importance": payload.get("importance"),
            "summary": payload.get("summary", ""),
            "metadata": payload.get("metadata") or {},
            "edges": {},
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
        raise the matching typed error so the caller can dead-letter.

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
