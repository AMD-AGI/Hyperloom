# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Read-only HTTP client for the central recipe-snapshot kb-service.

The local store (:class:`recipe_kb.LocalRecipeStore`) is the source
of truth for *writes*; the central kb-service is only ever consulted
for *reads* under this design (see commit message for the
discussion). This module is the read half of that contract:

* All write-side machinery from the prior v2 ``RecipeSnapshotClient``
  (NDJSON enqueue, drain_pending, dead_letter, flusher daemon) is
  intentionally absent.
* ``put_recipe`` / ``append_attempt`` / ``delete_recipe`` raise
  :class:`NotImplementedError` if a caller reaches in via the
  attribute name out of habit — better than silently no-op'ing.
* Read methods (``get_recipe`` / ``get_history`` / ``search`` /
  ``list_attempts`` / ``list_session_attempts`` / ``session_summary``)
  raise :class:`RemoteRecipeClientError` on transport or business
  errors so the dispatcher can degrade to the local store.

The transport layer keeps the same conservative defaults the prior
client used (2s foreground timeout + 1 retry, 10s + 3 retries
background) — operators who calibrated their alerting around those
numbers don't have to relearn them.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import httpx

from .. import recipe_snapshot_constants as C


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class RemoteRecipeClientError(RuntimeError):
    """Raised on any unrecoverable interaction with the central kb-service.

    The dispatcher catches this and degrades to the local store. Carries
    a ``category`` discriminator (``transport`` / ``business`` /
    ``validation`` / ``unknown``) so a future smarter dispatcher can
    decide between "retry with backoff" and "fall through immediately".
    """

    def __init__(
        self,
        message: str,
        *,
        category: str = "unknown",
        code: str = "",
        status: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the error with a category discriminator and context.

        Args:
            message (str): Human-readable error description.
            category (str): Failure class — one of ``transport`` /
                ``business`` / ``validation`` / ``unknown``.
            code (str): Machine-readable error code from the server
                envelope, if any.
            status (int | None): HTTP status code, if a response was
                received.
            details (Mapping[str, Any] | None): Extra structured error
                context; copied into ``self.details``.
        """
        super().__init__(message)
        self.category = category
        self.code = code
        self.status = status
        self.details = dict(details or {})


def _parse_error_envelope(
    resp: httpx.Response,
) -> tuple[str, str, str, dict[str, Any]]:
    """Decode the recipe-snapshot error envelope.

    Three categories per the API spec:
    * ``business``   — ``{"detail": {"error": {"code","message","details"}}}``;
    * ``validation`` — FastAPI default ``{"detail": [{loc,msg,type},...]}``;
    * ``unknown``    — anything else.

    Args:
        resp (httpx.Response): The error response to decode.

    Returns:
        tuple[str, str, str, dict[str, Any]]: ``(category, code,
            message, details)`` extracted from the envelope.
    """
    try:
        body = resp.json()
    except (ValueError, json.JSONDecodeError):
        return ("unknown", "UNKNOWN", (resp.text or "")[:512], {})
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict) and "error" in detail:
        err = detail["error"]
        return (
            "business",
            str(err.get("code", "")),
            str(err.get("message", "")),
            dict(err.get("details") or {}),
        )
    if isinstance(detail, list):
        parts: list[str] = []
        for item in detail:
            if not isinstance(item, Mapping):
                continue
            loc = ".".join(str(p) for p in (item.get("loc") or []))
            msg = str(item.get("msg") or "")
            parts.append(f"{loc}: {msg}".strip(": "))
        return (
            "validation",
            "VALIDATION_ERROR",
            "; ".join(parts) or "validation failed",
            {"raw": detail},
        )
    return ("unknown", "UNKNOWN", json.dumps(body)[:512], {"raw": body})


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------
@dataclass
class _HttpTransport:
    """Thin wrapper around ``httpx.Client`` with retry + concurrency cap.

    Identical retry / backoff policy to the legacy v2 client so
    operators don't have to relearn timing.
    """

    base_url: str
    timeout_sec: float
    token: str | None = None
    max_connections: int = C.DEFAULT_MAX_CONCURRENCY
    retry_attempts: int = C.DEFAULT_RETRY_ATTEMPTS

    _client: httpx.Client | None = field(default=None, init=False, repr=False)
    _semaphore: threading.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialise the concurrency-limiting semaphore.

        Sizes a ``threading.Semaphore`` to ``max_connections`` (at
        least 1) so ``request`` can cap simultaneous in-flight calls.
        """
        self._semaphore = threading.Semaphore(max(1, int(self.max_connections)))

    def _ensure_client(self) -> httpx.Client:
        """Lazily build and cache the underlying ``httpx.Client``.

        Created on first use so constructing the transport performs
        no network setup. Sets the User-Agent and optional bearer
        ``Authorization`` header and applies the connection limits.

        Returns:
            httpx.Client: The cached client instance.
        """
        if self._client is None:
            headers: dict[str, str] = {
                "User-Agent": "hyperloom-recipe-kb-remote",
            }
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_sec,
                limits=httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_connections,
                ),
                headers=headers,
            )
        return self._client

    def close(self) -> None:
        """Close and drop the cached ``httpx.Client`` if present.

        Best-effort: any error raised while closing is swallowed so
        shutdown never fails on a half-open client.
        """
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001 — best-effort close
                pass
            self._client = None

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        allow_404: bool = False,
    ) -> dict[str, Any] | None:
        """Issue ``method`` against ``path`` with transient-error retry.

        Returns the parsed JSON response (always a dict; bare lists
        are wrapped under ``_value``). ``allow_404=True`` makes a
        404 NOT_FOUND return ``None`` instead of raising — used by
        ``get_recipe`` where "row absent" is a normal state.

        Args:
            method (str): HTTP method (``GET`` / ``POST`` / ...).
            path (str): Request path relative to ``base_url``.
            body (Mapping[str, Any] | None): JSON request body.
            params (Mapping[str, Any] | None): Query params; ``None``
                values are dropped.
            allow_404 (bool): When ``True``, a 404 returns ``None``
                instead of raising.

        Returns:
            dict[str, Any] | None: Parsed JSON object (bare lists
                wrapped under ``_value``); ``{}`` for empty/204
                responses; ``None`` for an allowed 404.

        Raises:
            RemoteRecipeClientError: On a non-404 4xx, exhausted
                transport retries, or a non-JSON response.
        """
        client = self._ensure_client()
        last_exc: Exception | None = None
        attempts = max(1, int(self.retry_attempts))
        kwargs: dict[str, Any] = {}
        if body is not None:
            kwargs["json"] = dict(body)
        if params is not None:
            # Drop ``None`` values so callers can pass through
            # optional query params unconditionally.
            kwargs["params"] = {
                k: v for k, v in dict(params).items() if v is not None
            }
        for attempt in range(attempts):
            with self._semaphore:
                try:
                    response = client.request(method, path, **kwargs)
                except (
                    httpx.TimeoutException,
                    httpx.ConnectError,
                    httpx.ReadError,
                ) as exc:
                    last_exc = exc
                    self._backoff(attempt)
                    continue
                if response.status_code >= 500:
                    last_exc = RemoteRecipeClientError(
                        f"transport 5xx on {method} {path}: "
                        f"{response.status_code}",
                        category="transport", status=response.status_code,
                    )
                    self._backoff(attempt)
                    continue
                if response.status_code == 404 and allow_404:
                    return None
                if response.status_code >= 400:
                    category, code, message, details = _parse_error_envelope(response)
                    raise RemoteRecipeClientError(
                        f"{method} {path} -> {response.status_code}: {message}",
                        category=category, code=code,
                        status=response.status_code, details=details,
                    )
                if response.status_code == 204 or not response.content:
                    return {}
                try:
                    parsed = response.json()
                except (ValueError, json.JSONDecodeError) as exc:
                    raise RemoteRecipeClientError(
                        f"{method} {path}: response not JSON ({exc})",
                        category="unknown",
                        status=response.status_code,
                    ) from exc
                return (
                    parsed if isinstance(parsed, dict)
                    else {"_value": parsed}
                )
        raise RemoteRecipeClientError(
            f"transport_exhausted after {attempts} attempts: "
            f"{method} {path}: {last_exc}",
            category="transport",
        )

    @staticmethod
    def _backoff(attempt: int) -> None:
        """Sleep for the retry backoff interval for ``attempt``.

        Uses the legacy cortex multiplier schedule (200 ms x {1, 1.4,
        4}); the multiplier index is clamped to the last entry for
        higher attempt numbers.

        Args:
            attempt (int): Zero-based retry attempt index.
        """
        # 200 ms x {1, 1.4, 4} — matches the legacy cortex client.
        multipliers = (1.0, 1.4, 4.0)
        idx = min(attempt, len(multipliers) - 1)
        time.sleep(C.DEFAULT_RETRY_BASE_MS * multipliers[idx] / 1000.0)


# ---------------------------------------------------------------------------
# RemoteRecipeClient — read-only surface
# ---------------------------------------------------------------------------
@dataclass
class RemoteRecipeClient:
    """Read-only HTTP client for the central recipe-snapshot kb-service.

    Construction is cheap — no network I/O happens until the first
    method is called. ``enabled=False`` makes every method a
    no-op-style return (None / empty list / False) so ``--degraded-kb``
    or "no --cortex-kb-url passed" can wire a None-or-disabled
    instance into the dispatcher and the caller path stays uniform.

    Args:
        kb_url: Central kb-service URL (read-only). When neither this
            nor ``CORTEX_KB_URL`` is set the client forces
            ``enabled=False`` (local-only) — there is NO hard-coded
            default endpoint. A remote KB is consulted only when an
            operator explicitly supplies a URL.
        timeout_sec: Per-HTTP-call timeout. Env override
            ``CORTEX_KB_HTTP_TIMEOUT_SEC``. Defaults to the
            ``foreground=True`` profile (2s) since the dispatcher's
            main use case is the optimizer hot-path; flusher /
            background callers can pass a higher value explicitly.
        retry_attempts: Transient retry budget. Defaults to 1 in
            foreground, 3 in background (mirrors legacy cortex
            client tuning).
        enabled: ``False`` short-circuits every read to the
            "no info" return value (None / [] / False) without
            making a network call. The dispatcher uses this when no
            CLI URL was provided OR when ``--degraded-kb`` was set.
        max_connections: client-side cap; aligns with kb-service
            asyncpg pool (8).
        token: ``Authorization: Bearer`` header value (env
            ``KB_SERVICE_TOKEN``); ``None`` omits the header.
        foreground: ``True`` for hot-path callers (Coordinator
            warm-start lookup); ``False`` for one-off boot-time
            probes that can wait the full 10s timeout.
    """

    kb_url: str | None = None
    timeout_sec: float | None = None
    enabled: bool = True
    max_connections: int = C.DEFAULT_MAX_CONCURRENCY
    token: str | None = None
    foreground: bool = True
    retry_attempts: int | None = None

    _transport: _HttpTransport | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve config from explicit args then environment.

        Fills ``kb_url`` from ``CORTEX_KB_URL`` (forcing
        ``enabled=False`` when still unset), picks foreground vs.
        background defaults for timeout and retries, and applies the
        ``CORTEX_KB_*`` / ``KB_SERVICE_TOKEN`` environment overrides.
        """
        if not self.kb_url:
            self.kb_url = (os.environ.get("CORTEX_KB_URL") or "").strip() or None
        if not self.kb_url:
            # No URL configured anywhere. The old central-service
            # default was retired, so there is nothing to fall back
            # to: force local-only. A disabled client short-circuits
            # every read to "no info" without a network call —
            # identical to ``remote=None`` in the dispatcher.
            # Operators opt in to a remote KB only by passing a URL.
            self.enabled = False
        if self.timeout_sec is None:
            profile_default = (
                C.FOREGROUND_HTTP_TIMEOUT_SEC if self.foreground
                else C.DEFAULT_HTTP_TIMEOUT_SEC
            )
            self.timeout_sec = profile_default
        env_timeout = os.environ.get("CORTEX_KB_HTTP_TIMEOUT_SEC")
        if env_timeout:
            try:
                self.timeout_sec = float(env_timeout)
            except ValueError:
                pass
        if self.retry_attempts is None:
            profile_default_retry = (
                C.FOREGROUND_RETRY_ATTEMPTS if self.foreground
                else C.DEFAULT_RETRY_ATTEMPTS
            )
            self.retry_attempts = profile_default_retry
        env_retry = os.environ.get("CORTEX_KB_RETRY_ATTEMPTS")
        if env_retry:
            try:
                self.retry_attempts = int(env_retry)
            except ValueError:
                pass
        env_conc = os.environ.get("CORTEX_KB_MAX_CONCURRENCY")
        if env_conc:
            try:
                self.max_connections = int(env_conc)
            except ValueError:
                pass
        if self.token is None:
            self.token = os.environ.get("KB_SERVICE_TOKEN") or None

    def close(self) -> None:
        """Close the underlying transport if one was created.

        Safe to call when no transport has been lazily built yet
        (no-op in that case).
        """
        if self._transport is not None:
            self._transport.close()

    def _ensure_transport(self) -> _HttpTransport:
        """Lazily build and cache the HTTP transport.

        Constructed on first read so a disabled client never builds
        one. Strips a trailing slash from ``kb_url`` for the base URL.

        Returns:
            _HttpTransport: The cached transport instance.
        """
        if self._transport is None:
            self._transport = _HttpTransport(
                base_url=str(self.kb_url).rstrip("/"),
                timeout_sec=self.timeout_sec,
                token=self.token,
                max_connections=self.max_connections,
                retry_attempts=int(self.retry_attempts or C.DEFAULT_RETRY_ATTEMPTS),
            )
        return self._transport

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    def health(self) -> bool:
        """One-shot probe of ``GET /health``.

        Any failure (including disabled-client short-circuit)
        returns ``False`` so the caller can decide whether to
        downgrade. Never raises.

        Returns:
            bool: ``True`` iff the service replies 200 with a
                ``{"status": "ok"}`` body; ``False`` on any failure or
                when the client is disabled.
        """
        if not self.enabled:
            return False
        try:
            resp = self._ensure_transport().request("GET", C.PATH_HEALTH)
        except RemoteRecipeClientError:
            return False
        if not isinstance(resp, dict):
            return False
        return str(resp.get("status") or "") == "ok"

    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """``GET /recipe-snapshot/recipes/{canonical_id}``.

        Returns the live (or ``?version=N`` archived) recipe dict, or
        ``None`` when:

        * client is disabled (``--degraded-kb``);
        * server returned 404 (canonical_id absent or no archive at
          ``?version=N``).

        Args:
            canonical_id (str): Canonical recipe identity; must be
                non-empty.
            version (int | None): Specific archived version, or
                ``None`` for the live row.

        Returns:
            dict[str, Any] | None: Recipe dict, or ``None`` when the
                client is disabled or the server returns 404.

        Raises:
            ValueError: If ``canonical_id`` is empty.
            RemoteRecipeClientError: On transport / 422 / non-404 4xx
                — the dispatcher catches and falls through to the
                local store.
        """
        if not self.enabled:
            return None
        if not canonical_id:
            raise ValueError("get_recipe requires a non-empty canonical_id")
        path = C.format_recipe_path(C.PATH_RECIPE_TPL, canonical_id)
        params: dict[str, Any] = {}
        if version is not None:
            params[C.F_VERSION] = int(version)
        resp = self._ensure_transport().request(
            "GET", path,
            params=params if params else None,
            allow_404=True,
        )
        if resp is None:
            return None
        return dict(resp)

    def get_history(
        self,
        *,
        canonical_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """``GET /recipe-snapshot/recipes/{canonical_id}/history``.

        Returns the ``history`` array verbatim. Server replies with an
        empty array for unknown ids (no 404 here per spec) so this
        method never raises on absence; disabled client returns ``[]``.

        Args:
            canonical_id (str): Canonical recipe identity; must be
                non-empty.
            limit (int): Maximum number of archived rows to request.

        Returns:
            list[dict[str, Any]]: Archived versions, or ``[]`` when
                disabled / absent.

        Raises:
            ValueError: If ``canonical_id`` is empty.
        """
        if not self.enabled:
            return []
        if not canonical_id:
            raise ValueError("get_history requires a non-empty canonical_id")
        path = C.format_recipe_path(C.PATH_RECIPE_HISTORY_TPL, canonical_id)
        resp = self._ensure_transport().request(
            "GET", path, params={C.F_LIMIT: int(limit)},
        )
        if not isinstance(resp, dict):
            return []
        history = resp.get(C.F_HISTORY)
        return list(history) if isinstance(history, list) else []

    def list_recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """``GET /recipe-snapshot/recipes`` — recent listing,
        ``updated_at DESC``.

        Args:
            limit (int): Maximum number of recipes to request.

        Returns:
            list[dict[str, Any]]: Recent recipe rows, or ``[]`` when
                the client is disabled.
        """
        if not self.enabled:
            return []
        resp = self._ensure_transport().request(
            "GET", C.PATH_RECIPES_LIST,
            params={C.F_LIMIT: int(limit)},
        )
        if not isinstance(resp, dict):
            return []
        rows = resp.get(C.F_RECIPES)
        return list(rows) if isinstance(rows, list) else []

    def search(
        self,
        *,
        label_match: dict[str, Any] | None = None,
        metric_filters: dict[str, Any] | None = None,
        updated_since: str | None = None,
        order_by: str = C.ORDER_BY_UPDATED_AT_DESC,
        limit: int = 50,
        prefer: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """``POST /recipe-snapshot/recipes/search``.

        Server-side validates ``order_by`` against the 6-value
        whitelist; we forward whatever was passed and trust the
        server to reject. Disabled client returns ``[]``.

        ``prefer`` (workload-similarity hints) is accepted for the
        unified KB-interface signature. The central kb-service has no
        server-side prefer ranking, so the dispatcher applies a
        client-side rerank over the returned rows; this client only
        forwards the ``required`` filter.

        Args:
            label_match: Identity labels to filter on.
            metric_filters: ``{metric: {min, max}}`` bounds to apply.
            updated_since: Lower bound on ``updated_at``.
            order_by: Ordering directive forwarded to the server.
            limit: Maximum number of rows to request.
            prefer: Accepted for signature parity; ignored here (rerank lives
                in the dispatcher).

        Returns:
            The matching recipe rows, or ``[]`` when the client is disabled.
        """
        del prefer  # client-side rerank lives in RecipeKB
        if not self.enabled:
            return []
        body: dict[str, Any] = {
            C.F_ORDER_BY: order_by,
            C.F_LIMIT:    int(limit),
        }
        if label_match:
            body[C.F_LABEL_MATCH] = dict(label_match)
        if metric_filters:
            body[C.F_METRIC_FILTERS] = dict(metric_filters)
        if updated_since:
            body[C.F_UPDATED_SINCE] = str(updated_since)
        resp = self._ensure_transport().request(
            "POST", C.PATH_RECIPES_SEARCH, body=body,
        )
        if not isinstance(resp, dict):
            return []
        rows = resp.get(C.F_RECIPES)
        return list(rows) if isinstance(rows, list) else []

    def list_attempts(
        self,
        *,
        canonical_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """``GET /recipe-snapshot/recipes/{canonical_id}/attempts``.

        Spec defaults to newest-first.

        Args:
            canonical_id (str): Canonical recipe identity; must be
                non-empty.
            limit (int): Maximum number of attempts to request.

        Returns:
            list[dict[str, Any]]: Attempt rows, or ``[]`` when the
                client is disabled.

        Raises:
            ValueError: If ``canonical_id`` is empty.
        """
        if not self.enabled:
            return []
        if not canonical_id:
            raise ValueError("list_attempts requires a non-empty canonical_id")
        path = C.format_recipe_path(C.PATH_RECIPE_ATTEMPTS_TPL, canonical_id)
        resp = self._ensure_transport().request(
            "GET", path, params={C.F_LIMIT: int(limit)},
        )
        if not isinstance(resp, dict):
            return []
        rows = resp.get("attempts")
        return list(rows) if isinstance(rows, list) else []

    def list_session_attempts(
        self,
        *,
        session_id: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """``GET /recipe-snapshot/sessions/{session_id}/attempts``.

        Spec defaults to oldest-first (chronological for plotting).

        Args:
            session_id (str): Session identity; must be non-empty.
            limit (int): Maximum number of attempts to request.

        Returns:
            list[dict[str, Any]]: Attempt rows for the session, or
                ``[]`` when the client is disabled.

        Raises:
            ValueError: If ``session_id`` is empty.
        """
        if not self.enabled:
            return []
        if not session_id:
            raise ValueError(
                "list_session_attempts requires a non-empty session_id",
            )
        path = C.PATH_SESSION_ATTEMPTS_TPL.replace(
            "{session_id}", session_id,
        )
        resp = self._ensure_transport().request(
            "GET", path, params={C.F_LIMIT: int(limit)},
        )
        if not isinstance(resp, dict):
            return []
        rows = resp.get("attempts")
        return list(rows) if isinstance(rows, list) else []

    def session_summary(
        self,
        *,
        session_id: str,
    ) -> dict[str, Any] | None:
        """``GET /recipe-snapshot/sessions/{session_id}/summary``.

        The server returns ``total_attempts=0`` for unknown sessions
        (not 404), so absence is a normal-shape dict.

        Args:
            session_id (str): Session identity; must be non-empty.

        Returns:
            dict[str, Any] | None: Per-session roll-up dict, or
                ``None`` when the client is disabled.

        Raises:
            ValueError: If ``session_id`` is empty.
        """
        if not self.enabled:
            return None
        if not session_id:
            raise ValueError(
                "session_summary requires a non-empty session_id",
            )
        path = C.PATH_SESSION_SUMMARY_TPL.replace(
            "{session_id}", session_id,
        )
        resp = self._ensure_transport().request("GET", path)
        return dict(resp) if isinstance(resp, dict) else None


__all__ = [
    "RemoteRecipeClient",
    "RemoteRecipeClientError",
]
