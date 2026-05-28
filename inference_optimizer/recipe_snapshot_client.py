"""recipe-snapshot v2 HTTP client — single point of contact between
``inference_optimizer`` and the kb-service ``/recipe-snapshot/*`` API.

Replaces the legacy :mod:`inference_optimizer.cortex_kb_client` which
spoke the pre-v2 ``/v1/points`` graph surface. See
``primus-cortex-internal/docs/recipe-snapshot-api-reference.md`` for
the wire contract and CHANGELOG for the cutover rationale.

Three guarantees:

1. **All writes are channeled through the Coordinator**, never the
   reactor LLMs. PolicyGate enforces this; the facade itself doesn't
   check ACLs but every entrypoint takes a logical operation name so
   audit logs ascribe writes to ``inference_optimizer.coordinator``.
2. **Failure modes are well-defined.** Synchronous HTTP failures on
   write ops (``PUT /recipes/{cid}``) fall through to a per-session
   NDJSON queue under ``runtime/recipe_snapshot/.pending.ndjson``; a
   background flusher (Phase 3 of the cutover) drains the queue and
   the synchronous ``drain_pending`` helper is called at session
   close. Read ops (``GET /recipes/{cid}``, ``GET /history``) NEVER
   fall through — the caller is expected to catch
   :class:`RecipeSnapshotError` and degrade (skip warm-start, keep
   the cumulative data in-memory, etc.).
3. **Side-channel contract.** A slow / unreachable kb-service must
   not block the main optimizer loop. The foreground client profile
   uses a 2 s timeout + 1 retry (vs 10 s × 3 in background) so the
   worst-case main-loop stall is ~2.5 s per write, after which the
   row is queued for the flusher.

Wire schema: recipe-snapshot v2 (Final[str] constants in
:mod:`.recipe_snapshot_constants`).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import httpx

from . import recipe_snapshot_constants as C
from .session_paths import (
    recipe_snapshot_audit_jsonl,
    recipe_snapshot_dir,
    recipe_snapshot_pending_ndjson,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class RecipeSnapshotError(RuntimeError):
    """Raised for unrecoverable interactions with kb-service.

    Synchronous T0 (warm-start) treats this as fail-soft for the
    Coordinator (skip warm-start, log a warning) and fail-fast for
    the cli boot path (``sys.exit(2)``). Synchronous write ops
    (``put_recipe``) catch it internally and downgrade to an NDJSON
    enqueue so the Coordinator dispatcher never crashes on a
    transient kb-service outage.

    ``category`` discriminates business / validation / transport /
    unknown so callers can decide retry vs surface vs dead-letter.
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
        super().__init__(message)
        self.category = category
        self.code = code
        self.status = status
        self.details = dict(details or {})


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _ndjson_envelope(
    *,
    op: str,
    payload: Mapping[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    """Build one NDJSON row.

    Shape mirrors the legacy ``cortex`` envelope so the Phase 3
    flusher can consume both legacy + new rows side-by-side during
    the brief window where on-disk queues may carry leftover legacy
    entries (we expect to ship the flusher cutover in the same
    branch, but the on-disk format stays stable to make replay /
    forensic recovery one-grep simpler).
    """
    return {
        "op":              op,
        "payload":         dict(payload),
        "created_at":      _now_iso(),
        "idempotency_key": idempotency_key,
        "attempts":        0,
    }


def parse_error_envelope(
    resp: httpx.Response,
) -> tuple[str, str, str, dict[str, Any]]:
    """Parse the recipe-snapshot error envelope.

    Three categories per spec:

    * ``business``   — ``{"detail": {"error": {"code", "message", "details"}}}``
                       (400 / 404 from the recipe-snapshot router).
    * ``validation`` — ``{"detail": [{loc, msg, type}, ...]}``
                       (FastAPI's default 422 shape — pydantic
                       rejected the request body).
    * ``unknown``    — anything else (network, gateway, etc.).

    Returns ``(category, code, human_message, details_dict)``.
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

    Identical retry / backoff policy to the legacy cortex client so
    operators don't have to relearn timing. ``retry_attempts`` is
    passed by the client (foreground = 1 fail-fast, background = 3
    give-blips-a-chance).
    """

    base_url: str
    timeout_sec: float
    token: str | None = None
    max_connections: int = C.DEFAULT_MAX_CONCURRENCY
    retry_attempts: int = C.DEFAULT_RETRY_ATTEMPTS

    _client: httpx.Client | None = field(default=None, init=False, repr=False)
    _semaphore: threading.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._semaphore = threading.Semaphore(max(1, int(self.max_connections)))

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            headers: dict[str, str] = {
                "User-Agent": "hyperloom-recipe-snapshot-client",
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
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001 — best-effort close
                pass
            self._client = None

    # ------------------------------------------------------------------
    # Verb helpers
    # ------------------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        allow_404: bool = False,
    ) -> dict[str, Any] | None:
        """Issue ``method`` against ``path``; retry transient errors.

        Returns the parsed JSON response (always a dict; bare lists
        / scalars are wrapped under ``_value``). Raises
        :class:`RecipeSnapshotError` on:

        * exhausted retries (transient 5xx / connect / timeout);
        * 4xx business error (parsed via :func:`parse_error_envelope`);
        * 422 validation error (parsed via :func:`parse_error_envelope`).

        ``allow_404=True`` makes a ``404 NOT_FOUND`` body return
        ``None`` instead of raising — used by ``get_recipe`` /
        ``get_history`` where "row absent" is a normal state.
        """
        client = self._ensure_client()
        last_exc: Exception | None = None
        attempts = max(1, int(self.retry_attempts))
        kwargs: dict[str, Any] = {}
        if body is not None:
            kwargs["json"] = dict(body)
        if params is not None:
            # Drop ``None`` values so callers can pass through optional
            # query params unconditionally.
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
                    last_exc = RecipeSnapshotError(
                        f"transport 5xx on {method} {path}: "
                        f"{response.status_code}",
                        category="transport", status=response.status_code,
                    )
                    self._backoff(attempt)
                    continue
                if response.status_code == 404 and allow_404:
                    return None
                if response.status_code >= 400:
                    category, code, message, details = parse_error_envelope(response)
                    raise RecipeSnapshotError(
                        f"{method} {path} → {response.status_code}: {message}",
                        category=category, code=code,
                        status=response.status_code, details=details,
                    )
                if response.status_code == 204 or not response.content:
                    return {}
                try:
                    parsed = response.json()
                except (ValueError, json.JSONDecodeError) as exc:
                    raise RecipeSnapshotError(
                        f"{method} {path}: response not JSON ({exc})",
                        category="unknown",
                        status=response.status_code,
                    ) from exc
                return (
                    parsed if isinstance(parsed, dict)
                    else {"_value": parsed}
                )
        # Retry budget exhausted.
        raise RecipeSnapshotError(
            f"transport_exhausted after {attempts} attempts: "
            f"{method} {path}: {last_exc}",
            category="transport",
        )

    @staticmethod
    def _backoff(attempt: int) -> None:
        # 200 ms × {1, 1.4, 4} — matches the legacy cortex client.
        multipliers = (1.0, 1.4, 4.0)
        idx = min(attempt, len(multipliers) - 1)
        time.sleep(C.DEFAULT_RETRY_BASE_MS * multipliers[idx] / 1000.0)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
@dataclass
class RecipeSnapshotClient:
    """Single-process recipe-snapshot client used by the Coordinator.

    Construction is cheap — no network I/O happens until the first
    wire call. NDJSON enqueue is O(append) regardless.

    Args:
        session_dir: hyperloom session root (writable). Hosts the
            audit log + NDJSON queue under
            ``<sd>/runtime/recipe_snapshot/``.
        kb_url: ``CORTEX_KB_URL`` override; ``None`` reads env or
            :data:`recipe_snapshot_constants.DEFAULT_KB_URL`. (The
            env var name is shared with the legacy client on
            purpose — operators have one variable to set.)
        timeout_sec: per-HTTP-call timeout. Env override
            ``CORTEX_KB_HTTP_TIMEOUT_SEC``. Defaults adjust based on
            ``foreground``: 2 s for main-loop callers, 10 s for the
            flusher daemon.
        retry_attempts: how many transient retries. Defaults: 1 in
            foreground (fail-fast to NDJSON), 3 in background.
        enabled: when ``False`` every entrypoint becomes a no-op
            (used by ``--degraded-kb``). Audit log still records the
            skip so breakdown collection can flag the bypass.
        max_connections: client-side cap; aligns with kb-service
            asyncpg pool=8.
        token: ``Authorization: Bearer`` value
            (env ``KB_SERVICE_TOKEN``); ``None`` omits the header
            (H1 anonymous).
        smoke: stamp ``provenance.generator = SMOKE_GENERATOR`` on
            every write (env ``CORTEX_KB_SMOKE``).
        initiator: ``provenance.generator`` value when not smoke.
        foreground: ``True`` when used on the Coordinator main loop
            (KEEP / REVERT writes). ``False`` for the kb_flusher
            daemon / CLI boot path. Drives the timeout + retry
            profile defaults.
    """

    session_dir: Path
    kb_url: str | None = None
    timeout_sec: float | None = None
    enabled: bool = True
    max_connections: int = C.DEFAULT_MAX_CONCURRENCY
    token: str | None = None
    smoke: bool = False
    initiator: str = C.DEFAULT_GENERATOR
    foreground: bool = False
    retry_attempts: int | None = None

    _transport: _HttpTransport | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.session_dir = Path(self.session_dir)
        recipe_snapshot_dir(self.session_dir).mkdir(parents=True, exist_ok=True)
        if not self.kb_url:
            # Shared env var with the legacy client — operators
            # configure one URL, not two.
            self.kb_url = (
                os.environ.get("CORTEX_KB_URL") or C.DEFAULT_KB_URL
            )
        # Resolve timeout: caller-supplied > env override > profile default.
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
        # Same resolution for retry budget.
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
        if not self.smoke:
            self.smoke = (
                (os.environ.get("CORTEX_KB_SMOKE") or "")
                .strip().lower() in ("1", "true", "yes", "on")
            )

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    @property
    def pending_path(self) -> Path:
        return recipe_snapshot_pending_ndjson(self.session_dir)

    @property
    def audit_path(self) -> Path:
        return recipe_snapshot_audit_jsonl(self.session_dir)

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()

    def _ensure_transport(self) -> _HttpTransport:
        if self._transport is None:
            self._transport = _HttpTransport(
                base_url=str(self.kb_url).rstrip("/"),
                timeout_sec=self.timeout_sec,
                token=self.token,
                max_connections=self.max_connections,
                retry_attempts=int(self.retry_attempts or C.DEFAULT_RETRY_ATTEMPTS),
            )
        return self._transport

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        allow_404: bool = False,
    ) -> dict[str, Any] | None:
        transport = self._ensure_transport()
        started = time.monotonic()
        try:
            result = transport.request(
                method, path, body=body, params=params, allow_404=allow_404,
            )
        except RecipeSnapshotError as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            self._audit_record(
                op="http", status=exc.category or "error",
                method=method, path=path, elapsed_ms=elapsed_ms,
                error_code=exc.code, error_status=exc.status,
                error_message=str(exc)[:512],
            )
            raise
        elapsed_ms = int((time.monotonic() - started) * 1000)
        self._audit_record(
            op="http", status="ok",
            method=method, path=path, elapsed_ms=elapsed_ms,
        )
        return result

    def _audit_record(self, **fields: Any) -> None:
        """Append one structured line to ``.audit.jsonl``.

        Best-effort: a write failure is logged but never raised — the
        audit log is a forensic aid, not a correctness invariant.
        """
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            row = {"ts": _now_iso(), **fields}
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError as exc:
            log.warning(
                "recipe_snapshot audit append failed (%s): %s",
                self.audit_path, exc,
            )

    # ------------------------------------------------------------------
    # NDJSON enqueue (fallback path for failed writes)
    # ------------------------------------------------------------------
    def _enqueue(
        self,
        *,
        op: str,
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> None:
        envelope = _ndjson_envelope(
            op=op, payload=payload, idempotency_key=idempotency_key,
        )
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(envelope, sort_keys=True) + "\n"
        with self.pending_path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync is best-effort: tmpfs / certain wekafs mounts
                # reject it but the write is still visible.
                pass
        self._audit_record(
            op="enqueue", status="ok",
            envelope_op=op, idempotency_key=idempotency_key,
        )

    # ------------------------------------------------------------------
    # Provenance / EvidenceRef helpers
    # ------------------------------------------------------------------
    def _provenance(
        self,
        *,
        source: str | None = None,
        details: Mapping[str, Any] | None = None,
        generator_suffix: str = "",
    ) -> dict[str, Any]:
        """Build a ``Provenance`` dict satisfying the strict server schema.

        ``source`` / ``generator`` / ``generated_at`` are required.
        Smoke mode swaps the generator for ``SMOKE_GENERATOR`` so
        production dashboards can filter probe writes out.
        """
        generator = (
            f"{C.SMOKE_GENERATOR}@{generator_suffix}"
            if self.smoke and generator_suffix
            else C.SMOKE_GENERATOR if self.smoke
            else self.initiator
        )
        prov: dict[str, Any] = {
            C.F_PV_SOURCE:       source or C.DEFAULT_SOURCE,
            C.F_PV_GENERATOR:    generator,
            C.F_PV_GENERATED_AT: _now_iso(),
        }
        if details:
            prov[C.F_PV_DETAILS] = dict(details)
        return prov

    @staticmethod
    def _normalise_evidence_refs(
        evidence: list[Any] | None,
    ) -> list[dict[str, Any]]:
        """Convert mixed-shape evidence into strict ``EvidenceRef`` dicts.

        Accepts:
        * Strings ``"<kind>:<ref>"`` (``log:``, ``commit:``, ``url:``,
          ``profile_file:``) → split into ``{kind, ref}``.
        * Plain strings → wrapped as ``{kind: "log", ref: <string>}``.
        * Pre-shaped dicts → passed through (only ``kind`` / ``ref`` /
          ``note`` survive, anything else is dropped to keep
          ``extra="forbid"`` happy).
        """
        out: list[dict[str, Any]] = []
        known_kinds = {
            C.EV_KIND_URL, C.EV_KIND_COMMIT,
            C.EV_KIND_PROFILE_FILE, C.EV_KIND_LOG,
        }
        for raw in (evidence or []):
            if isinstance(raw, Mapping):
                kind = str(raw.get(C.F_EV_KIND) or C.EV_KIND_LOG)
                if kind not in known_kinds:
                    kind = C.EV_KIND_LOG
                ref = str(raw.get(C.F_EV_REF) or "")
                if not ref:
                    continue
                row: dict[str, Any] = {C.F_EV_KIND: kind, C.F_EV_REF: ref}
                note = raw.get(C.F_EV_NOTE)
                if note is not None:
                    row[C.F_EV_NOTE] = str(note)
                out.append(row)
                continue
            text = str(raw)
            if not text:
                continue
            kind = C.EV_KIND_LOG
            ref = text
            if ":" in text:
                prefix, _, rest = text.partition(":")
                if prefix in known_kinds:
                    kind = prefix
                    ref = rest
            out.append({C.F_EV_KIND: kind, C.F_EV_REF: ref})
        return out

    # ==================================================================
    # Public API
    # ==================================================================
    def health(self) -> bool:
        """``GET /health`` smoke check.

        Returns ``True`` iff the service responds 200 with the
        ``{"status": "ok"}`` body the preflight script expects.
        Failures (including disabled-client short-circuit) return
        ``False``; the caller decides whether to mark the run
        degraded.
        """
        if not self.enabled:
            return False
        try:
            resp = self._request("GET", C.PATH_HEALTH)
        except RecipeSnapshotError:
            return False
        if not isinstance(resp, dict):
            return False
        return str(resp.get("status") or "") == "ok"

    def put_recipe(
        self,
        *,
        canonical_id: str,
        labels: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        findings: list[Any] | None = None,
        failures: list[Any] | None = None,
        pitfalls: list[Any] | None = None,
        lessons: list[Any] | None = None,
        gaps: list[Any] | None = None,
        authority: str = C.AUTHORITY_EXPERIENTIAL,
        confidence: float = C.DEFAULT_CONFIDENCE,
        evidence: list[Any] | None = None,
        source: str | None = None,
        provenance_details: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        prefer_sync: bool = True,
        _enqueue_on_failure: bool = True,
    ) -> dict[str, Any]:
        """``PUT /recipe-snapshot/recipes/{canonical_id}`` — upsert.

        Returns a dict carrying:

        * ``status`` — ``"ok"`` on successful PUT, ``"queued"`` on
          failure that fell through to NDJSON, ``"skip_disabled"``
          when ``enabled=False``.
        * ``canonical_id`` — echoed verbatim.
        * ``version`` — server-assigned, present iff ``status="ok"``.
        * ``created`` — ``True`` on the first PUT of this id.

        On HTTP failure with ``prefer_sync=True`` the row is enqueued
        to ``runtime/recipe_snapshot/.pending.ndjson`` for the
        background flusher (Phase 3). ``_enqueue_on_failure=False``
        suppresses the enqueue and propagates the exception — used
        by ``drain_pending`` to avoid duplicating rows on permanent
        failures.

        Side-channel guarantee: the foreground client profile bounds
        a single failed write to ~2.5 s (2 s timeout + one 200 ms
        backoff) before NDJSON enqueue, regardless of how unhealthy
        kb-service is. The Coordinator main loop never blocks longer
        than this on a KEEP / REVERT fact write.
        """
        if not self.enabled:
            return {"status": "skip_disabled", C.F_CANONICAL_ID: canonical_id}
        if not canonical_id:
            raise ValueError("put_recipe requires a non-empty canonical_id")

        idem = idempotency_key or f"put_recipe:{canonical_id}"
        ev_refs = self._normalise_evidence_refs(evidence)

        wire_body: dict[str, Any] = {
            C.F_AUTHORITY:  authority,
            C.F_CONFIDENCE: float(confidence),
            C.F_PROVENANCE: self._provenance(
                source=source,
                details=provenance_details,
                generator_suffix=canonical_id,
            ),
        }
        # Only emit caller-defined dicts / arrays when the caller
        # actually supplied something. Sending ``{}`` / ``[]`` is
        # harmless on the v2 server (defaults are identical) but
        # keeping the request body terse makes the audit log easier
        # to grep on.
        if labels:
            wire_body[C.F_LABELS] = dict(labels)
        if body:
            wire_body[C.F_BODY] = dict(body)
        if metrics:
            wire_body[C.F_METRICS] = dict(metrics)
        if findings:
            wire_body[C.F_FINDINGS] = list(findings)
        if failures:
            wire_body[C.F_FAILURES] = list(failures)
        if pitfalls:
            wire_body[C.F_PITFALLS] = list(pitfalls)
        if lessons:
            wire_body[C.F_LESSONS] = list(lessons)
        if gaps:
            wire_body[C.F_GAPS] = list(gaps)
        if ev_refs:
            wire_body[C.F_EVIDENCE_REFS] = ev_refs

        # NDJSON enqueue must replay the full PUT body verbatim
        # (including the resolved provenance) so the flusher can
        # re-issue an identical request without consulting any
        # SharedState the Coordinator may have moved on from.
        ndjson_payload = {
            C.F_CANONICAL_ID: canonical_id,
            "wire_body":      dict(wire_body),
        }

        path = C.format_recipe_path(C.PATH_RECIPE_TPL, canonical_id)
        if prefer_sync:
            try:
                resp = self._request("PUT", path, body=wire_body)
                if not isinstance(resp, dict):
                    resp = {}
                version = resp.get(C.F_VERSION)
                created = bool(resp.get(C.F_CREATED, False))
                self._audit_record(
                    op="put_recipe", status="ok",
                    canonical_id=canonical_id,
                    version=int(version) if isinstance(version, int) else None,
                    created=created,
                )
                return {
                    "status":         "ok",
                    C.F_CANONICAL_ID: canonical_id,
                    C.F_VERSION:      version,
                    C.F_CREATED:      created,
                }
            except RecipeSnapshotError as exc:
                if not _enqueue_on_failure:
                    raise
                log.info(
                    "put_recipe sync failed (%s); enqueueing NDJSON for %s",
                    exc, canonical_id,
                )

        # Sync write disabled OR sync write failed → enqueue.
        self._audit_record(
            op="put_recipe", status="queued",
            canonical_id=canonical_id, idempotency_key=idem,
        )
        self._enqueue(
            op=C.OP_PUT_RECIPE,
            payload=ndjson_payload,
            idempotency_key=idem,
        )
        return {
            "status":         "queued",
            C.F_CANONICAL_ID: canonical_id,
        }

    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """``GET /recipe-snapshot/recipes/{canonical_id}``.

        Returns the parsed ``Recipe`` dict, or ``None`` when:

        * the client is disabled (``--degraded-kb``), OR
        * kb-service returned 404 (canonical_id absent, or the
          requested ``?version=N`` is not in history).

        Other failures (transport / 422 / 4xx other than 404) raise
        :class:`RecipeSnapshotError`. Callers in warm-start / read
        paths MUST catch this and degrade gracefully (skip warm-start,
        keep the in-memory state) — write paths must NOT silently
        overwrite list fields based on an empty read.
        """
        if not self.enabled:
            return None
        if not canonical_id:
            raise ValueError("get_recipe requires a non-empty canonical_id")
        path = C.format_recipe_path(C.PATH_RECIPE_TPL, canonical_id)
        params: dict[str, Any] = {}
        if version is not None:
            params[C.F_VERSION] = int(version)
        resp = self._request(
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

        Returns the ``history`` array verbatim (each entry is
        ``{version, archived_at, replaced_by, snapshot}``); the
        server replies with an empty array for unknown ids (per
        spec, no 404 here) so this method never raises on absence.
        Disabled client returns ``[]``.

        ``limit`` is clamped server-side to ``[1, 1000]``; we
        forward whatever the caller passed and trust the server to
        validate.
        """
        if not self.enabled:
            return []
        if not canonical_id:
            raise ValueError("get_history requires a non-empty canonical_id")
        path = C.format_recipe_path(C.PATH_RECIPE_HISTORY_TPL, canonical_id)
        resp = self._request(
            "GET", path, params={C.F_LIMIT: int(limit)},
        )
        if not isinstance(resp, dict):
            return []
        history = resp.get(C.F_HISTORY)
        return list(history) if isinstance(history, list) else []

    # ------------------------------------------------------------------
    # NDJSON replay (CLOSE-time drain)
    # ------------------------------------------------------------------
    def drain_pending(self) -> dict[str, Any]:
        """Replay queued PUTs synchronously.

        Used at session close (Coordinator T4 hook) to give the
        pending queue one more shot before the session moves on.
        Each row is re-issued via ``put_recipe(...,
        _enqueue_on_failure=False)`` so a permanent rejection
        bubbles up to the dead-letter path instead of being
        re-enqueued forever.

        Returns a summary dict:
        ``{pending: int, flushed: int, dead_lettered: int,
        remaining: int}``. The flusher daemon (Phase 3) uses the
        same primitive on a 5 s / 50-line cadence.

        Phase 1 ships a minimal drain that processes the queue once
        in-place; the dead-letter file plumbing lives in
        :mod:`.session_paths` already
        (:func:`recipe_snapshot_dead_letter_ndjson`) but the
        rotation logic is finalised in Phase 3 alongside the
        full flusher daemon.
        """
        from .session_paths import (
            recipe_snapshot_dead_letter_ndjson,
            recipe_snapshot_flushed_ndjson,
        )

        if not self.enabled:
            return {"pending": 0, "flushed": 0, "dead_lettered": 0, "remaining": 0}

        pending = self.pending_path
        if not pending.exists():
            return {"pending": 0, "flushed": 0, "dead_lettered": 0, "remaining": 0}

        try:
            raw = pending.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.warning("drain_pending: cannot read %s: %s", pending, exc)
            return {"pending": 0, "flushed": 0, "dead_lettered": 0, "remaining": 0}

        rows: list[dict[str, Any]] = []
        for line in raw:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except (ValueError, json.JSONDecodeError):
                log.warning(
                    "drain_pending: dropping malformed row in %s", pending,
                )

        flushed_rows: list[dict[str, Any]] = []
        dead_rows: list[dict[str, Any]] = []
        retained: list[dict[str, Any]] = []
        for row in rows:
            op = row.get("op")
            payload = row.get("payload") or {}
            idem = str(row.get("idempotency_key") or "")
            attempts = int(row.get("attempts") or 0)
            if op != C.OP_PUT_RECIPE:
                # Unknown op (future-proofing for Phase 3 attempts
                # append) — keep in queue for the flusher to handle.
                retained.append(row)
                continue
            cid = payload.get(C.F_CANONICAL_ID)
            wire_body = payload.get("wire_body") or {}
            if not cid or not isinstance(wire_body, Mapping):
                # Malformed payload — drop straight to dead-letter so
                # it doesn't loop forever.
                row["last_error"] = "malformed_payload"
                dead_rows.append(row)
                continue
            path = C.format_recipe_path(C.PATH_RECIPE_TPL, str(cid))
            try:
                self._request("PUT", path, body=wire_body)
                flushed_rows.append(row)
            except RecipeSnapshotError as exc:
                # Transient errors keep the row in queue (attempts+1)
                # up to ``MAX_FLUSH_ATTEMPTS``; permanent (business /
                # validation) errors short-circuit to dead-letter on
                # the first failure since retrying will not help.
                attempts += 1
                row["attempts"] = attempts
                row["last_error"] = f"{exc.category}:{exc.code}:{exc}"
                if (
                    exc.category in ("business", "validation")
                    or attempts >= C.MAX_FLUSH_ATTEMPTS
                ):
                    dead_rows.append(row)
                else:
                    retained.append(row)
                self._audit_record(
                    op="drain", status="retry" if row in retained else "dead",
                    canonical_id=cid, idempotency_key=idem,
                    error_message=str(exc)[:512],
                )

        # Persist the three buckets. ``flushed`` / ``dead`` get
        # appended (consumers can grep cross-session); ``pending``
        # is rewritten with the retained subset.
        if flushed_rows:
            self._append_jsonl(recipe_snapshot_flushed_ndjson(self.session_dir),
                                flushed_rows)
        if dead_rows:
            self._append_jsonl(
                recipe_snapshot_dead_letter_ndjson(self.session_dir),
                dead_rows,
            )
        try:
            if retained:
                pending.write_text(
                    "".join(
                        json.dumps(r, sort_keys=True) + "\n"
                        for r in retained
                    ),
                    encoding="utf-8",
                )
            else:
                pending.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("drain_pending: cannot rewrite %s: %s", pending, exc)

        return {
            "pending":       len(rows),
            "flushed":       len(flushed_rows),
            "dead_lettered": len(dead_rows),
            "remaining":     len(retained),
        }

    @staticmethod
    def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        """Atomic-ish append of JSONL rows. Best-effort fsync."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass


__all__ = [
    "RecipeSnapshotClient",
    "RecipeSnapshotError",
    "parse_error_envelope",
]
