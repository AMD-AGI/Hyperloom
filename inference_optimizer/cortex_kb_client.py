"""Cortex KB client — KnowledgePlane facade write surface (HTTP transport).

This module is the **single point of contact** between
``inference_optimizer`` and the Cortex KB service. Two guarantees:

1. **All writes are channeled through the Coordinator**, never the
   reactor LLMs. PolicyGate enforces this; the facade itself doesn't
   check ACLs but every entrypoint takes a logical operation name so
   audit logs ascribe writes to ``inference_optimizer.coordinator``.
2. **Failure modes are well-defined** — synchronous HTTP failures fall
   through to a per-session NDJSON queue (``runtime/cortex/``);
   the queue is later drained by ``cortex_kb_flusher`` or by the
   ``drain_pending()`` helper at T4.

Transport: ``httpx.Client`` against ``CORTEX_KB_URL`` with bounded
concurrency, exponential retry on 5xx/timeout, and dual-form error
envelope parsing (``business`` / ``validation`` / ``unknown``).

Wire schema: ``cortex-kb-http-branch-b-2026-05-20.md`` (locked at
``primus-cortex-internal`` ``f48a785`` = image ``kbsg-edeb3d1``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import httpx

from . import cortex_kb_constants as C
from .session_paths import (
    cortex_audit_jsonl,
    cortex_dir,
    cortex_pending_ndjson,
    cortex_sid_file,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class CortexKBError(RuntimeError):
    """Raised for unrecoverable interactions with the Cortex KB.

    The synchronous T0/T4 hooks treat this as fail-fast (PRELUDE
    rejection / ``stop_reason=cortex_drain_failed``). Async T2/T3 hooks
    catch it and downgrade to an NDJSON enqueue.

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
# Canonical id derivation (aligned with KB registered kinds — §5)
# ---------------------------------------------------------------------------
def _slug(value: str, default: str) -> str:
    """Lowercase + collapse to ``[A-Za-z0-9_-]`` snake-ish slug."""
    cleaned = (value or "").strip().replace("/", "_").replace(" ", "_")
    return cleaned or default


def recipe_canonical_id(model_name: str, hardware: str) -> str:
    """``recipe:{slug(model)}:{slug(hardware)}`` — KB-registered ``recipe`` kind.

    Replaces the legacy ``workload.<slug>.<gpu>`` name; aligns with
    ``shared/kinds/recipe.py`` so kb-explorer + warm-start can index it.
    """
    return f"recipe:{_slug(model_name, 'unknown_model')}:{_slug(hardware, 'unknown_hw').lower()}"


def experiment_canonical_id(cortex_session_id: str, iter_index: int) -> str:
    """``exp:{session_id}:{iter:04d}`` — KB-registered ``experiment`` kind.

    Replaces the legacy ``opt.session-{sid}.proposal-{msg_id}`` name.
    ``iter_index`` is the monotonic ``session_iter_index`` Coordinator
    maintains on :class:`SharedState`; per-variant ids append
    ``.variant-{name}`` (KB pass-through; the parent ``exp:`` point is
    the registered anchor).
    """
    sid = (cortex_session_id or "0").strip() or "0"
    return f"exp:{sid}:{int(iter_index):04d}"


def attempt_canonical_id(cortex_session_id: str, task_id: str) -> str:
    """``attempt.session-{sid}.task-{task_id}`` — unregistered KB kind,
    pass-through. Kept for cross-session reachability; KB does no schema
    validation since ``attempt_node`` is not in the registered set.
    """
    sid = (cortex_session_id or "unknown").strip() or "unknown"
    tid = (task_id or "unknown").strip() or "unknown"
    return f"attempt.session-{sid}.task-{tid}"


# ---------------------------------------------------------------------------
# NDJSON envelope
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _ndjson_envelope(
    *,
    op: str,
    payload: Mapping[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    """Build one NDJSON row (shape: ``{op, payload, created_at,
    idempotency_key, attempts}``). ``attempts`` counts NDJSON
    flusher retries so robustness can alert on stuck rows.
    """
    return {
        "op":              op,
        "payload":         dict(payload),
        "created_at":      _now_iso(),
        "idempotency_key": idempotency_key,
        "attempts":        0,
    }


# ---------------------------------------------------------------------------
# Error envelope parser (§4)
# ---------------------------------------------------------------------------
def parse_kb_error(
    resp: httpx.Response,
) -> tuple[str, str, str, dict[str, Any]]:
    """Parse Cortex error envelope into ``(category, code, message, details)``.

    Three categories (§4):
    * ``business``  — ``{detail.error.{code, message, details}}``; stable.
    * ``validation`` — ``{detail: [{loc, msg, ...}]}``; only ``loc + msg``
      consumed (``type`` is pydantic-version-volatile).
    * ``unknown`` — anything else.
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

    Concurrency cap is enforced via a semaphore aligned with the
    backend's ``asyncpg pool=8`` (§3); 5xx/timeout/connect errors
    retry up to :data:`cortex_kb_constants.DEFAULT_RETRY_ATTEMPTS`
    with exponential backoff.
    """

    base_url: str
    timeout_sec: float
    token: str | None = None
    max_connections: int = C.DEFAULT_MAX_CONCURRENCY

    _client: httpx.Client | None = field(default=None, init=False, repr=False)
    _semaphore: threading.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._semaphore = threading.Semaphore(max(1, int(self.max_connections)))

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            headers: dict[str, str] = {"User-Agent": "hyperloom-cortex-kb-client"}
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
            except Exception:  # noqa: BLE001 — best effort close
                pass
            self._client = None

    def post(
        self,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST ``path`` with JSON ``body``; retry transient errors.

        Returns the parsed JSON response (always a dict). Raises
        :class:`CortexKBError` on:
        * exhausted retries (transient 5xx / connect / timeout);
        * 4xx business error (parsed via :func:`parse_kb_error`);
        * 422 validation error (parsed via :func:`parse_kb_error`).
        """
        client = self._ensure_client()
        last_exc: Exception | None = None
        for attempt in range(C.DEFAULT_RETRY_ATTEMPTS):
            with self._semaphore:
                try:
                    response = client.post(path, json=dict(body or {}))
                except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
                    last_exc = exc
                    self._backoff(attempt)
                    continue
                if response.status_code >= 500:
                    last_exc = CortexKBError(
                        f"transport 5xx on {path}: {response.status_code}",
                        category="transport", status=response.status_code,
                    )
                    self._backoff(attempt)
                    continue
                if response.status_code >= 400:
                    category, code, message, details = parse_kb_error(response)
                    raise CortexKBError(
                        f"{path} → {response.status_code}: {message}",
                        category=category, code=code,
                        status=response.status_code, details=details,
                    )
                if response.status_code == 204 or not response.content:
                    return {}
                try:
                    parsed = response.json()
                except (ValueError, json.JSONDecodeError) as exc:
                    raise CortexKBError(
                        f"{path}: response not JSON ({exc})",
                        category="unknown", status=response.status_code,
                    ) from exc
                return parsed if isinstance(parsed, dict) else {"_value": parsed}
        # Retry budget exhausted.
        raise CortexKBError(
            f"transport_exhausted after {C.DEFAULT_RETRY_ATTEMPTS} attempts: "
            f"{path}: {last_exc}",
            category="transport",
        )

    @staticmethod
    def _backoff(attempt: int) -> None:
        # 200ms × {1, 1.4, 4}
        multipliers = (1.0, 1.4, 4.0)
        idx = min(attempt, len(multipliers) - 1)
        time.sleep(C.DEFAULT_RETRY_BASE_MS * multipliers[idx] / 1000.0)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
@dataclass
class CortexKBClient:
    """Single-process Cortex KB client used by the Coordinator + cli T0 hook.

    Construction is cheap — no network I/O happens until the first
    wire call. NDJSON enqueue is O(append) regardless.

    Args:
        session_dir: hyperloom session root (writable). Used for the
            audit log + NDJSON queue.
        kb_url: ``CORTEX_KB_URL`` override; ``None`` reads env or
            :data:`cortex_kb_constants.DEFAULT_KB_URL`.
        timeout_sec: per-HTTP-call timeout (env
            ``CORTEX_KB_HTTP_TIMEOUT_SEC``).
        enabled: when ``False``, every entrypoint becomes a no-op
            (used by ``--degraded-kb``). Audit log still records the
            skip so breakdown collection can flag the bypass.
        max_connections: client-side cap; aligns with kb-service
            asyncpg pool=8.
        token: ``Authorization: Bearer`` value (env ``KB_SERVICE_TOKEN``);
            ``None`` omits the header (H1 anonymous).
        smoke: tag ``attrs.kbsg_smoke=True`` + smoke generator
            provenance on every propose call (env ``CORTEX_KB_SMOKE``).
        initiator: ``provenance.generator`` + ``initiator`` value.
    """

    session_dir: Path
    kb_url: str | None = None
    timeout_sec: float = C.DEFAULT_HTTP_TIMEOUT_SEC
    enabled: bool = True
    max_connections: int = C.DEFAULT_MAX_CONCURRENCY
    token: str | None = None
    smoke: bool = False
    initiator: str = C.DEFAULT_GENERATOR

    _transport: _HttpTransport | None = field(default=None, init=False, repr=False)
    _point_id_cache: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.session_dir = Path(self.session_dir)
        cortex_dir(self.session_dir).mkdir(parents=True, exist_ok=True)
        if not self.kb_url:
            self.kb_url = os.environ.get("CORTEX_KB_URL") or C.DEFAULT_KB_URL
        env_timeout = os.environ.get("CORTEX_KB_HTTP_TIMEOUT_SEC")
        if env_timeout:
            try:
                self.timeout_sec = float(env_timeout)
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
            self.smoke = (os.environ.get("CORTEX_KB_SMOKE") or "").strip().lower() in (
                "1", "true", "yes", "on",
            )

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    @property
    def pending_path(self) -> Path:
        return cortex_pending_ndjson(self.session_dir)

    @property
    def audit_path(self) -> Path:
        return cortex_audit_jsonl(self.session_dir)

    @property
    def sid_path(self) -> Path:
        return cortex_sid_file(self.session_dir)

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
            )
        return self._transport

    def _post(self, path: str, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        transport = self._ensure_transport()
        started = time.monotonic()
        try:
            result = transport.post(path, body)
        except CortexKBError as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            self._audit_record(
                op="http", status=exc.category or "error",
                path=path, elapsed_ms=elapsed_ms,
                error_code=exc.code, error_status=exc.status,
                error_message=str(exc)[:512],
            )
            raise
        elapsed_ms = int((time.monotonic() - started) * 1000)
        self._audit_record(op="http", status="ok", path=path, elapsed_ms=elapsed_ms)
        return result

    def _audit_record(self, **fields: Any) -> None:
        """Append one structured line to ``.kb_audit.jsonl``.

        Best-effort: a write failure is logged but never raised — the
        audit log is a forensic aid, not a correctness invariant.
        """
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            row = {"ts": _now_iso(), **fields}
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError as exc:
            log.warning("cortex audit append failed (%s): %s", self.audit_path, exc)

    # ------------------------------------------------------------------
    # NDJSON enqueue (fallback path for async ops)
    # ------------------------------------------------------------------
    def _enqueue(
        self, *, op: str, payload: Mapping[str, Any], idempotency_key: str,
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
        extras: dict[str, Any] = {}
        if op in ("verify", "ingest_attempt"):
            outcome = str(payload.get("outcome") or "")
            if outcome:
                extras["payload_outcome"] = outcome
            if op == "verify":
                edge = str(payload.get("edge") or "")
                if edge:
                    extras["payload_edge"] = edge
                promoted = str(payload.get("promoted_authority") or "")
                if promoted:
                    extras["payload_promote"] = promoted
        self._audit_record(
            op="enqueue", status="ok",
            envelope_op=op, idempotency_key=idempotency_key,
            **extras,
        )

    # ------------------------------------------------------------------
    # Smoke + provenance helpers
    # ------------------------------------------------------------------
    def _provenance(self, generator_suffix: str = "") -> dict[str, Any]:
        generator = (
            f"{C.SMOKE_GENERATOR}@{generator_suffix}"
            if self.smoke and generator_suffix
            else C.SMOKE_GENERATOR if self.smoke
            else self.initiator
        )
        return {
            C.F_PV_SOURCE:       C.SOURCE_AGENT_OBSERVATION,
            C.F_PV_GENERATOR:    generator,
            C.F_PV_GENERATED_AT: _now_iso(),
        }

    def _smoke_attrs(self, attrs: Mapping[str, Any] | None) -> dict[str, Any]:
        out = dict(attrs or {})
        if self.smoke:
            out["kbsg_smoke"] = True
        return out

    @staticmethod
    def _evidence_refs(evidence: list[str] | None) -> list[dict[str, Any]]:
        """Convert legacy string evidence list into ``EvidenceRef`` dicts.

        Strings of the form ``"<kind>:<ref>"`` (``log:``, ``commit:``,
        ``url:``, ``profile_file:``, ``point_id:``, ``edge_id:``) are
        split; anything else falls back to ``kind="log"``.
        """
        out: list[dict[str, Any]] = []
        known = {
            C.EV_KIND_URL, C.EV_KIND_COMMIT, C.EV_KIND_PROFILE_FILE,
            C.EV_KIND_LOG, C.EV_KIND_POINT_ID, C.EV_KIND_EDGE_ID,
        }
        for raw in (evidence or []):
            text = str(raw)
            kind = C.EV_KIND_LOG
            ref = text
            if ":" in text:
                prefix, _, rest = text.partition(":")
                if prefix in known:
                    kind = prefix
                    ref = rest
            out.append({C.F_EV_KIND: kind, C.F_EV_REF: ref})
        return out

    def _resolve_point_id(self, canonical_id: str) -> int:
        """Look up the int ``point_id`` for ``canonical_id``.

        Hits the in-memory cache first; on miss issues
        ``POST /v1/points/query {canonical_id}`` and caches the result.
        Raises :class:`CortexKBError` (``category="business"``) when KB
        has no such point yet — caller decides retry / propose-first.
        """
        cached = self._point_id_cache.get(canonical_id)
        if cached is not None:
            return cached
        body = {C.F_CANONICAL_ID: canonical_id, C.F_LIMIT: 1, C.F_NEIGHBOR_PREVIEW: False}
        resp = self._post(C.PATH_QUERY_POINT, body)
        points = resp.get(C.F_POINTS) or []
        if not points:
            raise CortexKBError(
                f"canonical_id {canonical_id!r} not found in KB",
                category="business", code="NOT_FOUND",
            )
        pid = int(points[0].get("id") or 0)
        if pid <= 0:
            raise CortexKBError(
                f"canonical_id {canonical_id!r} resolved to invalid id",
                category="business", code="INVALID_ID",
            )
        self._point_id_cache[canonical_id] = pid
        return pid

    # ==================================================================
    # Public API — read side (T0)
    # ==================================================================
    def session_begin(
        self,
        *,
        workload: str,
        hw: str,
        image_digest: str = "",
        stack_fingerprint: Mapping[str, str] | None = None,
        extra_attrs: Mapping[str, Any] | None = None,
        goal: str = C.GOAL_FIND_RECOMMENDATION,
        thinking_style: str | None = None,
        initiator: str | None = None,
    ) -> str:
        """T0 — ``POST /v1/sessions/begin``.

        Synchronous; failures bubble as :class:`CortexKBError` so the cli
        layer can fail-fast unless ``--degraded-kb`` was passed. Caller
        is responsible for writing the returned sid into SharedState +
        ``.kb_sid``.
        """
        if not self.enabled:
            self._audit_record(op="session_begin", status="skip_disabled")
            return ""
        attrs: dict[str, Any] = {
            "workload":     workload,
            "hw":           hw,
            "image_digest": image_digest or "unknown",
            "stack_fingerprint": dict(stack_fingerprint or {}),
        }
        if extra_attrs:
            attrs.update(extra_attrs)
        body: dict[str, Any] = {
            C.F_GOAL:      goal,
            C.F_INITIATOR: initiator or self.initiator,
            C.F_ATTRS:     self._smoke_attrs(attrs),
        }
        if thinking_style:
            body[C.F_THINKING_STYLE] = thinking_style
        resp = self._post(C.PATH_BEGIN, body)
        sid_val = resp.get(C.F_SESSION_ID)
        if sid_val is None or sid_val == "":
            raise CortexKBError(
                f"session begin returned no session_id; resp={resp!r}",
                category="unknown",
            )
        sid = str(sid_val)
        try:
            self.sid_path.write_text(sid + "\n", encoding="utf-8")
        except OSError as exc:
            log.warning("failed to persist .kb_sid: %s", exc)
        self._audit_record(
            op="session_begin", status="ok",
            session_id=sid, workload=workload, hw=hw,
        )
        return sid

    def find_recipe(self, *, workload: str, hw: str) -> str:
        """T0 — ``POST /v1/points/query`` filtering ``kind=recipe`` for
        the (workload, hw) anchor.

        Failures are non-fatal in M1 (warm_start is consumed by M5);
        callers should swallow :class:`CortexKBError`.
        """
        if not self.enabled:
            return ""
        body = {
            C.F_CANONICAL_ID:     recipe_canonical_id(workload, hw),
            C.F_KIND:             C.KIND_RECIPE,
            C.F_NEIGHBOR_PREVIEW: True,
            C.F_LIMIT:            1,
        }
        try:
            resp = self._post(C.PATH_QUERY_POINT, body)
        except CortexKBError:
            return ""
        return json.dumps(resp, sort_keys=True)

    def traps(self, *, symptom: str) -> str:
        """T0 — query ``kind=pitfall`` filtered by symptom.

        Returns JSON-encoded response body. Failures non-fatal.
        """
        if not self.enabled:
            return ""
        body = {
            C.F_KIND:         C.KIND_PITFALL,
            C.F_ATTRS_FILTER: {"symptom": symptom} if symptom else {},
            C.F_LIMIT:        50,
        }
        try:
            resp = self._post(C.PATH_QUERY_POINT, body)
        except CortexKBError:
            return ""
        return json.dumps(resp, sort_keys=True)

    # ==================================================================
    # Public API — write side (T2 / T3 / T4)
    # ==================================================================
    def propose_point(
        self,
        *,
        canonical_id: str,
        kind: str,
        attrs: Mapping[str, Any] | None = None,
        authority: str = C.AUTHORITY_HYPOTHESIZED,
        evidence: list[str] | None = None,
        source: str = C.SOURCE_AGENT_OBSERVATION,
        idempotency_key: str | None = None,
        prefer_sync: bool = True,
        entity_type: str | None = None,
    ) -> dict[str, Any]:
        """T0 mint / T2 mint — ``POST /v1/points/propose``.

        Sync first (so the caller gets a ``point_id`` back); on HTTP
        failure, falls back to NDJSON enqueue (returns
        ``{"status": "queued"}``).

        H1: ``committeeEnabled=false`` so the response is always
        ``status="auto_accepted"`` and ``point_id == proposal_id``.
        """
        if not self.enabled:
            return {C.F_STATUS: "skip_disabled"}
        idem = idempotency_key or f"propose_point:{canonical_id}"
        ev_refs = self._evidence_refs(evidence)
        body: dict[str, Any] = {
            C.F_CANONICAL_ID:  canonical_id,
            C.F_KIND:          kind,
            C.F_AUTHORITY:     authority,
            C.F_ATTRS:         self._smoke_attrs(attrs),
            C.F_EVIDENCE_REFS: ev_refs or [
                {C.F_EV_KIND: C.EV_KIND_LOG, C.F_EV_REF: f"hyperloom:{canonical_id}"},
            ],
            C.F_PROVENANCE:    {**self._provenance(), C.F_PV_SOURCE: source},
        }
        if entity_type:
            body[C.F_ENTITY_TYPE] = entity_type
        payload_for_ndjson = {
            "canonical_id": canonical_id,
            "kind":         kind,
            "authority":    authority,
            "attrs":        dict(attrs or {}),
            "evidence":     list(evidence or []),
            "source":       source,
        }
        if prefer_sync:
            try:
                resp = self._post(C.PATH_PROPOSE_POINT, body)
                proposal_id = resp.get(C.F_PROPOSAL_ID)
                point_id = resp.get(C.F_POINT_ID, proposal_id)
                status = str(resp.get(C.F_STATUS) or C.STATUS_AUTO_ACCEPTED)
                if point_id is not None and canonical_id:
                    try:
                        self._point_id_cache[canonical_id] = int(point_id)
                    except (TypeError, ValueError):
                        pass
                self._audit_record(
                    op="propose_point", status=status,
                    canonical_id=canonical_id, kind=kind,
                    authority=authority, source=source,
                    point_id=str(point_id or ""),
                )
                return {
                    C.F_STATUS:      status,
                    C.F_POINT_ID:    str(point_id) if point_id is not None else "",
                    C.F_PROPOSAL_ID: str(proposal_id) if proposal_id is not None else "",
                }
            except CortexKBError as exc:
                log.info("propose_point sync failed (%s); enqueueing NDJSON", exc)
        self._audit_record(
            op="propose_point", status="queued",
            canonical_id=canonical_id, kind=kind,
            authority=authority, source=source,
        )
        self._enqueue(op="propose_point", payload=payload_for_ndjson, idempotency_key=idem)
        return {C.F_STATUS: "queued"}

    def hypothesize(
        self,
        *,
        sid: str,
        from_canonical: str,
        to_canonical: str,
        edge_type: str = C.EDGE_HYPOTHETICAL,
        reason: str = "",
        attrs: Mapping[str, Any] | None = None,
        evidence: list[str] | None = None,
        idempotency_key: str | None = None,
        prefer_sync: bool = True,
    ) -> dict[str, Any]:
        """T2 — ``POST /v1/sessions/{sid}/hypothesize``.

        Resolves ``from_canonical`` / ``to_canonical`` to int point ids
        first, then POSTs. ``tentative_edge_id`` in the response is a
        trace id (kb_traces row), **not** a real kb_edges row — the
        edge is materialised by verify(outcome=confirmed).

        Returns ``{"status": ..., "tentative_edge_id": "<int>"}`` on
        sync success; on failure enqueues NDJSON and returns
        ``{"status": "queued", "tentative_edge_id": ""}``.
        """
        if not self.enabled or not sid:
            return {C.F_STATUS: "skip_disabled", C.F_TENTATIVE_EDGE_ID: ""}
        idem = idempotency_key or f"hypothesize:{sid}:{from_canonical}->{to_canonical}"
        ndjson_payload = {
            "sid":      sid,
            "from":     from_canonical,
            "to":       to_canonical,
            "type":     edge_type,
            "reason":   reason,
            "attrs":    dict(attrs or {}),
            "evidence": list(evidence or []),
        }
        if prefer_sync:
            try:
                from_id = self._resolve_point_id(from_canonical)
                to_id = self._resolve_point_id(to_canonical)
                body: dict[str, Any] = {
                    C.F_FROM_POINT: from_id,
                    C.F_TO_POINT:   to_id,
                    C.F_EDGE_TYPE:  edge_type,
                    C.F_REASON:     reason or "",
                }
                if attrs:
                    body[C.F_ATTRS] = self._smoke_attrs(attrs)
                resp = self._post(
                    C.PATH_HYPOTHESIZE.format(session_id=sid), body,
                )
                edge_id = resp.get(C.F_TENTATIVE_EDGE_ID)
                return {
                    C.F_STATUS:            "ok",
                    C.F_TENTATIVE_EDGE_ID: str(edge_id) if edge_id is not None else "",
                }
            except CortexKBError as exc:
                log.info("hypothesize sync failed (%s); enqueueing NDJSON", exc)
        self._enqueue(op="hypothesize", payload=ndjson_payload, idempotency_key=idem)
        return {C.F_STATUS: "queued", C.F_TENTATIVE_EDGE_ID: ""}

    def ingest_attempt(
        self,
        *,
        sid: str,
        iter_id: int,
        outcome: str,
        metrics: Mapping[str, Any],
        plan_edge: str = "",
        evidence: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """T3 — record an attempt as a propose_point with
        ``kind=attempt_node`` (unregistered KB kind; pass-through).

        Async per KB_design §3.6: always enqueues NDJSON; the flusher
        (or T4 drain) replays as a single ``propose_point`` HTTP call.

        ``outcome ∈ {"PASS", "FAIL", "PARTIAL"}``.
        """
        if not self.enabled or not sid:
            return {C.F_STATUS: "skip_disabled"}
        idem = idempotency_key or f"ingest_attempt:{sid}:{iter_id}"
        payload = {
            "sid":       sid,
            "iter":      int(iter_id),
            "outcome":   outcome,
            "metrics":   dict(metrics or {}),
            "plan_edge": plan_edge,
            "evidence":  list(evidence or []),
        }
        self._enqueue(op="ingest_attempt", payload=payload, idempotency_key=idem)
        return {C.F_STATUS: "queued"}

    def verify(
        self,
        *,
        sid: str,
        edge_id: str,
        outcome: str,
        evidence: list[str] | None = None,
        promote_authority: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """T3 — ``POST /v1/sessions/{sid}/verify``.

        Async per KB_design §3.6; always enqueues. ``outcome ∈
        {"confirmed", "refuted"}``. ``promote_authority="EXPERIENTIAL"``
        is the standard promotion for KEEP outcomes.

        Note: the ``edge_id`` here is the ``tentative_edge_id`` returned
        by :meth:`hypothesize` (trace id); the real ``promoted_edge_id``
        only exists after verify(confirmed) materialises the edge.
        """
        if not self.enabled or not sid or not edge_id:
            return {C.F_STATUS: "skip_disabled"}
        idem = idempotency_key or f"verify:{sid}:{edge_id}"
        payload = {
            "sid":                sid,
            "edge":               edge_id,
            "outcome":            outcome,
            "evidence":           list(evidence or []),
            "promoted_authority": promote_authority or "",
        }
        self._enqueue(op="verify", payload=payload, idempotency_key=idem)
        return {C.F_STATUS: "queued"}

    def session_commit(self, sid: str) -> dict[str, Any]:
        """T4 — ``POST /v1/sessions/{sid}/commit``.

        Synchronous. Caller must :meth:`drain_pending` first so all
        queued T2/T3 rows land before commit closes the session.
        Returns parsed commit summary; ``derived_summary_id`` is
        currently always ``None`` (KB H1 wires it that way; M5 connects).
        """
        if not self.enabled or not sid:
            return {C.F_STATUS: "skip_disabled"}
        resp = self._post(C.PATH_COMMIT.format(session_id=sid), {})
        promoted = [
            str(eid) for eid in (resp.get(C.F_PROMOTED_EDGES) or [])
        ]
        derived = resp.get(C.F_DERIVED_SUMMARY_ID)
        summary = {
            C.F_STATUS:             str(resp.get(C.F_STATUS) or C.STATUS_COMMITTED),
            C.F_PROMOTED_EDGES:     promoted,
            C.F_DERIVED_SUMMARY_ID: str(derived) if derived is not None else "",
            "raw":                  json.dumps(resp, sort_keys=True),
        }
        self._audit_record(
            op="session_commit", status=summary[C.F_STATUS],
            session_id=sid, promoted_count=len(promoted),
        )
        return summary

    def session_abort(self, sid: str, *, reason: str = "") -> dict[str, Any]:
        """``POST /v1/sessions/{sid}/abort`` — fail-fast escape hatch.

        Called from cli failure paths (T0 succeeded but PRELUDE crashed)
        so the KB-side session doesn't linger.

        ``reason`` is recorded in audit but the new HTTP schema's body
        is empty (``trace_preserved`` is recorded in response, always
        ``True`` per kb_traces being append-only).
        """
        if not self.enabled or not sid:
            return {C.F_STATUS: "skip_disabled"}
        try:
            resp = self._post(C.PATH_ABORT.format(session_id=sid), {})
            self._audit_record(
                op="session_abort", status="ok", session_id=sid, reason=reason,
            )
            return {
                C.F_STATUS:           str(resp.get(C.F_STATUS) or "aborted"),
                C.F_TRACE_PRESERVED:  bool(resp.get(C.F_TRACE_PRESERVED, True)),
            }
        except CortexKBError as exc:
            self._audit_record(
                op="session_abort", status="error",
                session_id=sid, error=str(exc)[:512],
            )
            return {C.F_STATUS: "abort_failed", "error": str(exc)}

    # ==================================================================
    # NDJSON drain (T4 + flusher)
    # ==================================================================
    def drain_pending(self, *, timeout_sec: float = 60.0) -> dict[str, Any]:
        """Process every row in ``.kb_pending.ndjson`` synchronously.

        Used by T4 before ``session commit`` so the commit closes a
        complete view of the session. The flusher daemon may run in
        parallel; we rely on append-only semantics + per-row exclusive
        consumption (rename → ``.kb_pending.processing.<pid>``) to
        avoid double-flush.

        Returns ``{"drained": N, "remaining": M, "dead_letter": K,
        "elapsed_ms": ...}``.
        """
        if not self.enabled:
            return {"drained": 0, "remaining": 0, "dead_letter": 0, "elapsed_ms": 0}
        started = time.monotonic()
        pending = self.pending_path
        if not pending.exists() or pending.stat().st_size == 0:
            return {"drained": 0, "remaining": 0, "dead_letter": 0, "elapsed_ms": 0}
        snapshot = pending.with_suffix(
            f".processing.{os.getpid()}.{uuid.uuid4().hex[:6]}",
        )
        try:
            os.rename(pending, snapshot)
        except FileNotFoundError:
            return {"drained": 0, "remaining": 0, "dead_letter": 0, "elapsed_ms": 0}
        drained = 0
        dead_letter = 0
        leftover_lines: list[str] = []
        deadline = started + max(0.0, timeout_sec)
        with snapshot.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if time.monotonic() > deadline:
                    leftover_lines.append(stripped)
                    continue
                try:
                    envelope = json.loads(stripped)
                except json.JSONDecodeError:
                    dead_letter += 1
                    self._audit_record(
                        op="drain", status="malformed_row",
                        line=stripped[:512],
                    )
                    continue
                outcome = self._flush_one(envelope)
                if outcome == "ok":
                    drained += 1
                elif outcome == "permanent":
                    dead_letter += 1
                else:
                    leftover_lines.append(stripped)
        if leftover_lines:
            new_pending = pending.with_suffix(f".restore.{os.getpid()}")
            with new_pending.open("w", encoding="utf-8") as f:
                for ln in leftover_lines:
                    f.write(ln + "\n")
            try:
                if pending.exists():
                    with pending.open("r", encoding="utf-8") as src, new_pending.open("a", encoding="utf-8") as dst:
                        for line in src:
                            dst.write(line)
                    pending.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("drain leftover concat failed: %s", exc)
            os.replace(new_pending, pending)
        try:
            snapshot.unlink(missing_ok=True)
        except OSError:
            pass
        elapsed_ms = int((time.monotonic() - started) * 1000)
        self._audit_record(
            op="drain", status="done",
            drained=drained, dead_letter=dead_letter,
            remaining=len(leftover_lines), elapsed_ms=elapsed_ms,
        )
        return {
            "drained":     drained,
            "remaining":   len(leftover_lines),
            "dead_letter": dead_letter,
            "elapsed_ms":  elapsed_ms,
        }

    def _flush_one(self, envelope: Mapping[str, Any]) -> str:
        """Replay one NDJSON envelope as a single HTTP call.

        Returns ``"ok"`` / ``"transient"`` / ``"permanent"``.
        Business / validation errors → ``permanent`` (dead-letter).
        Transport errors → ``transient`` (retry on next drain).
        """
        op = str(envelope.get("op", ""))
        payload = envelope.get("payload", {}) or {}
        try:
            if op == "propose_point":
                self.propose_point(
                    canonical_id=str(payload.get("canonical_id", "")),
                    kind=str(payload.get("kind", "")),
                    attrs=payload.get("attrs") or {},
                    authority=str(payload.get("authority") or C.AUTHORITY_HYPOTHESIZED),
                    evidence=list(payload.get("evidence") or []),
                    source=str(payload.get("source") or C.SOURCE_AGENT_OBSERVATION),
                    prefer_sync=True,
                )
            elif op == "hypothesize":
                self.hypothesize(
                    sid=str(payload.get("sid", "")),
                    from_canonical=str(payload.get("from", "")),
                    to_canonical=str(payload.get("to", "")),
                    edge_type=str(payload.get("type") or C.EDGE_HYPOTHETICAL),
                    reason=str(payload.get("reason", "")),
                    attrs=payload.get("attrs") or {},
                    evidence=list(payload.get("evidence") or []),
                    prefer_sync=True,
                )
            elif op == "ingest_attempt":
                self._ingest_attempt_sync(
                    sid=str(payload.get("sid", "")),
                    iter_id=int(payload.get("iter") or 0),
                    outcome=str(payload.get("outcome") or "PARTIAL"),
                    metrics=payload.get("metrics") or {},
                    plan_edge=str(payload.get("plan_edge") or ""),
                    evidence=list(payload.get("evidence") or []),
                )
            elif op == "verify":
                self._verify_sync(
                    sid=str(payload.get("sid", "")),
                    edge_id=str(payload.get("edge", "")),
                    outcome=str(payload.get("outcome") or C.OUTCOME_CONFIRMED),
                    evidence=list(payload.get("evidence") or []),
                    promote_authority=str(payload.get("promoted_authority") or "") or None,
                )
            else:
                return "permanent"
        except CortexKBError as exc:
            log.info("flush_one %s deferred: %s", op, exc)
            if exc.category in ("business", "validation"):
                return "permanent"
            return "transient"
        return "ok"

    def _ingest_attempt_sync(
        self, *, sid: str, iter_id: int, outcome: str,
        metrics: Mapping[str, Any], plan_edge: str,
        evidence: list[str],
    ) -> None:
        """Flush path for ``ingest_attempt`` — propose an
        ``attempt_node`` point carrying the metrics in ``attrs``.

        KB has no dedicated ingest-attempt endpoint; ``attempt_node`` is
        an unregistered kind so KB does pass-through validation.
        """
        attrs = {
            "outcome":   outcome,
            "iter":      int(iter_id),
            "metrics":   dict(metrics or {}),
            "plan_edge": plan_edge,
            "session":   sid,
        }
        self.propose_point(
            canonical_id=attempt_canonical_id(sid, str(iter_id)),
            kind=C.KIND_ATTEMPT,
            authority=C.AUTHORITY_EXPERIENTIAL,
            attrs=attrs,
            evidence=evidence,
            prefer_sync=True,
        )

    def _verify_sync(
        self, *, sid: str, edge_id: str, outcome: str,
        evidence: list[str], promote_authority: str | None,
    ) -> None:
        """Flush path for ``verify`` — ``POST /v1/sessions/{sid}/verify``.

        ``edge_id`` is the ``tentative_edge_id`` (int as string) that
        ``hypothesize`` returned; promoted_authority defaults to
        ``EXPERIENTIAL`` per schema.
        """
        try:
            ted = int(edge_id)
        except (TypeError, ValueError) as exc:
            raise CortexKBError(
                f"verify: tentative_edge_id {edge_id!r} not an int",
                category="validation",
            ) from exc
        body: dict[str, Any] = {
            C.F_TENTATIVE_EDGE_ID:  ted,
            C.F_OUTCOME:            outcome,
            C.F_EVIDENCE_REFS:      self._evidence_refs(evidence),
            C.F_PROMOTED_AUTHORITY: promote_authority or C.AUTHORITY_EXPERIENTIAL,
        }
        self._post(C.PATH_VERIFY.format(session_id=sid), body)


__all__ = [
    "CortexKBClient",
    "CortexKBError",
    "attempt_canonical_id",
    "experiment_canonical_id",
    "parse_kb_error",
    "recipe_canonical_id",
]
