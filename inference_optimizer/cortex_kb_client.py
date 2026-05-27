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

import hashlib
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
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class CortexKBError(RuntimeError):
    """Raised for unrecoverable interactions with the Cortex KB.

    Synchronous T0 (warm-start) treats this as fail-fast for the
    cli boot path (``sys.exit(2)``) and fail-soft for the Coordinator
    SDK fallback. Synchronous fact writes (``propose_lesson`` /
    ``propose_pitfall`` / ``update_recipe`` / ``propose_edge``) catch
    it internally and downgrade to an NDJSON enqueue so the
    Coordinator dispatcher never crashes on a transient KB outage.

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
    """Lowercase + collapse to ``[a-z0-9_-]`` snake-ish slug.

    PR-A10 (Arbor-into-Hyperloom): KB canonical_ids are case-sensitive
    and the corpus convention is **all-lowercase** (e.g.
    ``recipe:deepseek-r1-0528:mi300x``). Earlier behaviour preserved
    model-name case which silently missed every existing
    ``recipe:deepseek-r1-0528:mi300x`` lookup. We also basename any
    path-style input so a CLI ``--model /wekafs/models/DeepSeek-R1-0528``
    does not produce a junk slug like ``_wekafs_models_deepseek-r1-0528``.
    """
    raw = (value or "").strip()
    if not raw:
        return default
    # Path-style → basename (last path component, then forward-slash fallback).
    if "/" in raw:
        raw = raw.rstrip("/").rsplit("/", 1)[-1] or raw
    cleaned = raw.replace(" ", "_").lower()
    return cleaned or default


def recipe_canonical_id(model_name: str, hardware: str) -> str:
    """``recipe:{slug(model)}:{slug(hardware)}`` — KB-registered ``recipe`` kind.

    Replaces the legacy ``workload.<slug>.<gpu>`` name; aligns with
    ``shared/kinds/recipe.py`` so kb-explorer + warm-start can index it.

    Both slug components are lowercased (PR-A10) so the canonical_id
    matches the KB corpus convention regardless of how the operator
    typed the CLI ``--model`` arg.
    """
    return f"recipe:{_slug(model_name, 'unknown_model')}:{_slug(hardware, 'unknown_hw')}"


# ---------------------------------------------------------------------------
# Model-family taxonomy (PR-A10 — drives find_recipe_with_fallback T3)
#
# Map a model slug → a coarse family key. Used by the fallback ladder so a
# DeepSeek-R1 cold-start can reuse a DeepSeek-V3.1 recipe (same architecture
# family, both MoE+MLA). Heuristic-only; lower-cased substring match.
# Adding a new family is a one-line append.
# ---------------------------------------------------------------------------
_MODEL_FAMILY_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("deepseek",  ("deepseek",)),
    ("qwen3",     ("qwen3",)),
    ("qwen2",     ("qwen2",)),
    ("qwen",      ("qwen",)),
    ("llama3",    ("llama-3", "llama3", "meta-llama_llama-3", "meta-llama_meta-llama-3")),
    ("llama2",    ("llama-2", "llama2")),
    ("llama",     ("llama",)),
    ("mixtral",   ("mixtral",)),
    ("mistral",   ("mistral",)),
    ("minimax",   ("minimax",)),
    ("kimi",      ("kimi",)),
    ("glm",       ("glm",)),
    ("phi",       ("phi-", "phi3", "phi4")),
    ("gemma",     ("gemma",)),
    ("yi",        ("01-ai_yi", "yi-")),
)


def model_family(model_name: str) -> str:
    """Return a coarse model-family key for fallback lookups.

    Empty string when no prefix matches (caller can decide to skip the
    family-tier fallback). The match is **lowercase-substring**, not
    regex, so it survives small naming variations (``deepseek-r1-0528``,
    ``deepseek-v3.1``, ``deepseek-coder-v2`` all → ``"deepseek"``).
    """
    s = _slug(model_name, "")
    if not s:
        return ""
    for family, prefixes in _MODEL_FAMILY_PREFIXES:
        for prefix in prefixes:
            if s.startswith(prefix) or prefix in s:
                return family
    return ""


def _hash16(payload: str) -> str:
    """Stable SHA-256 prefix used by lesson / pitfall canonical ids.

    kg-usage-guide §7.4 specifies a 16-hex SHA-256 prefix derived from
    the kind's identifying attrs. Both ``lesson`` and ``pitfall`` use
    ``{statement|description}|sorted(cited_citation_ids)``; sharing one
    helper keeps the two ids byte-compatible with anything else KB
    consumers compute from the same recipe.
    """
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def lesson_canonical_id(
    statement: str,
    cited_citation_ids: list[str] | None = None,
) -> str:
    """``lesson:{hash16(statement|sorted(cited_citation_ids))}`` — kg-usage-guide §7.4.

    Idempotent across sessions: two reactor turns that emit the same
    statement (with the same citation set) merge into one ``lesson``
    point instead of duplicating rows. ``cited_citation_ids`` is
    sorted before hashing so call-site ordering doesn't matter.
    """
    citations = sorted(str(c) for c in (cited_citation_ids or []))
    payload = f"{statement or ''}|{','.join(citations)}"
    return f"lesson:{_hash16(payload)}"


def pitfall_canonical_id(
    description: str,
    cited_citation_ids: list[str] | None = None,
) -> str:
    """``pitfall:{hash16(description|sorted(cited_citation_ids))}`` — kg-usage-guide §7.4.

    Same hashing scheme as :func:`lesson_canonical_id`; the prefix
    differs so a duplicate description registered under both kinds
    cannot accidentally merge.
    """
    citations = sorted(str(c) for c in (cited_citation_ids or []))
    payload = f"{description or ''}|{','.join(citations)}"
    return f"pitfall:{_hash16(payload)}"


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
        self._audit_record(
            op="enqueue", status="ok",
            envelope_op=op, idempotency_key=idempotency_key,
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
    def read_recipe_exact(
        self, *, model: str, hardware: str,
    ) -> dict[str, Any]:
        """Read the current ``recipe:{model}:{hardware}`` point as a dict.

        Returns the parsed KB point (``{id, canonical_id, kind,
        attrs, authority, confidence, ...}``) or ``{}`` when the
        anchor doesn't exist yet / the KB is disabled / the query
        fails.

        Unlike :meth:`find_recipe_with_fallback` this NEVER falls
        back to same-family / same-class records — it's the dedicated
        read primitive for ``update_recipe`` read-modify-write
        callers (e.g. CLOSE-time recipe finalize) that need to see
        the CURRENT anchor state to merge ``sessions[]`` without
        losing historical entries.

        Failures are non-fatal: the caller treats ``{}`` as "no
        prior state" and proceeds with a fresh write.
        """
        if not self.enabled:
            return {}
        body = {
            C.F_CANONICAL_ID:     recipe_canonical_id(model, hardware),
            C.F_KIND:             C.KIND_RECIPE,
            C.F_NEIGHBOR_PREVIEW: False,
            C.F_LIMIT:            1,
        }
        try:
            resp = self._post(C.PATH_QUERY_POINT, body)
        except CortexKBError as exc:
            log.info("read_recipe_exact failed (%s); treating as no prior", exc)
            return {}
        points = resp.get(C.F_POINTS) or resp.get("points") or []
        if isinstance(points, list) and points and isinstance(points[0], dict):
            return points[0]
        return {}

    def find_recipe(self, *, workload: str, hw: str) -> str:
        """T0 — ``POST /v1/points/query`` filtering ``kind=recipe`` for
        the (workload, hw) anchor.

        Failures are non-fatal in M1 (warm_start is consumed by M5);
        callers should swallow :class:`CortexKBError`. Kept as the
        narrow exact-match T1 lookup; new callers should prefer
        :meth:`find_recipe_with_fallback` which adds the same-family /
        same-class / same-hw / cross-hw fallback tiers (PR-A10).
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

    def find_recipe_with_fallback(
        self,
        *,
        workload: str,
        hw: str,
        model_class: str | None = None,
        framework: str | None = None,
        precision: str | None = None,
        tp: int | None = None,
    ) -> tuple[dict[str, Any], str, float]:
        """PR-A10 (Arbor-into-Hyperloom) — graceful warm-start fallback.

        Returns ``(recipe_dict, tier, tier_confidence)`` so the caller
        can render "we matched a related recipe at confidence X" into
        the specialist prompt instead of silently giving them nothing.

        Tiers (from precise → coarse, confidence stamped accordingly):

        =====  =====  ============================================================
        Tier   Conf   Lookup
        =====  =====  ============================================================
        T1     0.85   ``canonical_id == recipe:<model_slug>:<hw_slug>`` + has
                      a real best_config (``best_config`` dict OR Arbor-shape
                      ``best_config_args`` / ``best_config_envs`` populated)
        T2     0.70   ``model_family`` + ``hardware`` + ``precision`` + ``tp``
                      (+ ``framework`` when supplied) all match — workload-shape
                      hit, dramatically more transferable than family alone
                      because best_config diverges sharply on precision / TP
        T3     0.55   ``model_family`` + ``hardware`` (+ ``framework``) match
                      — same architecture family, same GPU + serving stack
        T4     0.40   ``model_class`` + ``hardware`` (+ ``framework``) match
                      — same coarse taxonomy (moe_mla / dense / ...), same GPU
        T5     0.25   any ``kind=recipe`` with ``hardware == hw`` — only
                      hw-level ROCm defaults are reusable (framework-agnostic
                      on purpose; sweep params transcend serving stack)
        T6     0.20   any ``kind=recipe`` with ``attrs.model == workload``
                      across hardware — cross-GPU port with caveats
        miss   0.00   nothing found
        =====  =====  ============================================================

        ``framework`` filter is applied on T2/T3/T4 only — a sglang
        session must NEVER pick up a vLLM recipe (the best_config
        ``extra_sglang_args`` blob is framework-specific and would
        crash the server). When ``framework`` is ``None`` (caller
        didn't pin one) the filter degrades to "any framework".

        On a hit, the returned dict carries the full Cortex point:
        ``{id, canonical_id, kind, entity_type, attrs, authority,
        confidence, ...}``. ``tier_confidence`` is the *prior* the
        fallback ladder assigns; the caller is free to multiply it by
        ``point["confidence"]`` from the KB record itself.

        Failures (network / 5xx / disabled) are non-fatal: returns the
        ``("", "miss", 0.0)`` triple so cortex_t0 just renders an empty
        warm-start.
        """
        empty: tuple[dict[str, Any], str, float] = ({}, "miss", 0.0)
        if not self.enabled:
            return empty

        slug_model = _slug(workload, "unknown_model")
        slug_hw = _slug(hw, "unknown_hw")
        family = model_family(workload)

        def _has_real_config(p: dict[str, Any]) -> bool:
            """Distinguish a usable warm-start recipe from a smoke seed.

            Two known shapes count as "real":

            * Arbor-replay shape — ``attrs.best_config_args`` /
              ``best_config_envs`` populated (e.g. the MiniMax-M2.7
              recipe).
            * analogy_seed shape — ``attrs._provenance.details``
              carries ``decomposition_pct`` (per-knob gain prior) or
              ``runtime_shape`` (e.g. the DeepSeek-V3.1 transplant).

            Smoke / offline seed records (just
            ``{model, hardware, _provenance, _evidence_refs}``) miss
            both shapes and are skipped by the fallback ladder so we
            don't surface empty warm-start to specialists.
            """
            attrs = (p or {}).get("attrs") or {}
            # New hyperloom fact-write shape (post T2/T3 retirement):
            # update_recipe writes ``best_config`` as a nested dict.
            # Without this branch every hyperloom-written recipe would
            # be classified as a "seed-only" record and the entire
            # fallback ladder would silently return ``miss`` ── the
            # warm-start surface would be dead.
            bc = attrs.get("best_config")
            if isinstance(bc, dict) and (
                bc.get("extra_sglang_args")
                or bc.get("extra_envs")
                or bc.get("args")
                or bc.get("envs")
                or bc.get("name")
            ):
                return True
            # Legacy Arbor shape: flat ``best_config_args`` /
            # ``best_config_envs``. Kept so a recipe migrated from
            # Arbor / offline ingest still counts as "real".
            if attrs.get("best_config_args") or attrs.get("best_config_envs"):
                return True
            prov = attrs.get("_provenance") or {}
            details = prov.get("details") if isinstance(prov, dict) else None
            if isinstance(details, dict):
                if details.get("decomposition_pct") or details.get("runtime_shape"):
                    return True
            return False

        def _query(body: Mapping[str, Any]) -> list[dict[str, Any]]:
            try:
                resp = self._post(C.PATH_QUERY_POINT, dict(body))
            except CortexKBError as exc:
                log.info("find_recipe_with_fallback: query failed: %s", exc)
                return []
            points = resp.get(C.F_POINTS) or resp.get("points") or []
            return list(points) if isinstance(points, list) else []

        # ── T1: exact canonical_id ───────────────────────────────────
        t1 = _query({
            C.F_CANONICAL_ID:     recipe_canonical_id(workload, hw),
            C.F_KIND:             C.KIND_RECIPE,
            C.F_NEIGHBOR_PREVIEW: True,
            C.F_LIMIT:            1,
        })
        for p in t1:
            if _has_real_config(p):
                return p, "T1_exact", 0.85
        # T1 hit but empty attrs (smoke / seed record) → keep walking
        t1_seed = t1[0] if t1 else None

        # ── T2: same family + same hw + same workload-shape ──────────
        # Inserted between T1 (exact) and T3 (same-family) because a
        # precision / TP match is dramatically more transferable than
        # a same-family match alone — DeepSeek-R1 fp8 TP=8 and
        # DeepSeek-R1 bf16 TP=4 share the architecture but their
        # best_config will diverge sharply on attention backend +
        # kv-cache-dtype + max-num-seqs.
        if family and (precision or tp):
            shape_filter: dict[str, Any] = {"hardware": slug_hw}
            if framework:
                # Critical: never cross frameworks — sglang's
                # ``extra_sglang_args`` blob is incompatible with vLLM
                # CLI flags and would crash the server.
                shape_filter["framework"] = framework
            if precision:
                shape_filter["precision"] = precision
            if tp:
                shape_filter["tp"] = int(tp)
            cand = _query({
                C.F_KIND:             C.KIND_RECIPE,
                C.F_ATTRS_FILTER:     shape_filter,
                C.F_LIMIT:            20,
                C.F_NEIGHBOR_PREVIEW: False,
            })
            cand = [p for p in cand
                    if model_family(((p.get("attrs") or {}).get("model") or "")) == family
                    and _has_real_config(p)]
            cand.sort(key=lambda p: float(p.get("confidence") or 0.0), reverse=True)
            if cand:
                return cand[0], "T2_same_shape", 0.70

        # ── T3: same family, same hardware ───────────────────────────
        if family:
            t3_filter: dict[str, Any] = {"hardware": slug_hw}
            if framework:
                t3_filter["framework"] = framework
            cand = _query({
                C.F_KIND:             C.KIND_RECIPE,
                C.F_ATTRS_FILTER:     t3_filter,
                C.F_LIMIT:            20,
                C.F_NEIGHBOR_PREVIEW: False,
            })
            cand = [p for p in cand
                    if model_family(((p.get("attrs") or {}).get("model") or "")) == family
                    and _has_real_config(p)]
            cand.sort(key=lambda p: float(p.get("confidence") or 0.0), reverse=True)
            if cand:
                return cand[0], "T3_same_family", 0.55

        # ── T4: same model_class, same hardware ──────────────────────
        if model_class:
            t4_filter: dict[str, Any] = {
                "model_class": model_class,
                "hardware":    slug_hw,
            }
            if framework:
                t4_filter["framework"] = framework
            cand = _query({
                C.F_KIND:             C.KIND_RECIPE,
                C.F_ATTRS_FILTER:     t4_filter,
                C.F_LIMIT:            20,
                C.F_NEIGHBOR_PREVIEW: False,
            })
            cand = [p for p in cand if _has_real_config(p)]
            cand.sort(key=lambda p: float(p.get("confidence") or 0.0), reverse=True)
            if cand:
                return cand[0], "T4_same_class", 0.40

        # ── T5: any recipe on this hardware ──────────────────────────
        cand = _query({
            C.F_KIND:             C.KIND_RECIPE,
            C.F_ATTRS_FILTER:     {"hardware": slug_hw},
            C.F_LIMIT:            10,
            C.F_NEIGHBOR_PREVIEW: False,
        })
        cand = [p for p in cand if _has_real_config(p)]
        cand.sort(key=lambda p: float(p.get("confidence") or 0.0), reverse=True)
        if cand:
            return cand[0], "T5_same_hw_any", 0.25

        # ── T6: same model name across hardware ──────────────────────
        cand = _query({
            C.F_KIND:             C.KIND_RECIPE,
            C.F_ATTRS_FILTER:     {"model": workload},
            C.F_LIMIT:            5,
            C.F_NEIGHBOR_PREVIEW: False,
        })
        cand = [p for p in cand if _has_real_config(p)]
        cand.sort(key=lambda p: float(p.get("confidence") or 0.0), reverse=True)
        if cand:
            return cand[0], "T6_cross_hw", 0.20

        # ── miss: surface the T1 seed (if any) so the caller can at
        # least show the canonical_id was minted, with confidence 0
        # so prompt rendering tags it clearly.
        if t1_seed is not None:
            return t1_seed, "T1_seed_only", 0.0
        return empty

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
        _enqueue_on_failure: bool = True,
    ) -> dict[str, Any]:
        """``POST /v1/points/propose`` — generic point write.

        The fact-write surface methods (``propose_lesson`` /
        ``propose_pitfall`` / ``update_recipe``) all wrap this with
        kind-specific canonical_id derivation and attrs shape; direct
        callers are T0 anchor backfill + NDJSON replay.

        Sync first (so the caller gets a ``point_id`` back); on HTTP
        failure, falls back to NDJSON enqueue (returns
        ``{"status": "queued"}``). When ``_enqueue_on_failure=False``
        (used by ``_flush_one`` replay) the CortexKBError propagates
        to the caller instead of duplicating the row.

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
                # ``_flush_one`` replays NDJSON rows with
                # ``_enqueue_on_failure=False`` so this code path is
                # only reached for live writes — re-enqueueing a row
                # the replayer just dequeued would duplicate it
                # forever on permanent errors.
                if not _enqueue_on_failure:
                    raise
                log.info("propose_point sync failed (%s); enqueueing NDJSON", exc)
        self._audit_record(
            op="propose_point", status="queued",
            canonical_id=canonical_id, kind=kind,
            authority=authority, source=source,
        )
        self._enqueue(op="propose_point", payload=payload_for_ndjson, idempotency_key=idem)
        return {C.F_STATUS: "queued"}

    # ==================================================================
    # Fact-write surface — direct propose (kg-usage-guide §3.1 / §3.2)
    #
    # The methods below write **standalone fact points** (lesson /
    # pitfall) and edges connecting them to source citations. Hyperloom
    # only writes facts after local verification (a KEEP / REVERT
    # decision), so the writes are session-less — there is no KB-side
    # session to register or commit (the legacy hypothesize/verify
    # protocol was retired).
    # ==================================================================
    def propose_edge(
        self,
        *,
        from_canonical_id: str,
        to_canonical_id: str,
        edge_type: str,
        relation: str = "",
        authority: str = C.AUTHORITY_EXPERIENTIAL,
        attrs: Mapping[str, Any] | None = None,
        evidence: list[str] | None = None,
        source: str = C.SOURCE_AGENT_OBSERVATION,
        idempotency_key: str | None = None,
        prefer_sync: bool = True,
        _enqueue_on_failure: bool = True,
    ) -> dict[str, Any]:
        """``POST /v1/edges/propose`` — direct, session-less edge write.

        Use this when both endpoints already exist as KB points and you
        want to record a structural / causal / empirical / negation /
        evolutionary relation between them. This is the only edge-write
        primitive (the legacy hypothesize/verify round-trip was retired).

        ``relation`` is folded into ``attrs.relation`` so the propose-
        edge validator can pair it with ``edge_type`` (see kg-usage-
        guide §7.3 for the 10 allowed pairings; omitting it is allowed
        but downstream renderers lose a hint).

        Resolves canonical_ids to int point ids before posting. On HTTP
        failure falls back to NDJSON enqueue and returns
        ``{"status": "queued"}``; on success returns
        ``{"status": ..., "edge_id": "<int>"}``.
        """
        if not self.enabled:
            return {C.F_STATUS: "skip_disabled"}
        idem = (
            idempotency_key
            or f"propose_edge:{from_canonical_id}->{to_canonical_id}:{edge_type}"
        )
        merged_attrs = self._smoke_attrs(attrs)
        if relation:
            merged_attrs.setdefault("relation", relation)
        ev_refs = self._evidence_refs(evidence) or [
            {C.F_EV_KIND: C.EV_KIND_LOG,
             C.F_EV_REF:  f"hyperloom:{from_canonical_id}->{to_canonical_id}"},
        ]
        ndjson_payload = {
            "from_canonical_id": from_canonical_id,
            "to_canonical_id":   to_canonical_id,
            "edge_type":         edge_type,
            "relation":          relation,
            "authority":         authority,
            "attrs":             dict(attrs or {}),
            "evidence":          list(evidence or []),
            "source":            source,
        }
        if prefer_sync:
            try:
                from_id = self._resolve_point_id(from_canonical_id)
                to_id = self._resolve_point_id(to_canonical_id)
                body: dict[str, Any] = {
                    C.F_FROM_POINT:    from_id,
                    C.F_TO_POINT:      to_id,
                    C.F_EDGE_TYPE:     edge_type,
                    C.F_AUTHORITY:     authority,
                    C.F_ATTRS:         merged_attrs,
                    C.F_EVIDENCE_REFS: ev_refs,
                    C.F_PROVENANCE:    {
                        **self._provenance(),
                        C.F_PV_SOURCE: source,
                    },
                }
                resp = self._post(C.PATH_PROPOSE_EDGE, body)
                edge_id = resp.get("edge_id") or resp.get(C.F_PROMOTED_EDGE_ID)
                status = str(resp.get(C.F_STATUS) or C.STATUS_AUTO_ACCEPTED)
                self._audit_record(
                    op="propose_edge", status=status,
                    edge_type=edge_type, relation=relation,
                    from_canonical=from_canonical_id,
                    to_canonical=to_canonical_id,
                    edge_id=str(edge_id or ""),
                )
                return {
                    C.F_STATUS: status,
                    "edge_id":  str(edge_id) if edge_id is not None else "",
                }
            except CortexKBError as exc:
                if not _enqueue_on_failure:
                    raise
                log.info("propose_edge sync failed (%s); enqueueing NDJSON", exc)
        self._audit_record(
            op="propose_edge", status="queued",
            edge_type=edge_type, relation=relation,
            from_canonical=from_canonical_id,
            to_canonical=to_canonical_id,
        )
        self._enqueue(
            op="propose_edge", payload=ndjson_payload, idempotency_key=idem,
        )
        return {C.F_STATUS: "queued"}

    def propose_lesson(
        self,
        *,
        statement: str,
        measured_impact: str,
        applicable_models: list[str] | None = None,
        applicable_hardware: list[str] | None = None,
        cited_citation_ids: list[str] | None = None,
        evidence: list[str] | None = None,
        authority: str = C.AUTHORITY_EXPERIENTIAL,
        source_session_id: str = "",
        source_task_id: str = "",
        source_variant_name: str = "",
        extra_attrs: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        prefer_sync: bool = True,
    ) -> dict[str, Any]:
        """High-level wrapper writing a registered ``lesson`` fact point.

        Canonical id is derived from ``statement`` + sorted
        ``cited_citation_ids`` so two sessions that learn the same
        thing converge on a single row (KB merge is new-wins shallow,
        authority is only-up — exactly what we want).

        Defaults to ``EXPERIENTIAL`` because the caller is supposed to
        invoke this only after local verification (KEEP). Use
        ``AUTHORITATIVE`` when seeding from upstream docs / release
        notes.

        ``source_session_id`` / ``source_task_id`` /
        ``source_variant_name`` are stamped on ``attrs`` for
        traceability. Note: KB does shallow new-wins merge on
        ``attrs`` keys, so when two sessions emit the SAME lesson
        statement, only the most recent ``source_*`` triple survives
        on the KB row — the older one is overwritten. This is the
        intended trade-off vs putting source ids in
        ``cited_citation_ids`` (which would change the canonical_id
        hash and create one separate lesson row per session, defeating
        cross-session dedup). Full historical attribution lives in
        the per-session ``optimization_journal.json`` instead.
        """
        cid = lesson_canonical_id(statement, cited_citation_ids)
        attrs: dict[str, Any] = {
            "statement":           statement,
            "measured_impact":     measured_impact,
            "applicable_models":   list(applicable_models or []),
            "applicable_hardware": list(applicable_hardware or []),
            "cited_citation_ids":  sorted(str(c) for c in (cited_citation_ids or [])),
        }
        if source_session_id:
            attrs["source_session_id"] = source_session_id
        if source_task_id:
            attrs["source_task_id"] = source_task_id
        if source_variant_name:
            attrs["source_variant_name"] = source_variant_name
        if extra_attrs:
            attrs.update(extra_attrs)
        return self.propose_point(
            canonical_id=cid,
            kind=C.KIND_LESSON,
            authority=authority,
            attrs=attrs,
            evidence=evidence,
            idempotency_key=idempotency_key or f"propose_lesson:{cid}",
            prefer_sync=prefer_sync,
        )

    def propose_pitfall(
        self,
        *,
        description: str,
        severity: str,
        applicable_models: list[str] | None = None,
        applicable_hardware: list[str] | None = None,
        cited_citation_ids: list[str] | None = None,
        evidence: list[str] | None = None,
        authority: str = C.AUTHORITY_EXPERIENTIAL,
        source_session_id: str = "",
        source_task_id: str = "",
        source_variant_name: str = "",
        extra_attrs: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        prefer_sync: bool = True,
    ) -> dict[str, Any]:
        """High-level wrapper writing a registered ``pitfall`` fact point.

        ``severity`` ∈ {``crash`` | ``regress`` | ``noop``} per
        kg-usage-guide §7.4. Hyperloom's threshold (KB_design — plan
        chapter for fact writes): crash/oom/hang map to ``crash``;
        gain_pct ≤ -5% maps to ``regress``; anything weaker is NOT
        written (signal/noise control). Callers should filter before
        invoking this method.

        ``source_session_id`` / ``source_task_id`` /
        ``source_variant_name`` are stamped on ``attrs`` for
        traceability. Same caveat as :meth:`propose_lesson`: KB
        shallow new-wins merge means only the most recent
        ``source_*`` triple survives when multiple sessions write
        the same pitfall description; full historical attribution
        lives in the per-session ``optimization_journal.json``.
        """
        cid = pitfall_canonical_id(description, cited_citation_ids)
        attrs: dict[str, Any] = {
            "description":         description,
            "severity":            severity,
            "applicable_models":   list(applicable_models or []),
            "applicable_hardware": list(applicable_hardware or []),
            "cited_citation_ids":  sorted(str(c) for c in (cited_citation_ids or [])),
        }
        if source_session_id:
            attrs["source_session_id"] = source_session_id
        if source_task_id:
            attrs["source_task_id"] = source_task_id
        if source_variant_name:
            attrs["source_variant_name"] = source_variant_name
        if extra_attrs:
            attrs.update(extra_attrs)
        return self.propose_point(
            canonical_id=cid,
            kind=C.KIND_PITFALL,
            authority=authority,
            attrs=attrs,
            evidence=evidence,
            idempotency_key=idempotency_key or f"propose_pitfall:{cid}",
            prefer_sync=prefer_sync,
        )

    def update_recipe(
        self,
        *,
        model: str,
        hardware: str,
        best_config: Mapping[str, Any] | None = None,
        best_throughput: float | None = None,
        what_worked: list[Mapping[str, Any]] | None = None,
        what_failed: list[Mapping[str, Any]] | None = None,
        remaining_gaps: list[Mapping[str, Any]] | None = None,
        pitfalls: list[Mapping[str, Any]] | None = None,
        stack_fingerprint: Mapping[str, str] | None = None,
        last_profiled: str = "",
        sessions: list[Mapping[str, Any]] | None = None,
        extra_attrs: Mapping[str, Any] | None = None,
        authority: str = C.AUTHORITY_EXPERIENTIAL,
        evidence: list[str] | None = None,
        idempotency_key: str | None = None,
        prefer_sync: bool = True,
    ) -> dict[str, Any]:
        """Merge fact fields into the (model, hardware) ``recipe`` anchor.

        Wraps :meth:`propose_point` with ``kind="recipe"`` and the
        canonical id ``recipe:{slug(model)}:{slug(hardware)}`` — the
        same anchor T0 mints, so KB-merge semantics make this an
        in-place update of the existing point.

        Only non-``None`` arguments are written; unspecified keys are
        left untouched on the server side (attrs is shallow-merged
        new-wins). Recipe is the closest hyperloom analogue of Arbor's
        ``Recipe`` dataclass — see kg-usage-guide §7.4 for the full
        field shape.
        """
        cid = recipe_canonical_id(model, hardware)
        attrs: dict[str, Any] = {
            "model":    model,
            "hardware": hardware,
        }
        if best_config is not None:
            attrs["best_config"] = dict(best_config)
        if best_throughput is not None:
            attrs["best_throughput"] = float(best_throughput)
        if what_worked is not None:
            attrs["what_worked"] = [dict(e) for e in what_worked]
        if what_failed is not None:
            attrs["what_failed"] = [dict(e) for e in what_failed]
        if remaining_gaps is not None:
            attrs["remaining_gaps"] = [dict(e) for e in remaining_gaps]
        if pitfalls is not None:
            attrs["pitfalls"] = [dict(e) for e in pitfalls]
        if stack_fingerprint is not None:
            attrs["stack_fingerprint"] = dict(stack_fingerprint)
        if last_profiled:
            attrs["last_profiled"] = last_profiled
        if sessions is not None:
            attrs["sessions"] = [dict(s) for s in sessions]
        if extra_attrs:
            attrs.update(extra_attrs)
        return self.propose_point(
            canonical_id=cid,
            kind=C.KIND_RECIPE,
            authority=authority,
            attrs=attrs,
            evidence=evidence,
            idempotency_key=idempotency_key or f"update_recipe:{cid}",
            prefer_sync=prefer_sync,
        )

    # ==================================================================
    # NDJSON drain (background flusher + CLOSE-time drain)
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
                    # Out-of-time rows are NOT attempted this drain, so
                    # ``attempts`` is preserved verbatim (no increment).
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
                attempts = int(envelope.get("attempts") or 0)
                if attempts >= C.MAX_FLUSH_ATTEMPTS:
                    # Belt-and-braces guard for rows enqueued before this
                    # check existed (or by a parallel writer): a row that
                    # already exhausted its budget is dead-lettered on
                    # the way in without re-attempting.
                    dead_letter += 1
                    self._audit_record(
                        op="drain", status="attempts_exhausted",
                        envelope_op=str(envelope.get("op", "")),
                        idempotency_key=str(envelope.get("idempotency_key", "")),
                        attempts=attempts,
                    )
                    continue
                outcome = self._flush_one(envelope)
                if outcome == "ok":
                    drained += 1
                elif outcome == "permanent":
                    dead_letter += 1
                else:
                    # Transient failure → bump attempts and re-serialise
                    # so subsequent drains can see the counter. When it
                    # crosses the threshold the row is dead-lettered
                    # rather than retried forever.
                    envelope = dict(envelope)
                    envelope["attempts"] = attempts + 1
                    if envelope["attempts"] >= C.MAX_FLUSH_ATTEMPTS:
                        dead_letter += 1
                        self._audit_record(
                            op="drain", status="attempts_exhausted",
                            envelope_op=str(envelope.get("op", "")),
                            idempotency_key=str(envelope.get("idempotency_key", "")),
                            attempts=envelope["attempts"],
                        )
                    else:
                        leftover_lines.append(
                            json.dumps(envelope, sort_keys=True),
                        )
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

        Each ``op`` is dispatched to its public client method with
        ``_enqueue_on_failure=False`` so a transient failure raises
        :class:`CortexKBError` back to us instead of silently
        re-enqueueing the row we just dequeued (which would cause
        infinite duplication for permanent errors).

        Classification:

        * ``transport`` / ``unknown`` → ``transient`` (KB unreachable;
          drain again next tick).
        * ``business`` with ``code="NOT_FOUND"`` → ``transient``. The
          dependency (a ``propose_point`` row that registers the
          referenced canonical_id) may still be ahead of us in the
          NDJSON queue — give the next drain a chance.
        * ``business`` (other codes) / ``validation`` → ``permanent``
          (dead-letter; row will never validate on the server).
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
                    _enqueue_on_failure=False,
                )
            elif op == "propose_edge":
                self.propose_edge(
                    from_canonical_id=str(payload.get("from_canonical_id", "")),
                    to_canonical_id=str(payload.get("to_canonical_id", "")),
                    edge_type=str(payload.get("edge_type", "")),
                    relation=str(payload.get("relation", "")),
                    authority=str(payload.get("authority") or C.AUTHORITY_EXPERIENTIAL),
                    attrs=payload.get("attrs") or {},
                    evidence=list(payload.get("evidence") or []),
                    source=str(payload.get("source") or C.SOURCE_AGENT_OBSERVATION),
                    prefer_sync=True,
                    _enqueue_on_failure=False,
                )
            else:
                return "permanent"
        except CortexKBError as exc:
            log.info("flush_one %s deferred: %s", op, exc)
            if exc.category == "validation":
                return "permanent"
            if exc.category == "business":
                # NOT_FOUND is treated as transient because the row
                # that registers the missing canonical_id may still be
                # ahead of this one in the queue; the attempts counter
                # caps how many drains before we give up and dead-letter
                # (handled in drain_pending).
                if (exc.code or "").upper() == "NOT_FOUND":
                    return "transient"
                return "permanent"
            return "transient"
        return "ok"


__all__ = [
    "CortexKBClient",
    "CortexKBError",
    "lesson_canonical_id",
    "parse_kb_error",
    "pitfall_canonical_id",
    "recipe_canonical_id",
]
