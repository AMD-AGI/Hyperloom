# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Read-vs-write dispatcher for the recipe-snapshot KB.

Routes calls between the local store and the central kb-service
according to the design fixed in 2026-05-28:

Writes — local-only::

    :meth:`put_recipe`, :meth:`append_attempt`,
    :meth:`delete_recipe` are forwarded verbatim to
    :class:`LocalRecipeStore`. The remote client never sees a
    write request, by construction (it doesn't expose write
    methods at all). The local store is the authoritative place
    new rows land.

Reads — remote-first via the SINGLE ``/recipes/search`` route, fall
through to local on absence / failure::

    The remote half is reached ONLY through ``/recipes/search``.
    ``get_recipe`` decodes the 5-tuple from the canonical_id into
    ``label_match`` and issues ONE search — the server decides
    exact-vs-relative fallback + ranking, the client takes the top
    row. ``get_history`` / ``list_recent`` / ``list_attempts`` /
    ``list_session_attempts`` are LOCAL-only (those routes are not
    used). For the search-backed read:
    1. If ``remote`` is configured AND enabled → call the central
       kb-service first.
    2. If the remote answer carries a non-empty result, return it
       as-is. The central server is the wider corpus (it
       aggregates rows written by other operators / older runs
       that shipped to ``/v1/points``); a hit there is most
       informative.
    3. If the remote answer is "absence" (None / empty list /
       404) we fall through to the local store. Rationale: the
       remote can lag arbitrarily behind local writes (writes go
       local-only under this design — the central server only
       picks up rows via separately-scheduled bulk ingest), so
       "remote says no" is not the same as "this row never
       existed". A local hit completes the read; a local miss
       returns the same absence shape the remote produced.
    4. If the remote raises :class:`RemoteRecipeClientError`
       (transport / 4xx / 5xx) we log + invoke the optional
       ``on_remote_failure`` callback, then fall through to the
       local store. Callers therefore never have to unwrap remote
       errors.

Reads — local-only mode::

    A dispatcher constructed with ``remote=None`` (e.g.
    ``--degraded-kb`` or no ``--cortex-kb-url``) skips step 1
    entirely; reads go directly to the local store. A
    ``remote.enabled=False`` client behaves the same way (the
    client itself short-circuits each call to "no info").

NB: We do NOT merge remote + local results. Each read is
satisfied by exactly one source. Merging would double-count rows
that exist in both stores at slightly different versions, and
silently obscure the fact that the central corpus is stale
relative to local writes.

This dispatcher is the only object the rest of the optimizer
should construct directly — the local store + remote client are
implementation details. Construction is cheap; all I/O is lazy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .canonical_id import InvalidCanonicalIdError, cid_to_path_components
from .local_store import LocalRecipeStore
from .remote_client import RemoteRecipeClient, RemoteRecipeClientError


log = logging.getLogger(__name__)

# Short, stable labels for the configured remote backend keyed by client class
# name (used by :meth:`RecipeKB._remote_label` for logs/audit trace). Unknown
# backends fall back to their class name.
_REMOTE_LABELS: dict[str, str] = {
    "RemoteRecipeClient": "cortex",
    "GbrainRemoteRecipeClient": "gbrain",
    "CompositeRemoteRecipeClient": "composite",
}


def _labels_from_canonical_id(canonical_id: str) -> dict[str, str]:
    """Decode a canonical_id into the 5-key ``label_match`` dict the
    central ``/recipes/search`` route expects.

    The five cid segments are already slug-clean (produced by
    ``recipe_canonical_id``), so they map 1:1 to the label values the
    server matches on.

    Args:
        canonical_id (str): Canonical recipe identity to decode.

    Returns:
        dict[str, str]: The 5-key ``label_match`` dict (``model`` /
            ``hardware`` / ``framework`` / ``framework_version`` /
            ``precision``).

    Raises:
        InvalidCanonicalIdError: If ``canonical_id`` is malformed; the
            caller falls back to a local read.
    """
    model, hardware, framework, framework_version, precision = (
        cid_to_path_components(canonical_id)
    )
    return {
        "model": model,
        "hardware": hardware,
        "framework": framework,
        "framework_version": framework_version,
        "precision": precision,
    }


# ---------------------------------------------------------------------------
# Schema translation: v2 wire → arbor on-disk
# ---------------------------------------------------------------------------
# The central kb-service speaks the v2 spec (``findings`` / ``failures`` /
# ``gaps`` / ``body`` / ``metrics``). The dispatcher returns rows in arbor
# shape (``what_worked`` / ``what_failed`` / ``remaining_gaps`` /
# top-level ``best_config`` / ``best_throughput`` / ``stack_fingerprint``)
# so callers always see one consistent shape regardless of which store
# satisfied the read.
#
# Translation rules:
#
# * v2 ``findings``    → arbor ``what_worked`` (list passed through
#                       verbatim — the server returns the agreed
#                       sub-shape; the client does not re-wrap items).
# * v2 ``failures``    → arbor ``what_failed``    (same).
# * v2 ``gaps``        → arbor ``remaining_gaps`` (same).
# * v2 ``body.best_config`` / ``body.stack_fingerprint`` /
#   ``body.last_profiled`` / ``body.sessions`` / ``body.prs_tested`` →
#   pulled out as top-level arbor fields.
# * v2 ``metrics.throughput`` (or ``body.best_throughput``) → arbor
#   ``best_throughput``.
# * v2 ``labels.{model,hardware,framework,framework_version,precision}``
#   → top-level arbor identity fields (so an arbor consumer can read
#   them without parsing canonical_id). The remote kb-service and local
#   store both key the 4th identity dimension as ``framework_version``.
# * v2-only fields (``authority``, ``confidence``, ``evidence_refs``,
#   ``provenance``, ``canonical_id``, ``version``, ``created_at``,
#   ``updated_at``) pass through unchanged — they're additive on top
#   of arbor's shape.
def _v2_to_arbor(v2_payload: dict[str, Any]) -> dict[str, Any]:
    """Translate a v2-spec recipe dict into the arbor on-disk shape.

    Tolerant of missing keys — the central server always returns the
    full v2 envelope, but a partially-populated row (e.g. an old
    archive that pre-dates the field) shouldn't crash the read.

    Args:
        v2_payload (dict[str, Any]): A v2-spec recipe dict from the
            central kb-service.

    Returns:
        dict[str, Any]: The same row in arbor on-disk shape; an empty
            dict if ``v2_payload`` is not a dict.
    """
    if not isinstance(v2_payload, dict):
        return {}
    # Idempotency guard. A remote may hand us a row that is ALREADY in
    # arbor shape: the gbrain client (``GbrainRemoteRecipeClient``)
    # pre-translates pages into the ``Recipe.to_dict()`` layout, unlike
    # the central kb-service which speaks the nested v2 envelope. Running
    # the v2->arbor projection again on an already-arbor row would read
    # from absent top-level ``body``/``labels``/``findings`` and silently
    # null out ``model`` / ``best_config`` / ``best_throughput`` /
    # ``what_worked`` — wiping the warm-start champion while leaving
    # canonical_id + pitfalls/lessons intact (so the hit still mis-reports
    # as exact). A real v2 row always carries ``labels`` + ``body``; the
    # absence of EVERY v2 envelope marker means the row is already arbor,
    # so we pass it through untouched.
    if not any(
        marker in v2_payload
        for marker in ("body", "labels", "findings", "failures", "gaps", "metrics")
    ):
        return dict(v2_payload)
    body    = v2_payload.get("body")    or {}
    metrics = v2_payload.get("metrics") or {}
    labels  = v2_payload.get("labels")  or {}
    if not isinstance(body, dict):
        body = {}
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(labels, dict):
        labels = {}

    arbor: dict[str, Any] = {
        # store-managed metadata
        "canonical_id":      v2_payload.get("canonical_id", ""),
        "version":           v2_payload.get("version", 1),
        "created_at":        v2_payload.get("created_at", ""),
        "updated_at":        v2_payload.get("updated_at", ""),
        # 5-tuple identity from labels (with empty fallback when the
        # central row pre-dates the labels stamp)
        "model":             str(labels.get("model") or ""),
        "hardware":          str(labels.get("hardware") or ""),
        "framework":         str(labels.get("framework") or ""),
        "framework_version": str(labels.get("framework_version") or ""),
        "precision":         str(labels.get("precision") or ""),
        # arbor payload pulled out of body / metrics
        "best_config":       dict(body.get("best_config") or {}),
        "best_throughput":   float(
            body.get("best_throughput")
            or metrics.get("throughput")
            or 0.0
        ),
        "what_worked":       list(v2_payload.get("findings") or []),
        "what_failed":       list(v2_payload.get("failures") or []),
        "remaining_gaps":    list(v2_payload.get("gaps") or []),
        "prs_tested":        list(body.get("prs_tested") or []),
        "pitfalls":          list(v2_payload.get("pitfalls") or []),
        "lessons":           list(v2_payload.get("lessons") or []),
        "last_profiled":     str(body.get("last_profiled") or ""),
        "stack_fingerprint": dict(body.get("stack_fingerprint") or {}),
        "sessions":          list(body.get("sessions") or []),
        # v2-only audit fields pass through
        "authority":     v2_payload.get("authority", "EXPERIENTIAL"),
        "confidence":    v2_payload.get("confidence", 0.85),
        "evidence_refs": list(v2_payload.get("evidence_refs") or []),
        "provenance":    dict(v2_payload.get("provenance") or {}),
    }
    # Carry through any extra top-level keys the envelope shipped that
    # aren't part of the v2 wire structure (``body`` / ``labels`` /
    # ``metrics`` / ``findings`` / ``failures`` / ``gaps``) and aren't
    # already mapped above. This mirrors the local store's arbor shape
    # (``Recipe.to_dict`` splats free-form ``extras`` — e.g. the workload
    # knobs ``tp`` / ``ep`` / ``conc`` / ``framework_version`` /
    # ``quant_scheme`` — at the top level), so the dispatcher's
    # ``prefer`` rerank reads identical fields whether the row came from
    # a remote or local. Reserved keys never get clobbered.
    _wire_only = {"body", "labels", "metrics", "findings", "failures", "gaps"}
    for key, val in v2_payload.items():
        if key in _wire_only or key in arbor:
            continue
        arbor[key] = val
    return arbor


# ---------------------------------------------------------------------------
# Client-side rerank by ``prefer`` similarity
# ---------------------------------------------------------------------------
# The ``required`` filter (5-tuple ``label_match``) decides reusability;
# ``prefer`` decides similarity. Neither the cortex kb-service nor the
# gbrain page store rank by ``prefer`` server-side, so the dispatcher does
# a stable client-side rerank over the already-arbor rows: higher
# prefer-hit count first, ties broken by the backend's original order
# (the sort is stable). Rows are NEVER dropped — prefer only reorders.
#
# Match rules per field:
# * ``framework_version`` / ``quant_scheme`` / ``workload_mode`` — exact
#   string equality (case-insensitive, stripped).
# * numeric workload knobs (``tp`` / ``ep`` / ``pp`` / ``conc`` / ``isl``
#   / ``osl`` / ``max_model_len``) — exact int/float equality.
# A field absent on the row contributes 0 (no penalty, no credit).
_PREFER_NUMERIC_KEYS: tuple[str, ...] = (
    "tp", "ep", "pp", "conc", "isl", "osl", "max_model_len",
)
_PREFER_STRING_KEYS: tuple[str, ...] = (
    "framework_version", "quant_scheme", "workload_mode",
)


def _prefer_score(row: dict[str, Any], prefer: dict[str, Any]) -> int:
    """Count how many ``prefer`` fields the (flat arbor) row matches."""
    if not isinstance(row, dict):
        return 0
    score = 0
    for key in _PREFER_NUMERIC_KEYS:
        want = prefer.get(key)
        if want in (None, "", 0):
            continue
        have = row.get(key)
        if have in (None, ""):
            continue
        try:
            if float(have) == float(want):
                score += 1
        except (TypeError, ValueError):
            continue
    for key in _PREFER_STRING_KEYS:
        want = str(prefer.get(key) or "").strip().lower()
        if not want:
            continue
        have = str(row.get(key) or "").strip().lower()
        if have and have == want:
            score += 1
    return score


def _rerank_by_prefer(
    rows: list[dict[str, Any]], prefer: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Stable rerank of flat arbor rows by ``prefer`` similarity.

    No-op when ``prefer`` is empty. Never drops rows; only reorders so a
    closer-workload recipe surfaces first.
    """
    if not prefer:
        return rows
    return sorted(rows, key=lambda r: _prefer_score(r, prefer), reverse=True)


@dataclass
class RecipeKB:
    """Local-write / remote-read-with-fallback dispatcher.

    Args:
        local: Authoritative local store. REQUIRED — there is no
            "remote-only" mode under this design (writes must
            always have somewhere to land).
        remote: Optional read-side central kb-service client.
            ``None`` (or a client with ``enabled=False``) makes
            reads short-circuit to the local store.
        on_remote_failure: Callback invoked when a read against
            the central kb-service fails and the dispatcher falls
            back to local. Receives ``(method_name, exception)``.
            Defaults to ``log.warning`` so audit-collector hooks
            can wire a structured event in without a
            dispatcher-internal change.
        audit_hook: Optional callback invoked (best-effort, never
            raising into the caller) after every read/write with a
            structured event dict describing whether the remote was
            used, the request, and the resolution. Defaults to
            ``None`` (no audit). Wired by the CLI to append a
            ``recipe_snapshot/.audit.jsonl`` trace.
    """

    local: LocalRecipeStore
    remote: RemoteRecipeClient | None = None
    on_remote_failure: Any = None
    audit_hook: Any = None

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """Always ``True`` — the dispatcher is usable whenever it
        exists, because the local store is always present (writes land
        locally; reads fall back to local). Remote reachability is a
        separate concern handled by :meth:`_remote_active`. Exposed so
        call sites that historically probed ``client.enabled`` on the
        v1 ``CortexKBClient`` keep working against the v2 dispatcher
        (e.g. ``coordinator._ensure_cortex_t0_anchored``); a missing
        attribute there would silently skip the SDK-fallback T0 anchor.

        Returns:
            bool: Always ``True``.
        """
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _remote_active(self) -> bool:
        """``True`` iff the remote client exists and is enabled.

        Healthcheck is intentionally NOT issued here — we trust the
        per-call retry budget (2s + 1 retry foreground / 10s + 3
        retries background) to detect "service unhealthy" within an
        SLO that matches what callers already expect from the prior
        client. Adding a separate ping doubles RTT for every read.

        Returns:
            bool: ``True`` iff a remote client exists and is enabled.
        """
        return self.remote is not None and bool(self.remote.enabled)

    def _normalize_remote_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Project a remote row into the consistent arbor shape callers see.

        Every remote backend (cortex kb-service, gbrain, and any future
        KB) returns the SAME unified nested KB-interface envelope
        (``labels`` / ``body`` / ``metrics`` / ``findings`` / ``failures``
        / ``gaps`` / ``lessons`` / ``pitfalls``). The dispatcher runs a
        single :func:`_v2_to_arbor` translation regardless of which
        backend served the row — backend-native storage shapes are the
        adapter's private detail and never reach this point.

        :func:`_v2_to_arbor` keeps an idempotency guard so a row that is
        already in flat arbor shape (e.g. a mis-built adapter) passes
        through untouched rather than being silently emptied.
        """
        return _v2_to_arbor(row)

    def _remote_label(self) -> str:
        """A short, stable label for the configured remote backend.

        Returns:
            str: ``"none"`` when no remote, else ``"cortex"`` /
                ``"gbrain"`` / ``"composite"`` (falling back to the
                client class name for any future backend).
        """
        if self.remote is None:
            return "none"
        return _REMOTE_LABELS.get(
            type(self.remote).__name__, type(self.remote).__name__,
        )

    def _emit_audit(self, event: dict[str, Any]) -> None:
        """Best-effort emit one audit event to ``audit_hook`` (never raises).

        Args:
            event (dict[str, Any]): The structured audit record.
        """
        hook = self.audit_hook
        if not callable(hook):
            return
        try:
            hook(event)
        except Exception:  # noqa: BLE001 - audit must never break a KB op
            log.debug("recipe_kb: audit_hook raised", exc_info=True)

    def _read_audit_event(
        self,
        *,
        method: str,
        resolution: str,
        row: dict[str, Any] | None,
        canonical_id: str = "",
        prefer: dict[str, Any] | None = None,
        label_match: dict[str, Any] | None = None,
        candidates: int = 0,
    ) -> dict[str, Any]:
        """Build a structured audit record for one recipe-snapshot read.

        Captures whether the remote was consulted and which backend served
        the row (``remote``), the request (``canonical_id`` / ``prefer`` /
        ``label_match``), how it resolved (``remote`` / ``remote_miss`` /
        ``remote_error`` / ``local``), and a compact view of the returned row
        so the trace answers "was the recipe-snapshot used, what did we ask,
        and what came back" without storing the full payload.

        Args:
            method (str): The read method (``get_recipe`` / ``search``).
            resolution (str): How the read resolved.
            row (dict[str, Any] | None): The resolved row (or first row).
            canonical_id (str): Requested canonical id (``get_recipe``).
            prefer (dict[str, Any] | None): Rerank hints, if any.
            label_match (dict[str, Any] | None): Search filter, if any.
            candidates (int): Number of remote candidate rows considered.

        Returns:
            dict[str, Any]: The audit event (no timestamp; the hook stamps it).
        """
        result: dict[str, Any] | None = None
        if isinstance(row, dict):
            result = {
                "canonical_id": str(row.get("canonical_id") or ""),
                "exact": bool(
                    canonical_id
                    and str(row.get("canonical_id") or "") == canonical_id
                ),
                "best_throughput": float(row.get("best_throughput") or 0.0),
                "best_config_nonempty": bool(row.get("best_config")),
            }
            # Per-path provenance — only populated when the composite remote
            # served the read (single-source backends leave these absent).
            # ``sources``: which backend(s) contributed this merged row;
            # ``best_config_source``: which path supplied the replayable
            # champion config; ``field_sources``: per-field attribution;
            # ``source_candidates``: candidate-row count each path returned.
            # These let the trace evaluate gbrain vs cortex contribution.
            field_sources = row.get("_field_sources")
            if isinstance(field_sources, dict) and field_sources:
                result["field_sources"] = dict(field_sources)
                result["best_config_source"] = field_sources.get("best_config")
            sources = row.get("_sources")
            if isinstance(sources, list) and sources:
                result["sources"] = list(sources)
            source_candidates = row.get("_source_candidates")
            if isinstance(source_candidates, dict) and source_candidates:
                result["source_candidates"] = dict(source_candidates)
        return {
            "method": method,
            "remote": self._remote_label(),
            "resolution": resolution,
            "hit": row is not None,
            "candidates": int(candidates),
            "request": {
                "canonical_id": canonical_id or None,
                "prefer_keys": sorted((prefer or {}).keys()),
                "label_match": label_match or None,
            },
            "result": result,
        }

    def _note_failure(self, method: str, exc: Exception) -> None:
        """Report a remote read failure before local fall-through.

        Invokes the ``on_remote_failure`` callback when one is
        configured (logging if the callback itself raises); otherwise
        logs a warning. Never raises.

        Args:
            method (str): Name of the dispatcher method that failed
                against the remote.
            exc (Exception): The remote failure being reported.
        """
        if callable(self.on_remote_failure):
            try:
                self.on_remote_failure(method, exc)
            except Exception:  # noqa: BLE001
                log.exception(
                    "on_remote_failure callback raised — continuing "
                    "with local fallback for %s", method,
                )
        else:
            log.warning(
                "recipe_kb: remote %s failed (%s); falling back to "
                "local store",
                method, exc,
            )

    # ==================================================================
    # Writes — local only
    # ==================================================================
    def put_recipe(
        self,
        *,
        canonical_id: str,
        model: str = "",
        hardware: str = "",
        framework: str = "",
        framework_version: str = "",
        precision: str = "",
        best_config: dict[str, str] | None = None,
        best_throughput: float = 0.0,
        what_worked: list[Any] | None = None,
        what_failed: list[Any] | None = None,
        remaining_gaps: list[Any] | None = None,
        prs_tested: list[Any] | None = None,
        pitfalls: list[Any] | None = None,
        lessons: list[Any] | None = None,
        last_profiled: str = "",
        stack_fingerprint: dict[str, str] | None = None,
        sessions: list[Any] | None = None,
        authority: str = "EXPERIENTIAL",
        confidence: float = 0.85,
        evidence_refs: list[Any] | None = None,
        provenance: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write a recipe row LOCALLY ONLY in the arbor schema.

        Never touches the central kb-service. Field shape mirrors
        arbor's ``Recipe`` (see :mod:`recipe_kb.schema`); all
        arguments are forwarded verbatim to
        :meth:`LocalRecipeStore.put_recipe`.

        Args:
            canonical_id (str): Canonical recipe identity; must be
                non-empty.
            model (str): Model identity slot.
            hardware (str): Hardware identity slot.
            framework (str): Framework identity slot.
            framework_version (str): Framework version identity slot.
            precision (str): Precision identity slot.
            best_config (dict[str, str] | None): Best-known config.
            best_throughput (float): Best measured throughput.
            what_worked (list[Any] | None): Findings that helped.
            what_failed (list[Any] | None): Findings that failed.
            remaining_gaps (list[Any] | None): Known remaining gaps.
            prs_tested (list[Any] | None): PRs tested.
            pitfalls (list[Any] | None): Known pitfalls.
            lessons (list[Any] | None): Lessons learned.
            last_profiled (str): Timestamp of last profiling run.
            stack_fingerprint (dict[str, str] | None): Stack
                fingerprint mapping.
            sessions (list[Any] | None): Per-session records.
            authority (str): Authority tier.
            confidence (float): Confidence score in ``[0, 1]``.
            evidence_refs (list[Any] | None): Supporting evidence refs.
            provenance (dict[str, Any] | None): Audit provenance.
            extras (dict[str, Any] | None): Free-form arbor keys.

        Returns:
            dict[str, Any]: ``{"canonical_id", "version", "created"}``.
        """
        return self.local.put_recipe(
            canonical_id=canonical_id,
            model=model,
            hardware=hardware,
            framework=framework,
            framework_version=framework_version,
            precision=precision,
            best_config=best_config,
            best_throughput=best_throughput,
            what_worked=what_worked,
            what_failed=what_failed,
            remaining_gaps=remaining_gaps,
            prs_tested=prs_tested,
            pitfalls=pitfalls,
            lessons=lessons,
            last_profiled=last_profiled,
            stack_fingerprint=stack_fingerprint,
            sessions=sessions,
            authority=authority,
            confidence=confidence,
            evidence_refs=evidence_refs,
            provenance=provenance,
            extras=extras,
        )

    def append_attempt(
        self,
        *,
        canonical_id: str,
        session_id: str,
        diff: dict[str, Any] | None = None,
        predicted_delta: dict[str, Any] | None = None,
        measured_metrics: dict[str, Any] | None = None,
        fitness: float | None = None,
        outcome: str = "",
        rationale: str = "",
        attempt_at: str | None = None,
    ) -> dict[str, Any]:
        """Append one attempt row LOCALLY ONLY.

        Forwards verbatim to :meth:`LocalRecipeStore.append_attempt`.

        Args:
            canonical_id (str): Parent recipe identity; must be
                non-empty.
            session_id (str): Owning session; must be non-empty.
            diff (dict[str, Any] | None): Config diff applied.
            predicted_delta (dict[str, Any] | None): Predicted metric
                deltas.
            measured_metrics (dict[str, Any] | None): Measured metrics.
            fitness (float | None): Scalar fitness score, or ``None``.
            outcome (str): Outcome label.
            rationale (str): Free-form rationale.
            attempt_at (str | None): Explicit ISO-8601 timestamp; auto
                stamped when ``None``.

        Returns:
            dict[str, Any]: A dict with keys ``id``,
                ``recipe_canonical_id`` and ``attempt_at``.
        """
        return self.local.append_attempt(
            canonical_id=canonical_id,
            session_id=session_id,
            diff=diff,
            predicted_delta=predicted_delta,
            measured_metrics=measured_metrics,
            fitness=fitness,
            outcome=outcome,
            rationale=rationale,
            attempt_at=attempt_at,
        )

    def delete_recipe(self, *, canonical_id: str) -> bool:
        """Delete the live recipe row LOCALLY ONLY (history preserved).

        Args:
            canonical_id (str): Canonical recipe identity; must be
                non-empty.

        Returns:
            bool: ``True`` iff a live row was removed.
        """
        return self.local.delete_recipe(canonical_id=canonical_id)

    # ==================================================================
    # Reads — remote-first, local fallback
    # ==================================================================
    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
        prefer: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Read a recipe row.

        Remote uses the SINGLE ``/recipes/search`` route: the 5-tuple
        decoded from ``canonical_id`` is passed as ``label_match`` and
        the central kb-service decides exact-vs-relative match +
        ranking. We deliberately do NOT hit ``GET /recipes/{cid}``;
        search is the only remote route.

        ``prefer`` (workload-similarity hints — ``tp`` / ``ep`` / ``pp``
        / ``conc`` / ``isl`` / ``osl`` / ``max_model_len`` /
        ``framework_version`` / ``quant_scheme`` / ``workload_mode``)
        does NOT change the ``required`` (5-tuple) filter; it only
        reranks the candidate rows so the closest-workload recipe is
        returned first. When ``prefer`` is set we ask the remote for a
        small candidate window instead of a single row, rerank
        client-side, and take the top.

        Local is an arbor-style exact read of the on-disk
        ``recipe.json`` for this canonical_id. It serves:

        * ``version``-pinned reads (history archive — search only
          returns live rows), and
        * the fall-through when remote is absent / empty / errors.

        Args:
            canonical_id (str): Canonical recipe identity.
            version (int | None): Specific archived version (served
                locally), or ``None`` for the live row.

        Returns:
            dict[str, Any] | None: The recipe row in arbor shape, or
                ``None`` when neither store has the row.
        """
        resolution = "local"
        if version is None and self._remote_active():
            try:
                labels = _labels_from_canonical_id(canonical_id)
                # With prefer hints we pull a candidate window so the
                # client-side rerank has rows to reorder; otherwise the
                # single top (server-ranked) row is enough.
                candidate_limit = 25 if prefer else 1
                rows = self.remote.search(  # type: ignore[union-attr]
                    label_match=labels, limit=candidate_limit, prefer=prefer,
                )
                if rows:
                    normalized = [self._normalize_remote_row(r) for r in rows]
                    ranked = _rerank_by_prefer(normalized, prefer)
                    self._emit_audit(self._read_audit_event(
                        method="get_recipe", resolution="remote",
                        row=ranked[0], canonical_id=canonical_id,
                        prefer=prefer, candidates=len(rows),
                    ))
                    return ranked[0]
                # remote miss — fall through to local.
                resolution = "remote_miss"
            except RemoteRecipeClientError as exc:
                self._note_failure("get_recipe", exc)
                resolution = "remote_error"
            except InvalidCanonicalIdError as exc:
                # Can't build label_match from a malformed cid; the
                # local store applies the same parse, so just degrade.
                log.warning("get_recipe: %s; local-only read", exc)
                resolution = "remote_error"
        local_row = self.local.get_recipe(
            canonical_id=canonical_id, version=version,
        )
        self._emit_audit(self._read_audit_event(
            method="get_recipe", resolution=resolution, row=local_row,
            canonical_id=canonical_id, prefer=prefer,
        ))
        return local_row

    def get_history(
        self,
        *,
        canonical_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read version history — LOCAL only.

        The central kb-service is reached through the single
        ``/recipes/search`` route only (see :meth:`get_recipe`);
        ``/history`` is not used. The local archive is authoritative
        for writes anyway, so the on-disk ``history/v{N}.json`` files
        are the source of truth here.

        Args:
            canonical_id (str): Canonical recipe identity.
            limit (int): Maximum number of archived rows to return.

        Returns:
            list[dict[str, Any]]: Archived prior versions ascending by
                version.
        """
        return self.local.get_history(
            canonical_id=canonical_id, limit=limit,
        )

    def list_recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Recent recipes — LOCAL only (remote = ``/recipes/search``
        route only; bare ``GET /recipes`` is not used).

        Args:
            limit (int): Maximum number of recipes to return.

        Returns:
            list[dict[str, Any]]: Recent live recipes, ordered
                ``updated_at DESC``.
        """
        return self.local.list_recent(limit=limit)

    def search(
        self,
        *,
        label_match: dict[str, Any] | None = None,
        metric_filters: dict[str, Any] | None = None,
        updated_since: str | None = None,
        order_by: str = "updated_at DESC",
        limit: int = 50,
        prefer: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Filtered search, remote-first with local fall-through.

        Single-source-of-record per call (no merging across stores
        — see the class docstring for why). Resolution:

        1. Remote returned a non-empty list → that list is reranked
           by ``prefer`` (similarity hints; never drops rows) and
           returned.
        2. Remote returned ``[]`` → fall through to local. A
           genuinely-empty central corpus matching this filter is
           rare; an empty result is more often "the central
           hasn't ingested our recent local writes yet", and
           callers nearly always benefit from seeing local matches
           in that case.
        3. Remote raised → log + fall through.

        ``prefer`` only reorders; the ``required`` filter
        (``label_match`` / ``metric_filters``) decides membership.
        """
        resolution = "local"
        if self._remote_active():
            try:
                rows = self.remote.search(  # type: ignore[union-attr]
                    label_match=label_match,
                    metric_filters=metric_filters,
                    updated_since=updated_since,
                    order_by=order_by,
                    limit=limit,
                    prefer=prefer,
                )
                # Empty remote search → fall through to local. A
                # genuinely-empty search corpus on a working remote
                # is rare; an empty result is more often a flaky
                # network or a freshly-bootstrapped service that
                # hasn't received our local writes yet.
                if rows:
                    normalized = [self._normalize_remote_row(r) for r in rows]
                    ranked = _rerank_by_prefer(normalized, prefer)
                    self._emit_audit(self._read_audit_event(
                        method="search", resolution="remote",
                        row=ranked[0] if ranked else None,
                        label_match=label_match, prefer=prefer,
                        candidates=len(rows),
                    ))
                    return ranked
                resolution = "remote_miss"
            except RemoteRecipeClientError as exc:
                self._note_failure("search", exc)
                resolution = "remote_error"
        local_rows = self.local.search(
            label_match=label_match,
            metric_filters=metric_filters,
            updated_since=updated_since,
            order_by=order_by,
            limit=limit,
        )
        ranked_local = _rerank_by_prefer(local_rows, prefer)
        self._emit_audit(self._read_audit_event(
            method="search", resolution=resolution,
            row=ranked_local[0] if ranked_local else None,
            label_match=label_match, prefer=prefer,
        ))
        return ranked_local

    def list_attempts(
        self,
        *,
        canonical_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Attempts for one recipe — LOCAL only (remote =
        ``/recipes/search`` route only).

        Args:
            canonical_id (str): Parent recipe identity.
            limit (int): Maximum number of attempts to return.

        Returns:
            list[dict[str, Any]]: Attempt rows newest-first.
        """
        return self.local.list_attempts(
            canonical_id=canonical_id, limit=limit,
        )

    def list_session_attempts(
        self,
        *,
        session_id: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Session attempts — LOCAL only (remote = ``/recipes/search``
        route only).

        Args:
            session_id (str): Session whose attempts to collect.
            limit (int): Maximum number of attempts to return.

        Returns:
            list[dict[str, Any]]: Attempts for the session oldest-first.
        """
        return self.local.list_session_attempts(
            session_id=session_id, limit=limit,
        )

    # ==================================================================
    # Lifecycle
    # ==================================================================
    def close(self) -> None:
        """Release the remote client's HTTP transport (no-op for the
        local store).

        Idempotent. Safe to call from a CLI atexit handler.
        """
        if self.remote is not None:
            try:
                self.remote.close()
            except Exception:  # noqa: BLE001
                log.exception("recipe_kb: remote.close raised")


__all__ = [
    "RecipeKB",
]
