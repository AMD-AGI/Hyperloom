"""Read-vs-write dispatcher for the recipe-snapshot KB.

Routes calls between the local store and the central kb-service
according to the design fixed in 2026-05-28:

Writes — local-only:
    :meth:`put_recipe`, :meth:`append_attempt`,
    :meth:`delete_recipe` are forwarded verbatim to
    :class:`LocalRecipeStore`. The remote client never sees a
    write request, by construction (it doesn't expose write
    methods at all). The local store is the authoritative place
    new rows land.

Reads — remote-first via the SINGLE ``/recipes/search`` route, fall
through to local on absence / failure:
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

Reads — local-only mode:
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


def _labels_from_canonical_id(canonical_id: str) -> dict[str, str]:
    """Decode a canonical_id into the 5-key ``label_match`` dict the
    central ``/recipes/search`` route expects.

    The five cid segments are already slug-clean (produced by
    ``recipe_canonical_id``), so they map 1:1 to the label values the
    server matches on. Raises :class:`InvalidCanonicalIdError` for a
    malformed id — the caller falls back to a local read.
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
#   them without parsing canonical_id).
# * v2-only fields (``authority``, ``confidence``, ``evidence_refs``,
#   ``provenance``, ``canonical_id``, ``version``, ``created_at``,
#   ``updated_at``) pass through unchanged — they're additive on top
#   of arbor's shape.
def _v2_to_arbor(v2_payload: dict[str, Any]) -> dict[str, Any]:
    """Translate a v2-spec recipe dict into the arbor on-disk shape.

    Tolerant of missing keys — the central server always returns the
    full v2 envelope, but a partially-populated row (e.g. an old
    archive that pre-dates the field) shouldn't crash the read.
    """
    if not isinstance(v2_payload, dict):
        return {}
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
    return arbor


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
    """

    local: LocalRecipeStore
    remote: RemoteRecipeClient | None = None
    on_remote_failure: Any = None

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
        """
        return self.remote is not None and bool(self.remote.enabled)

    def _note_failure(self, method: str, exc: Exception) -> None:
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

        Returns ``{"canonical_id", "version", "created"}``. Never
        touches the central kb-service. Field shape mirrors arbor's
        ``Recipe`` (see :mod:`recipe_kb.schema`).
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
        """Append one attempt row LOCALLY ONLY."""
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
        """Delete the live recipe row LOCALLY ONLY (history preserved)."""
        return self.local.delete_recipe(canonical_id=canonical_id)

    # ==================================================================
    # Reads — remote-first, local fallback
    # ==================================================================
    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Read a recipe row.

        Remote uses the SINGLE ``/recipes/search`` route: the 5-tuple
        decoded from ``canonical_id`` is passed as ``label_match`` and
        the central kb-service decides exact-vs-relative match +
        ranking — the client issues ONE search and takes the top
        (server-ranked) row. We deliberately do NOT hit
        ``GET /recipes/{cid}``; search is the only remote route.

        Local is an arbor-style exact read of the on-disk
        ``recipe.json`` for this canonical_id. It serves:

        * ``version``-pinned reads (history archive — search only
          returns live rows), and
        * the fall-through when remote is absent / empty / errors.

        Returns ``None`` only when neither store has the row.
        """
        if version is None and self._remote_active():
            try:
                labels = _labels_from_canonical_id(canonical_id)
                rows = self.remote.search(  # type: ignore[union-attr]
                    label_match=labels, limit=1,
                )
                if rows:
                    return _v2_to_arbor(rows[0])
                # remote miss — fall through to local.
            except RemoteRecipeClientError as exc:
                self._note_failure("get_recipe", exc)
            except InvalidCanonicalIdError as exc:
                # Can't build label_match from a malformed cid; the
                # local store applies the same parse, so just degrade.
                log.warning("get_recipe: %s; local-only read", exc)
        return self.local.get_recipe(
            canonical_id=canonical_id, version=version,
        )

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
        """
        return self.local.get_history(
            canonical_id=canonical_id, limit=limit,
        )

    def list_recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Recent recipes — LOCAL only (remote = ``/recipes/search``
        route only; bare ``GET /recipes`` is not used)."""
        return self.local.list_recent(limit=limit)

    def search(
        self,
        *,
        label_match: dict[str, Any] | None = None,
        metric_filters: dict[str, Any] | None = None,
        updated_since: str | None = None,
        order_by: str = "updated_at DESC",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Filtered search, remote-first with local fall-through.

        Single-source-of-record per call (no merging across stores
        — see the class docstring for why). Resolution:

        1. Remote returned a non-empty list → that list is
           returned verbatim.
        2. Remote returned ``[]`` → fall through to local. A
           genuinely-empty central corpus matching this filter is
           rare; an empty result is more often "the central
           hasn't ingested our recent local writes yet", and
           callers nearly always benefit from seeing local matches
           in that case.
        3. Remote raised → log + fall through.
        """
        if self._remote_active():
            try:
                rows = self.remote.search(  # type: ignore[union-attr]
                    label_match=label_match,
                    metric_filters=metric_filters,
                    updated_since=updated_since,
                    order_by=order_by,
                    limit=limit,
                )
                # Empty remote search → fall through to local. A
                # genuinely-empty search corpus on a working remote
                # is rare; an empty result is more often a flaky
                # network or a freshly-bootstrapped service that
                # hasn't received our local writes yet.
                if rows:
                    return [_v2_to_arbor(r) for r in rows]
            except RemoteRecipeClientError as exc:
                self._note_failure("search", exc)
        return self.local.search(
            label_match=label_match,
            metric_filters=metric_filters,
            updated_since=updated_since,
            order_by=order_by,
            limit=limit,
        )

    def list_attempts(
        self,
        *,
        canonical_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Attempts for one recipe — LOCAL only (remote =
        ``/recipes/search`` route only)."""
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
        route only)."""
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
