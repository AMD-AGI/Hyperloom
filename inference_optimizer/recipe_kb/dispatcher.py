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

Reads — remote-first, fall through to local on absence / failure:
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

from .local_store import LocalRecipeStore
from .remote_client import RemoteRecipeClient, RemoteRecipeClientError


log = logging.getLogger(__name__)


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
        labels: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        findings: list[Any] | None = None,
        failures: list[Any] | None = None,
        pitfalls: list[Any] | None = None,
        lessons: list[Any] | None = None,
        gaps: list[Any] | None = None,
        authority: str = "EXPERIENTIAL",
        confidence: float = 0.85,
        evidence_refs: list[Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write a recipe row LOCALLY ONLY.

        Returns ``{"canonical_id", "version", "created"}`` matching
        the central server's PUT response shape. Never touches the
        central kb-service.
        """
        return self.local.put_recipe(
            canonical_id=canonical_id,
            labels=labels,
            body=body,
            metrics=metrics,
            findings=findings,
            failures=failures,
            pitfalls=pitfalls,
            lessons=lessons,
            gaps=gaps,
            authority=authority,
            confidence=confidence,
            evidence_refs=evidence_refs,
            provenance=provenance,
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
        """Read recipe row, remote-first with local fall-through.

        Resolution order (matches the class-level docstring):

        1. Remote enabled → ``remote.get_recipe(...)``.
           A non-None response is returned verbatim.
        2. Remote returned ``None`` (server replied 404 → row
           absent on central) → check the local store. The local
           row may exist if this operator wrote it but the central
           ingest hasn't caught up yet.
        3. Remote raised → log + ``on_remote_failure`` callback,
           then check the local store.
        4. Remote disabled / not configured → local-only.

        Returns ``None`` only when both stores agree the row is
        absent. ``version=N`` is forwarded to whichever store
        ends up satisfying the call.
        """
        if self._remote_active():
            try:
                row = self.remote.get_recipe(  # type: ignore[union-attr]
                    canonical_id=canonical_id, version=version,
                )
                if row is not None:
                    return row
                # remote miss — fall through to local (see docstring).
            except RemoteRecipeClientError as exc:
                self._note_failure("get_recipe", exc)
        return self.local.get_recipe(
            canonical_id=canonical_id, version=version,
        )

    def get_history(
        self,
        *,
        canonical_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read history.

        Remote returns an empty list for unknown cids (per spec, no
        404), so we treat ``[]`` as "remote authoritative" only if
        the remote answered without error. On a transport / business
        failure we fall through to the local store; the caller still
        sees the local archive even if the central one is missing.
        """
        if self._remote_active():
            try:
                rows = self.remote.get_history(  # type: ignore[union-attr]
                    canonical_id=canonical_id, limit=limit,
                )
                if rows:
                    return rows
                # Remote returned []; check the local store too —
                # the dispatcher's invariant is "writes are local",
                # so the local archive can be richer than central.
            except RemoteRecipeClientError as exc:
                self._note_failure("get_history", exc)
        return self.local.get_history(
            canonical_id=canonical_id, limit=limit,
        )

    def list_recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if self._remote_active():
            try:
                rows = self.remote.list_recent(  # type: ignore[union-attr]
                    limit=limit,
                )
                if rows:
                    return rows
            except RemoteRecipeClientError as exc:
                self._note_failure("list_recent", exc)
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
                    return rows
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
        if self._remote_active():
            try:
                rows = self.remote.list_attempts(  # type: ignore[union-attr]
                    canonical_id=canonical_id, limit=limit,
                )
                if rows:
                    return rows
            except RemoteRecipeClientError as exc:
                self._note_failure("list_attempts", exc)
        return self.local.list_attempts(
            canonical_id=canonical_id, limit=limit,
        )

    def list_session_attempts(
        self,
        *,
        session_id: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if self._remote_active():
            try:
                rows = self.remote.list_session_attempts(  # type: ignore[union-attr]
                    session_id=session_id, limit=limit,
                )
                if rows:
                    return rows
            except RemoteRecipeClientError as exc:
                self._note_failure("list_session_attempts", exc)
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
