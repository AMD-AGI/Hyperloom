"""Read-vs-write dispatcher for the recipe-snapshot KB.

Routes calls between the local store and the central kb-service
according to the design fixed in 2026-05-28:

* **Writes go LOCAL ONLY** — :meth:`put_recipe`,
  :meth:`append_attempt`, :meth:`delete_recipe` are all forwarded
  verbatim to :class:`LocalRecipeStore`. The remote client never
  sees a write request, by construction (it doesn't expose write
  methods at all).
* **Reads prefer the central kb-service when configured AND
  reachable, falling back to the local store on any failure.** The
  fallback is silent at the call-site (the dispatcher logs a
  WARNING) so warm-start / search paths don't have to unwrap a
  RemoteRecipeClientError on every call.
* **Reads go local-only when no remote is configured.** A caller
  who passes ``remote=None`` (e.g. ``--degraded-kb`` or no
  ``--cortex-kb-url``) sees the same dispatcher API; reads just
  short-circuit to the local store.

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
        """Read recipe row.

        * If ``remote`` is active, try it first.
        * On any :class:`RemoteRecipeClientError`, log + fall through
          to the local store.
        * On a remote 404 (returned as ``None``), DO NOT fall through
          — the central server is the canonical source for "this id
          was never registered there", and the local row (if any) is
          a parallel-universe artifact the caller should still see if
          they ask the dispatcher again with ``--degraded-kb``.
          Behaviour is: ``remote=None`` + local hit → return local;
          ``remote=hit`` → return remote; ``remote=miss`` → return
          local (so a freshly-bootstrapped remote can't shadow our
          local-of-truth). This is intentional and matches the
          local-write design: the local store is authoritative.
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
        """Filtered search.

        We do NOT merge remote + local results — central server is
        the authoritative search corpus when reachable. Local
        fallback only fires when remote is absent / unhealthy.
        Merging would risk double-counting recipes that exist in
        both stores at slightly different versions.
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
