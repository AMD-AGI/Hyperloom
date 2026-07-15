# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Recipe KB dispatcher: local writes, remote-first reads with local fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from . import RemoteRecipeClientError
from .canonical_id import InvalidCanonicalIdError, cid_to_path_components
from .local_store import LocalRecipeStore


log = logging.getLogger(__name__)

# Short, stable labels for the configured remote backend keyed by client class
# name. Unknown backends fall back to their class name.
_REMOTE_LABELS: dict[str, str] = {
    "GbrainRemoteRecipeClient": "gbrain",
}


def _labels_from_canonical_id(canonical_id: str) -> dict[str, str]:
    """Decode a canonical_id into the 7-key ``label_match`` dict the
    ``/recipes/search`` route expects.

    The seven cid segments are already slug-clean, so they map 1:1 to the
    label values the server matches on.

    Args:
        canonical_id (str): Canonical recipe identity to decode.

    Returns:
        dict[str, str]: The 7-key ``label_match`` dict (``model`` /
            ``hardware`` / ``framework_name`` / ``framework_version`` /
            ``precision`` / ``model_type`` / ``architectures``).

    Raises:
        InvalidCanonicalIdError: If ``canonical_id`` is malformed; the
            caller falls back to a local read.
    """
    model, hardware, framework_name, model_type, architectures, framework_version, precision = cid_to_path_components(
        canonical_id
    )
    return {
        "model": model,
        "hardware": hardware,
        "framework_name": framework_name,
        "model_type": model_type,
        "architectures": architectures,
        "framework_version": framework_version,
        "precision": precision,
    }


# Schema translation: v2 wire → arbor on-disk. The central kb-service speaks
# the v2 spec (findings / failures / gaps / body / metrics); the dispatcher
# returns rows in arbor shape so callers see one consistent shape regardless of
# which store satisfied the read.
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
    # Idempotency guard: a remote may hand us a row already in arbor shape.
    # Re-running the v2->arbor projection on it would null out identity /
    # config fields. A real v2 row always carries ``labels`` + ``body``, so
    # the absence of every v2 envelope marker means the row is already arbor.
    if not any(marker in v2_payload for marker in ("body", "labels", "findings", "failures", "gaps", "metrics")):
        return dict(v2_payload)
    body = v2_payload.get("body") or {}
    metrics = v2_payload.get("metrics") or {}
    labels = v2_payload.get("labels") or {}
    if not isinstance(body, dict):
        body = {}
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(labels, dict):
        labels = {}

    arbor: dict[str, Any] = {
        # store-managed metadata
        "canonical_id": v2_payload.get("canonical_id", ""),
        "version": v2_payload.get("version", 1),
        "created_at": v2_payload.get("created_at", ""),
        "updated_at": v2_payload.get("updated_at", ""),
        # identity from labels (empty fallback when the row lacks the stamp)
        "model": str(labels.get("model") or ""),
        "hardware": str(labels.get("hardware") or ""),
        # Back-compat: rows predating the framework_name rename use ``framework``.
        "framework_name": str(labels.get("framework_name") or labels.get("framework") or ""),
        "framework_version": str(labels.get("framework_version") or ""),
        "precision": str(labels.get("precision") or ""),
        # arbor payload pulled out of body / metrics.
        # kb-extract recipes store optimized args directly in body.extra_server_args
        # rather than body.best_config; synthesize best_config when absent.
        "best_config": dict(body.get("best_config") or {})
        or (
            {"extra_server_args": str(body.get("extra_server_args") or "").strip()}
            if body.get("extra_server_args")
            else {}
        ),
        "best_throughput": float(body.get("best_throughput") or metrics.get("throughput") or 0.0),
        "what_worked": list(v2_payload.get("findings") or []),
        "what_failed": list(v2_payload.get("failures") or []),
        "remaining_gaps": list(v2_payload.get("gaps") or []),
        "prs_tested": list(body.get("prs_tested") or []),
        "pitfalls": list(v2_payload.get("pitfalls") or []),
        "lessons": list(v2_payload.get("lessons") or []),
        "last_profiled": str(body.get("last_profiled") or ""),
        "stack_fingerprint": dict(body.get("stack_fingerprint") or {}),
        "sessions": list(body.get("sessions") or []),
        "authority": v2_payload.get("authority", "EXPERIENTIAL"),
        "confidence": v2_payload.get("confidence", 0.85),
        "evidence_refs": list(v2_payload.get("evidence_refs") or []),
        "provenance": dict(v2_payload.get("provenance") or {}),
    }
    # Carry through any extra top-level keys not part of the v2 wire structure
    # and not already mapped above, so the ``prefer`` rerank reads identical
    # fields whether the row came from remote or local. Reserved keys are never
    # clobbered.
    _wire_only = {"body", "labels", "metrics", "findings", "failures", "gaps"}
    for key, val in v2_payload.items():
        if key in _wire_only or key in arbor:
            continue
        arbor[key] = val
    return arbor


# Client-side rerank by ``prefer`` similarity. The ``required`` filter
# (7-tuple ``label_match``) decides reusability; ``prefer`` only reorders the
# already-arbor rows (higher prefer-hit count first, stable ties). Numeric
# knobs use int/float equality; string keys use case-insensitive equality; a
# field absent on the row contributes 0.
_PREFER_NUMERIC_KEYS: tuple[str, ...] = (
    "tp",
    "ep",
    "pp",
    "conc",
    "isl",
    "osl",
    "max_model_len",
)
_PREFER_STRING_KEYS: tuple[str, ...] = (
    "framework_version",
    "quant_scheme",
    "workload_mode",
)


def _prefer_score(row: dict[str, Any], prefer: dict[str, Any]) -> int:
    """Count how many ``prefer`` fields the (flat arbor) row matches.

    Args:
        row: A flat arbor recipe row.
        prefer: Workload-similarity hints to compare against the row.

    Returns:
        The count of matching numeric and string preference fields.
    """
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
    rows: list[dict[str, Any]],
    prefer: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Stable rerank of flat arbor rows by ``prefer`` similarity.

    No-op when ``prefer`` is empty. Never drops rows; only reorders so a
    closer-workload recipe surfaces first.

    Args:
        rows: The flat arbor rows to reorder.
        prefer: Workload-similarity hints, or ``None`` to leave order intact.

    Returns:
        The rows reordered by descending preference score (or unchanged when
        ``prefer`` is empty).
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
    remote: Any = None  # read-side gbrain client (duck-typed); None = local-only
    on_remote_failure: Any = None
    audit_hook: Any = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _remote_active(self) -> bool:
        """``True`` iff the remote client exists and is enabled.

        Healthcheck is intentionally NOT issued here — we trust the per-call
        retry budget to detect an unhealthy service; adding a separate ping
        doubles RTT for every read.

        Returns:
            bool: ``True`` iff a remote client exists and is enabled.
        """
        return self.remote is not None and bool(self.remote.enabled)

    def _normalize_remote_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Project a remote row into the consistent arbor shape callers see.

        Every remote backend returns the same unified nested KB-interface
        envelope, so the dispatcher runs a single :func:`_v2_to_arbor`
        translation regardless of which backend served the row.
        :func:`_v2_to_arbor`'s idempotency guard passes an already-flat arbor
        row through untouched.

        Args:
            row: The remote row in the unified nested KB-interface envelope.

        Returns:
            The row projected into the flat arbor shape callers expect.
        """
        return _v2_to_arbor(row)

    def _remote_label(self) -> str:
        """A short, stable label for the configured remote backend.

        Returns:
            str: ``"none"`` when no remote, else ``"gbrain"`` (falling
                back to the client class name for any future backend).
        """
        if self.remote is None:
            return "none"
        return _REMOTE_LABELS.get(
            type(self.remote).__name__,
            type(self.remote).__name__,
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
                "exact": bool(canonical_id and str(row.get("canonical_id") or "") == canonical_id),
                "best_throughput": float(row.get("best_throughput") or 0.0),
                "best_config_nonempty": bool(row.get("best_config")),
            }
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

        Invokes the ``on_remote_failure`` callback when configured (logging if
        it raises); otherwise logs a warning. Never raises.

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
                    "on_remote_failure callback raised — continuing with local fallback for %s",
                    method,
                )
        else:
            log.warning(
                "recipe_kb: remote %s failed (%s); falling back to local store",
                method,
                exc,
            )

    # Writes — local only
    def put_recipe(
        self,
        *,
        canonical_id: str,
        model: str = "",
        hardware: str = "",
        framework_name: str = "",
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
            framework_name (str): Framework identity slot.
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
            framework_name=framework_name,
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

    # Reads — remote-first, local fallback
    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
        prefer: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Read a recipe row.

        Remote uses the single ``/recipes/search`` route: the 7-tuple decoded
        from ``canonical_id`` is passed as ``label_match`` and the central
        kb-service decides exact-vs-relative match + ranking.

        ``prefer`` (workload-similarity hints) does NOT change the ``required``
        (7-tuple) filter; it only reranks candidates so the closest-workload
        recipe is returned first. When set, the remote is asked for a small
        candidate window, reranked client-side, and the top is taken.

        Local is an arbor-style exact read of the on-disk ``recipe.json``. It
        serves ``version``-pinned reads (history archive) and the fall-through
        when remote is absent / empty / errors.

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
                # Fast path: delegate to the remote's slug-based get_recipe
                # rather than the expensive search scan.
                try:
                    direct = self.remote.get_recipe(  # type: ignore[union-attr]
                        canonical_id=canonical_id,
                        version=version,
                    )
                except (RemoteRecipeClientError, Exception):  # noqa: BLE001
                    direct = None
                if direct is not None and isinstance(direct, dict) and direct:
                    normalized = self._normalize_remote_row(direct)
                    if normalized and normalized.get("canonical_id"):
                        self._emit_audit(
                            self._read_audit_event(
                                method="get_recipe",
                                resolution="remote",
                                row=normalized,
                                canonical_id=canonical_id,
                                prefer=prefer,
                                candidates=1,
                            )
                        )
                        return normalized
                # Fast path miss — try label-match search with prefer rerank.
                labels = _labels_from_canonical_id(canonical_id)
                candidate_limit = 25 if prefer else 1
                rows = self.remote.search(  # type: ignore[union-attr]
                    label_match=labels,
                    limit=candidate_limit,
                    prefer=prefer,
                )
                if rows:
                    normalized_rows = [self._normalize_remote_row(r) for r in rows]
                    ranked = _rerank_by_prefer(normalized_rows, prefer)
                    self._emit_audit(
                        self._read_audit_event(
                            method="get_recipe",
                            resolution="remote",
                            row=ranked[0],
                            canonical_id=canonical_id,
                            prefer=prefer,
                            candidates=len(rows),
                        )
                    )
                    return ranked[0]
                # Retry without framework_version so version drift doesn't hide
                # reusable pages (only after exact + full label-match miss).
                relaxed_labels = dict(labels)
                relaxed_labels.pop("framework_version", None)
                rows = self.remote.search(  # type: ignore[union-attr]
                    label_match=relaxed_labels,
                    limit=candidate_limit,
                    prefer=prefer,
                )
                if rows:
                    normalized_rows = [self._normalize_remote_row(r) for r in rows]
                    ranked = _rerank_by_prefer(normalized_rows, prefer)
                    self._emit_audit(
                        self._read_audit_event(
                            method="get_recipe",
                            resolution="remote",
                            row=ranked[0],
                            canonical_id=canonical_id,
                            prefer=prefer,
                            label_match=relaxed_labels,
                            candidates=len(rows),
                        )
                    )
                    return ranked[0]
                resolution = "remote_miss"
            except RemoteRecipeClientError as exc:
                self._note_failure("get_recipe", exc)
                resolution = "remote_error"
            except InvalidCanonicalIdError as exc:
                log.warning("get_recipe: %s; local-only read", exc)
                resolution = "remote_error"
        local_row = self.local.get_recipe(
            canonical_id=canonical_id,
            version=version,
        )
        self._emit_audit(
            self._read_audit_event(
                method="get_recipe",
                resolution=resolution,
                row=local_row,
                canonical_id=canonical_id,
                prefer=prefer,
            )
        )
        return local_row

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

        Single-source-of-record per call (no merging across stores).
        Resolution:

        1. Remote returned a non-empty list → reranked by ``prefer`` and
           returned.
        2. Remote returned ``[]`` → fall through to local (an empty remote
           result is more often "hasn't ingested our recent writes yet").
        3. Remote raised → log + fall through.

        ``prefer`` only reorders; the ``required`` filter
        (``label_match`` / ``metric_filters``) decides membership.

        Args:
            label_match: Identity labels deciding membership.
            metric_filters: ``{metric: {min, max}}`` bounds deciding membership.
            updated_since: Lower bound on ``updated_at``.
            order_by: Ordering directive forwarded to the store.
            limit: Maximum number of rows to return.
            prefer: Workload-similarity hints used only to rerank results.

        Returns:
            The matching recipe rows (remote-first, local fall-through),
            reranked by ``prefer``.
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
                # Empty remote search → fall through to local (more often a
                # flaky/bootstrapping service than a truly empty corpus).
                if rows:
                    normalized = [self._normalize_remote_row(r) for r in rows]
                    ranked = _rerank_by_prefer(normalized, prefer)
                    self._emit_audit(
                        self._read_audit_event(
                            method="search",
                            resolution="remote",
                            row=ranked[0] if ranked else None,
                            label_match=label_match,
                            prefer=prefer,
                            candidates=len(rows),
                        )
                    )
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
        self._emit_audit(
            self._read_audit_event(
                method="search",
                resolution=resolution,
                row=ranked_local[0] if ranked_local else None,
                label_match=label_match,
                prefer=prefer,
            )
        )
        return ranked_local

    # Lifecycle
    def close(self) -> None:
        """Release the remote client's HTTP transport (no-op for the local
        store). Idempotent; safe to call from a CLI atexit handler.
        """
        if self.remote is not None:
            try:
                self.remote.close()
            except Exception:  # noqa: BLE001
                log.exception("recipe_kb: remote.close raised")


__all__ = [
    "RecipeKB",
]
