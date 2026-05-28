"""Data shapes for the local recipe-snapshot KB store.

These dataclasses mirror the wire contract documented in
``primus-cortex-internal/docs/recipe-snapshot-api-reference.md``
(v2). The local store reads / writes plain dicts on disk, so the
classes here are mostly normalisation helpers — they round-trip
through ``to_dict`` / ``from_dict`` and never escape the local
store's API surface (callers see plain dicts identical to the
central kb-service responses).

Why dataclasses at all (vs. raw dicts)?

* Round-trip safety — :meth:`Recipe.to_dict` always emits the same
  superset of keys, so two recipes written by different code paths
  produce comparable JSON on disk (avoids the ``set(a.keys()) ^
  set(b.keys())`` trap when introducing a new optional field).
* Default values — every optional field gets a documented zero
  value, so a Pydantic-rejected input from an older Commit-0 ingest
  doesn't crash a Commit-2 read by missing a key.
* Type hints — IDE / mypy can flag a typo'd ``cononical_id`` at
  edit time instead of at the v1-history regression test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Recipe — the full v2 row shape
# ---------------------------------------------------------------------------
# Matches the documented response shape in the API reference §
# "Recipe (response shape)". Server-managed fields (``version``,
# ``created_at``, ``updated_at``) are populated by the local store
# during put_recipe; callers MUST NOT set them. ``canonical_id`` is
# also store-stamped (derived from the put_recipe argument) but kept
# in the dataclass for symmetry with the wire shape.
@dataclass
class Recipe:
    """Full local recipe row, isomorphic to the central wire shape."""

    canonical_id: str
    version: int = 1
    labels: dict[str, Any] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[Any] = field(default_factory=list)
    failures: list[Any] = field(default_factory=list)
    pitfalls: list[Any] = field(default_factory=list)
    lessons: list[Any] = field(default_factory=list)
    gaps: list[Any] = field(default_factory=list)
    authority: str = "EXPERIENTIAL"
    confidence: float = 0.85
    evidence_refs: list[Any] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Render to a JSON-serialisable dict.

        Field order matches the wire shape documented in the API
        reference so a quick visual diff between a local row and a
        central GET response is straightforward.
        """
        return {
            "canonical_id":  self.canonical_id,
            "version":       int(self.version),
            "labels":        dict(self.labels),
            "body":          dict(self.body),
            "metrics":       dict(self.metrics),
            "findings":      list(self.findings),
            "failures":      list(self.failures),
            "pitfalls":      list(self.pitfalls),
            "lessons":       list(self.lessons),
            "gaps":          list(self.gaps),
            "authority":     str(self.authority),
            "confidence":    float(self.confidence),
            "evidence_refs": list(self.evidence_refs),
            "provenance":    dict(self.provenance),
            "created_at":    str(self.created_at),
            "updated_at":    str(self.updated_at),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Recipe:
        """Construct from a JSON dict.

        Defensive on every field — an old archive (history/v1.json
        from a pre-Commit-2 release) might be missing keys the new
        code expects. The default values applied here keep a stale
        archive readable, even if downstream consumers should be
        defensive about empty collections.
        """
        return cls(
            canonical_id=str(d.get("canonical_id", "")),
            version=int(d.get("version", 1)),
            labels=dict(d.get("labels") or {}),
            body=dict(d.get("body") or {}),
            metrics=dict(d.get("metrics") or {}),
            findings=list(d.get("findings") or []),
            failures=list(d.get("failures") or []),
            pitfalls=list(d.get("pitfalls") or []),
            lessons=list(d.get("lessons") or []),
            gaps=list(d.get("gaps") or []),
            authority=str(d.get("authority") or "EXPERIENTIAL"),
            confidence=float(d.get("confidence") or 0.85),
            evidence_refs=list(d.get("evidence_refs") or []),
            provenance=dict(d.get("provenance") or {}),
            created_at=str(d.get("created_at") or ""),
            updated_at=str(d.get("updated_at") or ""),
        )


# ---------------------------------------------------------------------------
# Attempt — append-only optimization-attempt record
# ---------------------------------------------------------------------------
# Matches the documented POST /recipes/{cid}/attempts request shape
# AND the GET /recipes/{cid}/attempts response shape (the local store
# stamps ``id`` + ``recipe_canonical_id`` + ``attempt_at`` on
# append, mirroring server-side stamping).
@dataclass
class Attempt:
    """One append-only evolutionary attempt against a recipe."""

    id: int = 0
    recipe_canonical_id: str = ""
    session_id: str = ""
    attempt_at: str = ""
    diff: dict[str, Any] = field(default_factory=dict)
    predicted_delta: dict[str, Any] = field(default_factory=dict)
    measured_metrics: dict[str, Any] = field(default_factory=dict)
    fitness: float | None = None
    outcome: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id":                  int(self.id),
            "recipe_canonical_id": str(self.recipe_canonical_id),
            "session_id":          str(self.session_id),
            "attempt_at":          str(self.attempt_at),
            "diff":                dict(self.diff),
            "predicted_delta":     dict(self.predicted_delta),
            "measured_metrics":    dict(self.measured_metrics),
            "outcome":             str(self.outcome),
            "rationale":           str(self.rationale),
        }
        # ``fitness`` is optional on the wire — emit only when set so
        # JSON readers can distinguish "not measured" from
        # "measured zero".
        if self.fitness is not None:
            out["fitness"] = float(self.fitness)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Attempt:
        fitness = d.get("fitness")
        return cls(
            id=int(d.get("id", 0)),
            recipe_canonical_id=str(d.get("recipe_canonical_id") or ""),
            session_id=str(d.get("session_id") or ""),
            attempt_at=str(d.get("attempt_at") or ""),
            diff=dict(d.get("diff") or {}),
            predicted_delta=dict(d.get("predicted_delta") or {}),
            measured_metrics=dict(d.get("measured_metrics") or {}),
            fitness=float(fitness) if fitness is not None else None,
            outcome=str(d.get("outcome") or ""),
            rationale=str(d.get("rationale") or ""),
        )


__all__ = [
    "Attempt",
    "Recipe",
]
