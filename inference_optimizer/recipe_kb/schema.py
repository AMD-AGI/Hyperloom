"""Arbor-aligned data shapes for the local recipe-snapshot KB.

The on-disk JSON layout for ``recipe.json`` follows the arbor
``Recipe`` dataclass (see ``Arbor/src/arbor/recipes.py``) so an
operator who knows arbor can open one of our local files and read
it without translation. Specifically:

* ``model`` / ``hardware`` / ``best_config`` / ``best_throughput``
  / ``stack_fingerprint`` / ``last_profiled`` / ``sessions`` /
  ``prs_tested`` are all top-level fields, not buried inside
  ``body`` like the v2 wire spec puts them.
* The four experience arrays use arbor's names —
  ``what_worked`` / ``what_failed`` / ``remaining_gaps`` /
  ``pitfalls`` (NOT v2's ``findings`` / ``failures`` / ``gaps``).
* Each row has the same nested sub-shapes arbor uses
  (``Finding{description, measured_impact}`` etc.).

We keep a small superset of arbor fields:

* ``canonical_id`` / ``version`` / ``created_at`` /
  ``updated_at`` — store-managed metadata for the atomic-archive
  contract (arbor has no version concept; we need it for
  ``history/v{N}.json`` rollbacks).
* ``framework`` / ``framework_version`` / ``precision`` — the
  three identity dimensions arbor doesn't have (it's a 2-tuple
  ``model``+``hardware`` while we're a 5-tuple).
* ``lessons`` / ``authority`` / ``confidence`` / ``evidence_refs`` /
  ``provenance`` — v2-spec fields the central kb-service expects.
  The dispatcher uses them to round-trip rows back to ``/v1`` /
  audit.

Schema translation from the v2 wire shape (central kb-service) to
this arbor shape lives in :mod:`recipe_kb.dispatcher`
(``_v2_to_arbor``), applied on read. There is intentionally no
reverse ``_arbor_to_v2``: writes are local-only and are never
marshalled back to the v2 wire shape. The local store NEVER speaks
v2 directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Arbor-aligned sub-shapes
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    """An "X helped" insight — what worked + the measured impact."""
    description: str
    measured_impact: str


@dataclass
class Failure:
    """An "X didn't help" insight — what failed + the reason."""
    description: str
    reason: str


@dataclass
class Gap:
    """A "we still don't know" — open question + relevant metrics."""
    description: str
    metrics: str


@dataclass
class PRResult:
    """An upstream PR that was tried during the optimisation run."""
    repo: str
    number: int
    outcome: str
    notes: str = ""


@dataclass
class Pitfall:
    """A "watch out for X" — operator-readable description.

    ``severity`` (e.g. ``crash`` / ``regress``) is a hyperloom superset
    field the Coordinator stamps so the warm-start prompt can rank
    pitfalls by disruption; arbor consumers simply ignore it.
    """
    description: str
    severity: str = ""


@dataclass
class Lesson:
    """A "X is the lesson" insight — statement + optional impact.

    arbor has no separate ``Lesson`` shape (it stores everything
    under ``what_worked``); we keep ``Lesson`` because the v2 wire
    contract has a dedicated ``lessons`` array and the optimizer's
    Coordinator emits these distinct from ``what_worked``.

    ``measured_impact`` is free-form: the Coordinator writes a
    structured dict (``gain_pct`` / ``throughput_after`` / ...), but a
    plain string is also accepted for arbor-compat — it is preserved
    verbatim (never str()-ed) so the structured payload round-trips.
    """
    statement: str
    measured_impact: Any = ""


@dataclass
class StackFingerprint:
    """Software-stack identity — used to detect "is this recipe stale?"

    Mirrors arbor's ``StackFingerprint`` exactly; we don't add new
    fields here even though sglang / atom would benefit, because
    the goal is binary readability between arbor and our local KB.
    Operators who need extra-stack info should put it in
    ``stack_fingerprint_extras`` (a free-form sibling field is
    coordinated separately).
    """
    vllm_version: str = ""
    aiter_commit: str = ""
    rocm_version: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "vllm_version": self.vllm_version,
            "aiter_commit": self.aiter_commit,
            "rocm_version": self.rocm_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StackFingerprint:
        return cls(
            vllm_version=str(d.get("vllm_version") or ""),
            aiter_commit=str(d.get("aiter_commit") or ""),
            rocm_version=str(d.get("rocm_version") or ""),
        )


@dataclass
class KernelOptimization:
    """One KEEP'd kernel-optimization outcome — micro result + E2E verdict.

    Hyperloom superset (arbor has no kernel-level concept). Captures a
    kernel the GEAK/kernel agent produced and KEEP'd at the micro layer,
    together with the end-to-end integrate verification result when the
    optimizer actually integrated + re-benchmarked it. The point is that a
    kernel can be a genuine micro win (``micro_speedup`` > 1) yet show no
    end-to-end gain (``e2e_gain_pct`` ~ 0) because its share of total GPU
    time is tiny — that whole conclusion is valuable warm-start signal
    ("tried k006 rmsnorm: 1.32x micro, but E2E flat — skip it"), and used
    to be dropped on the floor because ``what_worked`` only records
    E2E-validated stack entries.

    ``e2e_gain_pct`` / ``e2e_tput`` default to 0.0 and ``integrated``
    to False for a kernel that was KEEP'd at the micro layer but never
    integrated (so warm-start can tell "micro-only, E2E unknown" apart
    from "E2E-verified, no gain").
    """
    kernel_id: str = ""
    source_file: str = ""
    artifact_path: str = ""
    micro_speedup: float = 0.0
    decision: str = ""
    e2e_gain_pct: float = 0.0
    e2e_tput: float = 0.0
    integrated: bool = False
    ts: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_id":     str(self.kernel_id),
            "source_file":   str(self.source_file),
            "artifact_path": str(self.artifact_path),
            "micro_speedup": float(self.micro_speedup),
            "decision":      str(self.decision),
            "e2e_gain_pct":  float(self.e2e_gain_pct),
            "e2e_tput":      float(self.e2e_tput),
            "integrated":    bool(self.integrated),
            "ts":            str(self.ts),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KernelOptimization:
        return cls(
            kernel_id=str(d.get("kernel_id") or ""),
            source_file=str(d.get("source_file") or ""),
            artifact_path=str(d.get("artifact_path") or ""),
            micro_speedup=float(d.get("micro_speedup") or 0.0),
            decision=str(d.get("decision") or ""),
            e2e_gain_pct=float(d.get("e2e_gain_pct") or 0.0),
            e2e_tput=float(d.get("e2e_tput") or 0.0),
            integrated=bool(d.get("integrated") or False),
            ts=str(d.get("ts") or ""),
        )


@dataclass
class SessionSummary:
    """One optimisation-session entry — one row per CLOSE.

    Superset of arbor's session log. The Coordinator's CLOSE finalize
    writes ``session_id`` / ``gain_pct`` / ``stack_len`` — ``session_id``
    is REQUIRED for the per-session dedup on rewrite (two finalises of
    the same session must not double-append) and ``gain_pct`` feeds
    warm-replay. arbor's ``date`` / ``throughput_before`` /
    ``throughput_after`` / ``actions_taken`` stay available for arbor
    consumers (all default to empty so a hyperloom-only write is valid).
    """
    date: str = ""
    throughput_before: float = 0.0
    throughput_after: float = 0.0
    actions_taken: list[str] = field(default_factory=list)
    session_id: str = ""
    gain_pct: float = 0.0
    stack_len: int = 0


# ---------------------------------------------------------------------------
# Recipe — arbor superset
# ---------------------------------------------------------------------------
@dataclass
class Recipe:
    """One on-disk recipe row, isomorphic to arbor's ``Recipe`` plus
    the version + provenance metadata our atomic-archive needs.

    The ``to_dict`` output is what lands in ``recipe.json``. An
    arbor consumer pointing ``ARBOR_RECIPES_DIR`` at our store
    would only see two extra keys (``canonical_id`` / ``version``
    / ``framework_version`` / ``precision`` / ``provenance``) —
    everything else is byte-compatible.
    """

    # ----- store-managed metadata -----
    canonical_id: str
    version: int = 1
    created_at: str = ""
    updated_at: str = ""

    # ----- 5-tuple identity (arbor 2-tuple is model + hardware) -----
    model: str = ""
    hardware: str = ""
    framework: str = ""
    framework_version: str = ""
    precision: str = ""

    # ----- arbor payload (verbatim shape) -----
    best_config: dict[str, str] = field(default_factory=dict)
    best_throughput: float = 0.0
    what_worked: list[Finding] = field(default_factory=list)
    what_failed: list[Failure] = field(default_factory=list)
    remaining_gaps: list[Gap] = field(default_factory=list)
    prs_tested: list[PRResult] = field(default_factory=list)
    pitfalls: list[Pitfall] = field(default_factory=list)
    lessons: list[Lesson] = field(default_factory=list)
    last_profiled: str = ""
    stack_fingerprint: StackFingerprint = field(default_factory=StackFingerprint)
    sessions: list[SessionSummary] = field(default_factory=list)

    # ----- hyperloom superset: KEEP'd kernel optimizations -----
    # Records kernel-level wins (micro_speedup) + their end-to-end
    # verification outcome. arbor consumers ignore the extra top-level
    # key (same as framework_version / lessons). Warm-start reads it to
    # skip re-optimizing kernels already proven to have no E2E payoff.
    kernel_optimizations: list[KernelOptimization] = field(default_factory=list)

    # ----- v2 audit / wire-compat fields (kept so dispatcher can
    # round-trip to the central server) -----
    authority: str = "EXPERIENTIAL"
    confidence: float = 0.85
    evidence_refs: list[Any] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    # ----- free-form extras (forward-compat: arbor's existing
    # recipes carry e.g. ``session_20260515_findings`` keys that
    # ``Recipe.from_dict`` doesn't parse but should preserve) -----
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "canonical_id":      self.canonical_id,
            "version":           int(self.version),
            "created_at":        str(self.created_at),
            "updated_at":        str(self.updated_at),
            "model":             str(self.model),
            "hardware":          str(self.hardware),
            "framework":         str(self.framework),
            "framework_version": str(self.framework_version),
            "precision":         str(self.precision),
            "best_config":       dict(self.best_config),
            "best_throughput":   float(self.best_throughput),
            "what_worked": [
                {"description": f.description,
                 "measured_impact": f.measured_impact}
                for f in self.what_worked
            ],
            "what_failed": [
                {"description": f.description, "reason": f.reason}
                for f in self.what_failed
            ],
            "remaining_gaps": [
                {"description": g.description, "metrics": g.metrics}
                for g in self.remaining_gaps
            ],
            "prs_tested": [
                {"repo": p.repo, "number": int(p.number),
                 "outcome": p.outcome, "notes": p.notes}
                for p in self.prs_tested
            ],
            "pitfalls": [
                {"description": p.description, "severity": p.severity}
                for p in self.pitfalls
            ],
            "lessons": [
                {"statement": l.statement,
                 "measured_impact": l.measured_impact}
                for l in self.lessons
            ],
            "last_profiled":     str(self.last_profiled),
            "stack_fingerprint": self.stack_fingerprint.to_dict(),
            "sessions": [
                {"date": s.date,
                 "throughput_before": float(s.throughput_before),
                 "throughput_after": float(s.throughput_after),
                 "actions_taken": list(s.actions_taken),
                 "session_id": s.session_id,
                 "gain_pct": float(s.gain_pct),
                 "stack_len": int(s.stack_len)}
                for s in self.sessions
            ],
            "kernel_optimizations": [
                k.to_dict() for k in self.kernel_optimizations
            ],
            "authority":     str(self.authority),
            "confidence":    float(self.confidence),
            "evidence_refs": list(self.evidence_refs),
            "provenance":    dict(self.provenance),
        }
        # Splat the free-form extras at the top level so they look
        # exactly the way arbor stores them (no nested ``extras`` key
        # on disk). Reserved keys above always win — we don't let an
        # arbitrary extras entry shadow a well-known field.
        for key, val in self.extras.items():
            if key not in out:
                out[key] = val
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Recipe:
        # Stripped set of well-known top-level keys we parse — anything
        # else gets bucketed into ``extras`` so a future read by a
        # newer (or older) writer doesn't lose data.
        well_known = {
            "canonical_id", "version", "created_at", "updated_at",
            "model", "hardware", "framework", "framework_version",
            "precision",
            "best_config", "best_throughput",
            "what_worked", "what_failed", "remaining_gaps",
            "prs_tested", "pitfalls", "lessons",
            "last_profiled", "stack_fingerprint", "sessions",
            "kernel_optimizations",
            "authority", "confidence", "evidence_refs", "provenance",
        }
        extras = {k: v for k, v in d.items() if k not in well_known}
        return cls(
            canonical_id=str(d.get("canonical_id") or ""),
            version=int(d.get("version") or 1),
            created_at=str(d.get("created_at") or ""),
            updated_at=str(d.get("updated_at") or ""),
            model=str(d.get("model") or ""),
            hardware=str(d.get("hardware") or ""),
            framework=str(d.get("framework") or ""),
            framework_version=str(d.get("framework_version") or ""),
            precision=str(d.get("precision") or ""),
            best_config=dict(d.get("best_config") or {}),
            best_throughput=float(d.get("best_throughput") or 0.0),
            what_worked=[
                Finding(
                    description=str(f.get("description") or ""),
                    measured_impact=str(f.get("measured_impact") or ""),
                )
                for f in (d.get("what_worked") or [])
                if isinstance(f, dict)
            ],
            what_failed=[
                Failure(
                    description=str(f.get("description") or ""),
                    reason=str(f.get("reason") or ""),
                )
                for f in (d.get("what_failed") or [])
                if isinstance(f, dict)
            ],
            remaining_gaps=[
                Gap(
                    description=str(g.get("description") or ""),
                    metrics=str(g.get("metrics") or ""),
                )
                for g in (d.get("remaining_gaps") or [])
                if isinstance(g, dict)
            ],
            prs_tested=[
                PRResult(
                    repo=str(p.get("repo") or ""),
                    number=int(p.get("number") or 0),
                    outcome=str(p.get("outcome") or ""),
                    notes=str(p.get("notes") or ""),
                )
                for p in (d.get("prs_tested") or [])
                if isinstance(p, dict)
            ],
            pitfalls=[
                Pitfall(
                    description=str(p.get("description") or ""),
                    severity=str(p.get("severity") or ""),
                )
                for p in (d.get("pitfalls") or [])
                if isinstance(p, dict)
            ],
            lessons=[
                Lesson(
                    statement=str(l.get("statement") or ""),
                    measured_impact=l.get("measured_impact") or "",
                )
                for l in (d.get("lessons") or [])
                if isinstance(l, dict)
            ],
            last_profiled=str(d.get("last_profiled") or ""),
            stack_fingerprint=StackFingerprint.from_dict(
                d.get("stack_fingerprint") or {},
            ),
            sessions=[
                SessionSummary(
                    date=str(s.get("date") or ""),
                    throughput_before=float(s.get("throughput_before") or 0.0),
                    throughput_after=float(s.get("throughput_after") or 0.0),
                    actions_taken=list(s.get("actions_taken") or []),
                    session_id=str(s.get("session_id") or ""),
                    gain_pct=float(s.get("gain_pct") or 0.0),
                    stack_len=int(s.get("stack_len") or 0),
                )
                for s in (d.get("sessions") or [])
                if isinstance(s, dict)
            ],
            kernel_optimizations=[
                KernelOptimization.from_dict(k)
                for k in (d.get("kernel_optimizations") or [])
                if isinstance(k, dict)
            ],
            authority=str(d.get("authority") or "EXPERIENTIAL"),
            confidence=float(d.get("confidence") or 0.85),
            evidence_refs=list(d.get("evidence_refs") or []),
            provenance=dict(d.get("provenance") or {}),
            extras=extras,
        )


# ---------------------------------------------------------------------------
# Attempt — append-only optimization-attempt record
# ---------------------------------------------------------------------------
@dataclass
class Attempt:
    """One append-only evolutionary attempt against a recipe.

    Same shape as before — attempts are not arbor-defined; this
    layer is identical to the v2 wire ``Attempt`` shape because
    the central server speaks the same shape on read-side too.
    """

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
        if self.fitness is not None:
            out["fitness"] = float(self.fitness)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Attempt:
        fitness = d.get("fitness")
        return cls(
            id=int(d.get("id") or 0),
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
    "Failure",
    "Finding",
    "Gap",
    "KernelOptimization",
    "Lesson",
    "Pitfall",
    "PRResult",
    "Recipe",
    "SessionSummary",
    "StackFingerprint",
]
