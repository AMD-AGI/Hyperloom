# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Arbor-aligned data shapes for the local recipe-snapshot KB.

The on-disk ``recipe.json`` layout follows the arbor ``Recipe`` dataclass so
an operator who knows arbor can read our local files without translation:

* ``model`` / ``hardware`` / ``best_config`` / ``best_throughput`` /
  ``stack_fingerprint`` / ``last_profiled`` / ``sessions`` / ``prs_tested``
  are all top-level fields.
* The four experience arrays use arbor's names — ``what_worked`` /
  ``what_failed`` / ``remaining_gaps`` / ``pitfalls``.
* Each row has the same nested sub-shapes arbor uses.

We keep a small superset of arbor fields:

* ``canonical_id`` / ``version`` / ``created_at`` / ``updated_at`` —
  store-managed metadata for the atomic-archive contract.
* ``framework_name`` / ``framework_version`` / ``precision`` — the three
  identity dimensions arbor lacks (arbor is a 2-tuple model+hardware).
* ``lessons`` / ``authority`` / ``confidence`` / ``evidence_refs`` /
  ``provenance`` — v2-spec fields the central kb-service expects.

Schema translation from the v2 wire shape to this arbor shape lives in
:mod:`recipe_kb.dispatcher`, applied on read. Writes are local-only and
never marshalled back to v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from .gbrain_ingest import _best_config_split, _coerce_server_args
except ImportError:  # pragma: no cover - defensive fallback

    def _coerce_server_args(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return " ".join(str(v).strip() for v in value if str(v).strip())
        return str(value)

    def _best_config_split(best_config: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
        args = _coerce_server_args(best_config.get("extra_server_args")).strip()
        nested = best_config.get("extra_envs")
        if not isinstance(nested, Mapping):
            nested = best_config.get("envs")
        if isinstance(nested, Mapping):
            envs = {str(k): str(v) for k, v in nested.items()}
        else:
            non_env_keys = {"extra_server_args", "extra_envs", "envs", "args", "name", "tput", "accuracy"}
            envs = {
                str(k): str(v)
                for k, v in best_config.items()
                if k not in non_env_keys and not isinstance(v, (Mapping, list, tuple))
            }
        return args, envs


def _normalize_best_config(best_config: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize a legacy ``{args, envs}`` best_config shape on read.

    Some producers (KG-derived warm-start candidates, older writers) emit
    ``args`` / ``envs`` (or a nested ``envs`` map) instead of the canonical
    ``extra_server_args`` / ``extra_envs`` keys. Unwrap that shape here so
    every in-memory ``Recipe`` uses the canonical keys; already-canonical
    dicts and any other unknown keys pass through unchanged.
    """
    if "args" not in best_config and "envs" not in best_config:
        return dict(best_config)
    remapped = dict(best_config)
    if "extra_server_args" not in remapped and "args" in remapped:
        remapped["extra_server_args"] = remapped["args"]
    envs_val = remapped.get("envs")
    if "extra_envs" not in remapped and isinstance(envs_val, Mapping):
        remapped["extra_envs"] = envs_val
    args, envs = _best_config_split(remapped)
    drop = {"args", "envs", *envs.keys()}
    out = {k: v for k, v in best_config.items() if k not in drop}
    if args:
        out.setdefault("extra_server_args", args)
    if envs:
        out.setdefault("extra_envs", envs)
    return out


# Arbor-aligned sub-shapes
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

    ``severity`` (e.g. ``crash`` / ``regress``) is a hyperloom superset field
    the Coordinator stamps so the warm-start prompt can rank pitfalls by
    disruption; arbor consumers ignore it.
    """

    description: str
    severity: str = ""


@dataclass
class Lesson:
    """A "X is the lesson" insight — statement + optional impact.

    We keep ``Lesson`` (arbor stores everything under ``what_worked``)
    because the v2 wire contract has a dedicated ``lessons`` array and the
    Coordinator emits these distinct from ``what_worked``. ``measured_impact``
    is free-form (a structured dict or plain string) and preserved verbatim.
    """

    statement: str
    measured_impact: Any = ""


@dataclass
class StackFingerprint:
    """Software-stack identity — used to detect "is this recipe stale?"

    Mirrors arbor's ``StackFingerprint`` exactly to keep binary readability
    between arbor and our local KB.
    """

    vllm_version: str = ""
    aiter_commit: str = ""
    rocm_version: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialise the fingerprint to a plain dict.

        Returns:
            dict[str, str]: A dict with keys ``vllm_version``,
                ``aiter_commit`` and ``rocm_version``.
        """
        return {
            "vllm_version": self.vllm_version,
            "aiter_commit": self.aiter_commit,
            "rocm_version": self.rocm_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StackFingerprint:
        """Build a fingerprint from a (possibly partial) dict.

        Missing keys default to empty strings.

        Args:
            d (dict[str, Any]): Source mapping; recognised keys are
                ``vllm_version`` / ``aiter_commit`` / ``rocm_version``.

        Returns:
            StackFingerprint: The reconstructed fingerprint.
        """
        return cls(
            vllm_version=str(d.get("vllm_version") or ""),
            aiter_commit=str(d.get("aiter_commit") or ""),
            rocm_version=str(d.get("rocm_version") or ""),
        )


@dataclass
class KernelOptimization:
    """One KEEP'd kernel-optimization outcome — micro result + E2E verdict.

    Hyperloom superset (arbor has no kernel-level concept). Captures a kernel
    KEEP'd at the micro layer plus its end-to-end integrate verification
    result. A kernel can be a genuine micro win yet show no end-to-end gain
    because its share of total GPU time is tiny — that conclusion is valuable
    warm-start signal.

    ``e2e_gain_pct`` / ``e2e_tput`` default to 0.0 and ``integrated`` to False
    for a kernel KEEP'd at the micro layer but never integrated (so warm-start
    can tell "micro-only, E2E unknown" apart from "E2E-verified, no gain").
    """

    kernel_id: str = ""
    source_file: str = ""
    artifact_path: str = ""
    micro_speedup: float = 0.0
    decision: str = ""
    e2e_gain_pct: float = 0.0
    e2e_tput: float = 0.0
    # Integrate-layer verdict (KEEP / REVERT / NEEDS_REVIEW); ``decision``
    # above stays the micro-layer KEEP.
    e2e_decision: str = ""
    integrated: bool = False
    ts: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise the kernel-optimization record to a plain dict.

        Returns:
            dict[str, Any]: All fields coerced to their JSON-friendly
                types (floats, ints, strings, bool).
        """
        return {
            "kernel_id": str(self.kernel_id),
            "source_file": str(self.source_file),
            "artifact_path": str(self.artifact_path),
            "micro_speedup": float(self.micro_speedup),
            "decision": str(self.decision),
            "e2e_gain_pct": float(self.e2e_gain_pct),
            "e2e_tput": float(self.e2e_tput),
            "e2e_decision": str(self.e2e_decision),
            "integrated": bool(self.integrated),
            "ts": str(self.ts),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KernelOptimization:
        """Build a kernel-optimization record from a dict.

        Missing keys fall back to the dataclass defaults; numeric and
        boolean fields are coerced.

        Args:
            d (dict[str, Any]): Source mapping matching the
                :meth:`to_dict` shape.

        Returns:
            KernelOptimization: The reconstructed record.
        """
        return cls(
            kernel_id=str(d.get("kernel_id") or ""),
            source_file=str(d.get("source_file") or ""),
            artifact_path=str(d.get("artifact_path") or ""),
            micro_speedup=float(d.get("micro_speedup") or 0.0),
            decision=str(d.get("decision") or ""),
            e2e_gain_pct=float(d.get("e2e_gain_pct") or 0.0),
            e2e_tput=float(d.get("e2e_tput") or 0.0),
            e2e_decision=str(d.get("e2e_decision") or ""),
            integrated=bool(d.get("integrated") or False),
            ts=str(d.get("ts") or ""),
        )


@dataclass
class SessionSummary:
    """One optimisation-session entry — one row per CLOSE.

    Superset of arbor's session log. ``session_id`` is required for
    per-session dedup on rewrite and ``gain_pct`` feeds warm-replay. arbor's
    ``date`` / ``throughput_before`` / ``throughput_after`` / ``actions_taken``
    stay available (all default to empty so a hyperloom-only write is valid).
    """

    date: str = ""
    throughput_before: float = 0.0
    throughput_after: float = 0.0
    actions_taken: list[str] = field(default_factory=list)
    session_id: str = ""
    gain_pct: float = 0.0
    stack_len: int = 0


# Recipe — arbor superset
@dataclass
class Recipe:
    """One on-disk recipe row, isomorphic to arbor's ``Recipe`` plus the
    version + provenance metadata our atomic-archive needs.

    The ``to_dict`` output is what lands in ``recipe.json``. An arbor consumer
    pointing ``ARBOR_RECIPES_DIR`` at our store sees only a few extra keys
    (``canonical_id`` / ``version`` / ``framework_version`` / ``precision`` /
    ``provenance``); everything else is byte-compatible.
    """

    # ----- store-managed metadata -----
    canonical_id: str
    version: int = 1
    created_at: str = ""
    updated_at: str = ""

    # ----- 5-tuple identity (arbor 2-tuple is model + hardware) -----
    model: str = ""
    hardware: str = ""
    framework_name: str = ""
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
    # Kernel-level wins + their E2E verification outcome. Warm-start reads it
    # to skip re-optimizing kernels already proven to have no E2E payoff.
    kernel_optimizations: list[KernelOptimization] = field(default_factory=list)

    # ----- v2 audit / wire-compat fields -----
    authority: str = "EXPERIENTIAL"
    confidence: float = 0.85
    evidence_refs: list[Any] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    # ----- free-form extras (unrecognised top-level keys, preserved verbatim) -----
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the recipe to the on-disk ``recipe.json`` shape.

        Nested sub-shapes (findings, failures, gaps, PRs, pitfalls,
        lessons, sessions, kernel optimizations, stack fingerprint)
        are expanded to plain dicts, and free-form ``extras`` are
        splatted at the top level without shadowing well-known keys.

        Returns:
            dict[str, Any]: The arbor-shape recipe row.
        """
        out: dict[str, Any] = {
            "canonical_id": self.canonical_id,
            "version": int(self.version),
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at),
            "model": str(self.model),
            "hardware": str(self.hardware),
            "framework_name": str(self.framework_name),
            "framework_version": str(self.framework_version),
            "precision": str(self.precision),
            "best_config": dict(self.best_config),
            "best_throughput": float(self.best_throughput),
            "what_worked": [
                {"description": f.description, "measured_impact": f.measured_impact} for f in self.what_worked
            ],
            "what_failed": [{"description": f.description, "reason": f.reason} for f in self.what_failed],
            "remaining_gaps": [{"description": g.description, "metrics": g.metrics} for g in self.remaining_gaps],
            "prs_tested": [
                {"repo": p.repo, "number": int(p.number), "outcome": p.outcome, "notes": p.notes}
                for p in self.prs_tested
            ],
            "pitfalls": [{"description": p.description, "severity": p.severity} for p in self.pitfalls],
            "lessons": [{"statement": l.statement, "measured_impact": l.measured_impact} for l in self.lessons],
            "last_profiled": str(self.last_profiled),
            "stack_fingerprint": self.stack_fingerprint.to_dict(),
            "sessions": [
                {
                    "date": s.date,
                    "throughput_before": float(s.throughput_before),
                    "throughput_after": float(s.throughput_after),
                    "actions_taken": list(s.actions_taken),
                    "session_id": s.session_id,
                    "gain_pct": float(s.gain_pct),
                    "stack_len": int(s.stack_len),
                }
                for s in self.sessions
            ],
            "kernel_optimizations": [k.to_dict() for k in self.kernel_optimizations],
            "authority": str(self.authority),
            "confidence": float(self.confidence),
            "evidence_refs": list(self.evidence_refs),
            "provenance": dict(self.provenance),
        }
        # Splat extras at the top level (no nested ``extras`` key on disk);
        # reserved keys above always win.
        for key, val in self.extras.items():
            if key not in out:
                out[key] = val
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Recipe:
        """Build a recipe from an on-disk / wire dict.

        Nested arrays are parsed back into their typed sub-shapes
        (skipping non-dict entries), and any unrecognised top-level
        keys are preserved verbatim in ``extras`` so a newer/older
        writer's data is never dropped.

        Args:
            d (dict[str, Any]): Source mapping in arbor recipe shape.

        Returns:
            Recipe: The reconstructed recipe row.
        """
        # Well-known top-level keys we parse; anything else goes into ``extras``.
        well_known = {
            "canonical_id",
            "version",
            "created_at",
            "updated_at",
            "model",
            "hardware",
            "framework_name",
            # Legacy framework-identity key; consumed into framework_name below,
            # listed here so it never leaks into extras.
            "framework",
            "framework_version",
            "precision",
            "best_config",
            "best_throughput",
            "what_worked",
            "what_failed",
            "remaining_gaps",
            "prs_tested",
            "pitfalls",
            "lessons",
            "last_profiled",
            "stack_fingerprint",
            "sessions",
            "kernel_optimizations",
            "authority",
            "confidence",
            "evidence_refs",
            "provenance",
            # Composite-Cortex provenance markers — dead weight in a local
            # recipe row, never persisted into extras.
            "_field_sources",
            "_sources",
        }
        extras = {k: v for k, v in d.items() if k not in well_known}
        return cls(
            canonical_id=str(d.get("canonical_id") or ""),
            version=int(d.get("version") or 1),
            created_at=str(d.get("created_at") or ""),
            updated_at=str(d.get("updated_at") or ""),
            model=str(d.get("model") or ""),
            hardware=str(d.get("hardware") or ""),
            # Fall back to the legacy ``framework`` key.
            framework_name=str(d.get("framework_name") or d.get("framework") or ""),
            framework_version=str(d.get("framework_version") or ""),
            precision=str(d.get("precision") or ""),
            best_config=_normalize_best_config(d.get("best_config") or {}),
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
                KernelOptimization.from_dict(k) for k in (d.get("kernel_optimizations") or []) if isinstance(k, dict)
            ],
            authority=str(d.get("authority") or "EXPERIENTIAL"),
            confidence=float(d.get("confidence") or 0.85),
            evidence_refs=list(d.get("evidence_refs") or []),
            provenance=dict(d.get("provenance") or {}),
            extras=extras,
        )


# Attempt — append-only optimization-attempt record
@dataclass
class Attempt:
    """One append-only evolutionary attempt against a recipe.

    Attempts are not arbor-defined; this layer is identical to the v2 wire
    ``Attempt`` shape, which the central server speaks on read too.
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
        """Serialise the attempt to a plain dict.

        ``fitness`` is omitted entirely when ``None`` (rather than
        emitted as null) to match the on-disk NDJSON contract.

        Returns:
            dict[str, Any]: The attempt row; includes ``fitness`` only
                when it is set.
        """
        out: dict[str, Any] = {
            "id": int(self.id),
            "recipe_canonical_id": str(self.recipe_canonical_id),
            "session_id": str(self.session_id),
            "attempt_at": str(self.attempt_at),
            "diff": dict(self.diff),
            "predicted_delta": dict(self.predicted_delta),
            "measured_metrics": dict(self.measured_metrics),
            "outcome": str(self.outcome),
            "rationale": str(self.rationale),
        }
        if self.fitness is not None:
            out["fitness"] = float(self.fitness)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Attempt:
        """Build an attempt from a dict.

        ``fitness`` stays ``None`` when absent; all other fields fall
        back to their dataclass defaults.

        Args:
            d (dict[str, Any]): Source mapping matching the
                :meth:`to_dict` shape.

        Returns:
            Attempt: The reconstructed attempt row.
        """
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
