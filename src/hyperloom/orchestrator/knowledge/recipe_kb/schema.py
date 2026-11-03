# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Arbor-aligned data shapes for the local recipe-snapshot KB."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, TypeVar


_Row = TypeVar("_Row")


def _text(d: Mapping[str, Any], key: str) -> str:
    """Read one text field, mapping a missing or empty value to ``""``."""

    return str(d.get(key) or "")


def _rows(
    d: Mapping[str, Any],
    key: str,
    build: Callable[[Mapping[str, Any]], _Row],
) -> list[_Row]:
    """Read one list-valued field, building every dict-shaped row and dropping the rest."""

    return [build(row) for row in (d.get(key) or []) if isinstance(row, dict)]


def _coerce_server_args(value: Any) -> str:
    """Normalize launch arguments into one command-line string."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value)


def _best_config_split(
    best_config: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    """Split canonical launch arguments from environment variables."""

    args = _coerce_server_args(best_config.get("extra_server_args")).strip()
    nested = best_config.get("extra_envs")
    if not isinstance(nested, Mapping):
        nested = best_config.get("envs")
    if isinstance(nested, Mapping):
        envs = {str(k): str(v) for k, v in nested.items()}
    else:
        non_env_keys = {
            "extra_server_args",
            "extra_envs",
            "envs",
            "args",
            "name",
            "tput",
            "accuracy",
        }
        envs = {
            str(k): str(v)
            for k, v in best_config.items()
            if k not in non_env_keys and not isinstance(v, (Mapping, list, tuple))
        }
    return args, envs


def _normalize_best_config(best_config: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize a legacy ``{args, envs}`` best_config shape on read."""
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
def _experience_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep one ``what_worked`` / ``what_failed`` row whole, rejecting the two types its readers cannot use.

    The Coordinator owns the keys on these rows, so a row is stored as written
    rather than through a field list that a new key would have to be taught.
    ``extra_envs`` and ``gain_pct`` are checked when present because the warm-start
    readers index one as a mapping and grade on the other as a number; a wrong type
    there is worth a failed write rather than a row that persists with the value gone.
    """
    out = dict(row)
    envs = out.get("extra_envs")
    if envs is not None and not isinstance(envs, Mapping):
        raise TypeError(f"extra_envs must be a mapping, got {type(envs).__name__}")
    gain = out.get("gain_pct")
    if gain is not None and not isinstance(gain, (int, float)):
        raise TypeError(f"gain_pct must be a number, got {type(gain).__name__}")
    if not out.get("name"):
        # Only a row that actually stored a ``description`` can name its variant this way. Coordinator rows went
        # through a projection that wrote ``description: ""``, so theirs was already lost when it was written.
        legacy = out.get("description")
        if legacy:
            out["name"] = str(legacy)
    return out


@dataclass
class Gap:
    """A "we still don't know" — open question + relevant metrics."""

    description: str
    metrics: str

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Gap:
        """Build a gap from a (possibly partial) dict."""
        return cls(description=_text(d, "description"), metrics=_text(d, "metrics"))


@dataclass
class Pitfall:
    """A \"watch out for X\" — operator-readable description."""

    description: str
    severity: str = ""

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Pitfall:
        """Build a pitfall from a (possibly partial) dict."""
        return cls(description=_text(d, "description"), severity=_text(d, "severity"))


@dataclass
class Lesson:
    """A \"X is the lesson\" insight — statement + optional impact."""

    statement: str
    measured_impact: Any = ""

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Lesson:
        """Build a lesson from a (possibly partial) dict."""
        # ``measured_impact`` is free-form; keep it verbatim rather than coercing it to text.
        return cls(statement=_text(d, "statement"), measured_impact=d.get("measured_impact") or "")


@dataclass
class StackFingerprint:
    """Software-stack identity — used to detect "is this recipe stale?"."""

    vllm_version: str = ""
    aiter_commit: str = ""
    rocm_version: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialise the fingerprint to a plain dict."""
        return {
            "vllm_version": self.vllm_version,
            "aiter_commit": self.aiter_commit,
            "rocm_version": self.rocm_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> StackFingerprint:
        """Build a fingerprint from a (possibly partial) dict."""
        return cls(
            vllm_version=_text(d, "vllm_version"),
            aiter_commit=_text(d, "aiter_commit"),
            rocm_version=_text(d, "rocm_version"),
        )


@dataclass
class SessionSummary:
    """One optimisation-session entry — one row per CLOSE."""

    date: str = ""
    throughput_before: float = 0.0
    throughput_after: float = 0.0
    actions_taken: list[str] = field(default_factory=list)
    session_id: str = ""
    gain_pct: float = 0.0
    stack_len: int = 0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SessionSummary:
        """Build a session summary from a (possibly partial) dict."""
        return cls(
            date=_text(d, "date"),
            throughput_before=float(d.get("throughput_before") or 0.0),
            throughput_after=float(d.get("throughput_after") or 0.0),
            actions_taken=list(d.get("actions_taken") or []),
            session_id=_text(d, "session_id"),
            gain_pct=float(d.get("gain_pct") or 0.0),
            stack_len=int(d.get("stack_len") or 0),
        )


# Recipe — arbor superset
@dataclass
class Recipe:
    """One on-disk recipe row, isomorphic to arbor's ``Recipe`` plus the version + provenance metadata our atomic-archive needs."""

    # ----- store-managed metadata -----
    canonical_id: str
    version: int = 1
    created_at: str = ""
    updated_at: str = ""

    # ----- 7-tuple identity -----
    model: str = ""
    hardware: str = ""
    framework_name: str = ""
    framework_version: str = ""
    precision: str = ""

    # ----- arbor payload (verbatim shape, except the experience columns below, which are stored as the producer wrote them) -----
    best_config: dict[str, str] = field(default_factory=dict)
    best_throughput: float = 0.0
    what_worked: list[dict[str, Any]] = field(default_factory=list)
    what_failed: list[dict[str, Any]] = field(default_factory=list)
    remaining_gaps: list[Gap] = field(default_factory=list)
    pitfalls: list[Pitfall] = field(default_factory=list)
    lessons: list[Lesson] = field(default_factory=list)
    last_profiled: str = ""
    stack_fingerprint: StackFingerprint = field(default_factory=StackFingerprint)
    sessions: list[SessionSummary] = field(default_factory=list)

    # ----- v2 audit / wire-compat fields -----
    authority: str = "EXPERIENTIAL"
    confidence: float = 0.85
    evidence_refs: list[Any] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    # ----- free-form extras (unrecognised top-level keys, preserved verbatim) -----
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the recipe to the on-disk ``recipe.json`` shape."""
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
            "what_worked": [dict(f) for f in self.what_worked],
            "what_failed": [dict(f) for f in self.what_failed],
            "remaining_gaps": [{"description": g.description, "metrics": g.metrics} for g in self.remaining_gaps],
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
            "authority": str(self.authority),
            "confidence": float(self.confidence),
            "evidence_refs": list(self.evidence_refs),
            "provenance": dict(self.provenance),
        }
        # Splat extras at the top level (no nested ``extras`` key on disk); reserved keys above always win.
        for key, val in self.extras.items():
            if key not in out:
                out[key] = val
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Recipe:
        """Build a recipe from an on-disk / wire dict."""
        # Well-known top-level keys we parse; anything else goes into ``extras``.
        well_known = {
            "canonical_id",
            "version",
            "created_at",
            "updated_at",
            "model",
            "hardware",
            "framework_name",
            # Legacy framework-identity key; consumed into framework_name below, listed here so it never leaks into
            # extras.
            "framework",
            "framework_version",
            "precision",
            "best_config",
            "best_throughput",
            "what_worked",
            "what_failed",
            "remaining_gaps",
            "pitfalls",
            "lessons",
            "last_profiled",
            "stack_fingerprint",
            "sessions",
            "authority",
            "confidence",
            "evidence_refs",
            "provenance",
            # Composite-KB provenance markers — dead weight in a local recipe row, never persisted into extras.
            "_field_sources",
            "_sources",
            "prs_tested",
        }
        extras = {k: v for k, v in d.items() if k not in well_known}
        return cls(
            canonical_id=_text(d, "canonical_id"),
            version=int(d.get("version") or 1),
            created_at=_text(d, "created_at"),
            updated_at=_text(d, "updated_at"),
            model=_text(d, "model"),
            hardware=_text(d, "hardware"),
            # Fall back to the legacy ``framework`` key.
            framework_name=_text(d, "framework_name") or _text(d, "framework"),
            framework_version=_text(d, "framework_version"),
            precision=_text(d, "precision"),
            best_config=_normalize_best_config(d.get("best_config") or {}),
            best_throughput=float(d.get("best_throughput") or 0.0),
            what_worked=_rows(d, "what_worked", _experience_row),
            what_failed=_rows(d, "what_failed", _experience_row),
            remaining_gaps=_rows(d, "remaining_gaps", Gap.from_dict),
            pitfalls=_rows(d, "pitfalls", Pitfall.from_dict),
            lessons=_rows(d, "lessons", Lesson.from_dict),
            last_profiled=_text(d, "last_profiled"),
            stack_fingerprint=StackFingerprint.from_dict(
                d.get("stack_fingerprint") or {},
            ),
            sessions=_rows(d, "sessions", SessionSummary.from_dict),
            authority=_text(d, "authority") or "EXPERIENTIAL",
            confidence=float(d.get("confidence") or 0.85),
            evidence_refs=list(d.get("evidence_refs") or []),
            provenance=dict(d.get("provenance") or {}),
            extras=extras,
        )


__all__ = [
    "Gap",
    "Lesson",
    "Pitfall",
    "Recipe",
    "SessionSummary",
    "StackFingerprint",
]
