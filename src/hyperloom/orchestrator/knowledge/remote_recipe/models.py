# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small, dependency-free models and validation for Remote Recipe KB V2."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

MAX_KNOWLEDGE_BYTES = 5 * 1024 * 1024
MAX_FILES = 512
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_PATH_BYTES = 1024

#: The columns a published Recipe carries, in the order a reader applies them.
#: A column name is also the prefix every artifact it owns lives under, so this
#: is the one place the on-the-wire names are spelled.
CONFIG_SECTION = "config"
PATCH_SECTION = "patch"
KERNEL_SECTION = "kernel"
RECIPE_SECTIONS: tuple[str, ...] = (CONFIG_SECTION, PATCH_SECTION, KERNEL_SECTION)

_ARTIFACT_REF_KEYS: frozenset[str] = frozenset(
    {
        "artifact_files",
        "artifact_path",
        "artifacts",
        "changed_files",
        "experience_document",
        "files",
        "final_report_path",
        "patch",
        "patch_path",
        "patches",
        "patches_applied",
        "report_path",
        "source_file",
        "source_files",
        "target_file",
        "target_files",
        "tuned_file",
    }
)
#: Only these columns own artifacts; ``config`` is pure data.
_BUILDER_REF_PREFIXES: tuple[str, ...] = (
    f"{KERNEL_SECTION}/",
    f"{PATCH_SECTION}/",
)


class RemoteRecipeValidationError(ValueError):
    """A locally-built remote recipe violates the KB Store contract."""


def _scope_int(value: Any) -> int | None:
    """Normalize an integer scope value echoed through a URL query."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


@dataclass(frozen=True)
class RecipeScope:
    """The partition a Recipe belongs to in the KB Store."""

    kernel_optimizer: str
    tp: int
    conc: int
    # AgentX trace replay omits the fixed-length InferenceX dimensions.
    isl: int | None = None
    osl: int | None = None

    @classmethod
    def from_state(cls, state: Any) -> "RecipeScope":
        """Build the scope this session writes to and reads from."""
        optimizer = str(getattr(state, "kernel_optimizer", "") or "").strip().lower()
        # CLI bootstrap records an explicitly enabled Forge backend as "native"; KB Store uses the public backend name
        # "forge".
        backend = "forge" if optimizer in {"native", "forge", "kernel_agent_forge"} else optimizer
        from hyperloom.common.perf_metric import agentx_active

        is_agentx = agentx_active(
            benchmark_mode=getattr(state, "benchmark_mode", "")
        )
        scope = cls(
            kernel_optimizer=backend,
            tp=int(getattr(state, "tp", 0) or 0),
            conc=int(getattr(state, "conc", 0) or 0),
            isl=None if is_agentx else int(getattr(state, "isl", 0) or 0),
            osl=None if is_agentx else int(getattr(state, "osl", 0) or 0),
        )
        scope.validate()
        return scope

    def validate(self) -> None:
        """Reject a scope the KB Store cannot partition on."""
        if self.kernel_optimizer not in {"forge", "geak"}:
            raise RemoteRecipeValidationError(f"unsupported kernel_optimizer: {self.kernel_optimizer!r}")
        if min(self.tp, self.conc) <= 0:
            raise RemoteRecipeValidationError("Recipe scope tp/conc must be positive")
        if (self.isl is None) != (self.osl is None):
            raise RemoteRecipeValidationError("Recipe scope isl/osl must be present together")
        if self.isl is not None and min(self.isl, self.osl or 0) <= 0:
            raise RemoteRecipeValidationError("Recipe scope isl/osl must be positive")

    def as_dict(self) -> dict[str, Any]:
        """Return the scope as the Store's query / payload mapping."""
        scope = {
            "kernel_optimizer": self.kernel_optimizer,
            "tp": self.tp,
            "conc": self.conc,
        }
        if self.isl is not None:
            scope.update({"isl": self.isl, "osl": self.osl})
        return scope

    @property
    def identity_scheme(self) -> str:
        """The KB identity scheme whose scope contract this represents."""
        return "agentx" if self.isl is None else "inference"

    def matches(self, value: Any) -> bool:
        """True when a View's recorded scope is exactly this one."""
        expected = self.as_dict()
        comparable = (
            {key: item for key, item in value.items() if key != "scope_schema"}
            if isinstance(value, dict)
            else value
        )
        return (
            isinstance(comparable, dict)
            and set(comparable) == set(expected)
            and comparable.get("kernel_optimizer") == self.kernel_optimizer
            and self.matches_workload_shape(comparable)
        )

    def matches_workload_shape(self, value: Any) -> bool:
        """True when workload dimensions match, accepting URL string echoes."""
        return isinstance(value, dict) and all(
            _scope_int(value.get(key)) == expected
            for key, expected in self.as_dict().items()
            if key != "kernel_optimizer"
        )


@dataclass(frozen=True)
class KBSelectionProfile:
    """Mode-specific scope and metric values for the shared KB client."""

    scope: RecipeScope
    primary_metric: str
    primary_value: float
    objective_schema: str
    metrics: dict[str, float]

    @classmethod
    def from_state(
        cls,
        state: Any,
        *,
        scope: RecipeScope | None = None,
    ) -> "KBSelectionProfile":
        from hyperloom.common.gain_math import gain_pct
        from hyperloom.common.perf_metric import agentx_active

        scope = scope or RecipeScope.from_state(state)
        current = getattr(state, "current_best", {}) or {}
        if not isinstance(current, dict):
            raise RemoteRecipeValidationError("current_best must be a mapping")
        if agentx_active(benchmark_mode=getattr(state, "benchmark_mode", "")):
            baseline = getattr(state, "baseline_perf", {}) or {}
            baseline_i = float(baseline.get("e2e_norm_intvty_p90") or 0.0)
            candidate_i = float(current.get("e2e_norm_intvty_p90") or 0.0)
            total = float(current.get("total_throughput") or 0.0)
            gain = gain_pct(candidate_i, baseline_i)
            if gain is None or not math.isfinite(gain) or total <= 0:
                raise RemoteRecipeValidationError(
                    "AgentX write needs baseline-relative interactivity gain "
                    "and positive total_throughput"
                )
            return cls(
                scope=scope,
                primary_metric="interactivity_gain_pct",
                primary_value=float(gain),
                objective_schema="agentx_keep",
                metrics={
                    "interactivity_gain_pct": float(gain),
                    "total_throughput": total,
                    "baseline_interactivity": baseline_i,
                    "candidate_interactivity": candidate_i,
                    "baseline_total_throughput": float(
                        baseline.get("total_throughput") or 0.0
                    ),
                },
            )
        throughput = float(current.get("tput") or 0.0)
        if not math.isfinite(throughput) or throughput <= 0:
            raise RemoteRecipeValidationError(
                "InferenceX write needs positive optimized_throughput"
            )
        return cls(
            scope=scope,
            primary_metric="optimized_throughput",
            primary_value=throughput,
            objective_schema="single_throughput",
            metrics={"optimized_throughput": throughput},
        )


def validate_relative_path(value: str) -> str:
    """Return a normalized KB artifact path, rejecting traversal and ``files/``."""
    raw = str(value or "")
    if not raw or "\\" in raw or raw.startswith("/"):
        raise RemoteRecipeValidationError(f"invalid artifact path: {raw!r}")
    if len(raw.encode("utf-8")) > MAX_PATH_BYTES:
        raise RemoteRecipeValidationError(f"artifact path exceeds the {MAX_PATH_BYTES}-byte KB Store limit")
    path = PurePosixPath(raw)
    if any(part in ("", ".", "..") for part in path.parts):
        raise RemoteRecipeValidationError(f"invalid artifact path: {raw!r}")
    normalized = path.as_posix()
    if normalized == "files" or normalized.startswith("files/"):
        raise RemoteRecipeValidationError("knowledge paths must not include the bundle files/ prefix")
    return normalized


def extract_knowledge_artifact_refs(
    knowledge: Any,
    artifact_paths: Iterable[str] = (),
) -> set[str]:
    """Collect artifact refs actually present in the final knowledge document."""
    known_paths = {validate_relative_path(path) for path in artifact_paths if str(path or "").strip()}
    refs: set[str] = set()

    def declared_ref(raw: str, key: str) -> str | None:
        if key not in _ARTIFACT_REF_KEYS or "/" not in raw:
            return None
        try:
            normalized = validate_relative_path(raw)
        except RemoteRecipeValidationError:
            return None
        if key == "files" or normalized.startswith(_BUILDER_REF_PREFIXES):
            return normalized
        return None

    def visit(value: Any, *, key: str = "") -> None:
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                visit(nested_value, key=str(nested_key))
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item, key=key)
            return
        if not isinstance(value, str):
            return
        raw = value.strip()
        if not raw or "://" in raw:
            return
        if raw in known_paths:
            refs.add(validate_relative_path(raw))
            return
        declared = declared_ref(raw, key)
        if declared is not None:
            refs.add(declared)

    visit(knowledge)
    return refs


@dataclass(frozen=True)
class Artifact:
    """One local file and the relative path used in knowledge."""

    path: str
    source: Path
    kind: str = "other"
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        validate_relative_path(self.path)
        if self.source.is_symlink():
            raise RemoteRecipeValidationError(f"artifact source must not be a symlink: {self.source}")
        if not self.source.is_file():
            raise RemoteRecipeValidationError(f"artifact source is not a file: {self.source}")
        size = self.source.stat().st_size
        if size > MAX_FILE_BYTES:
            raise RemoteRecipeValidationError(
                f"artifact {self.source} is {size} bytes; KB Store limit is {MAX_FILE_BYTES}"
            )


@dataclass
class KnowledgeBundle:
    """Opaque knowledge plus the exact local files referenced by it."""

    knowledge: dict[str, Any]
    artifacts: list[Artifact] = field(default_factory=list)

    def validate(self) -> None:
        try:
            encoded = json.dumps(
                self.knowledge,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise RemoteRecipeValidationError(f"knowledge is not strict JSON: {exc}") from exc
        if len(encoded) > MAX_KNOWLEDGE_BYTES:
            raise RemoteRecipeValidationError(
                f"knowledge is {len(encoded)} bytes; KB Store limit is {MAX_KNOWLEDGE_BYTES}"
            )
        if len(self.artifacts) > MAX_FILES:
            raise RemoteRecipeValidationError(f"artifact count {len(self.artifacts)} exceeds {MAX_FILES}")
        paths: set[str] = set()
        for artifact in self.artifacts:
            artifact.validate()
            normalized = validate_relative_path(artifact.path)
            if normalized in paths:
                raise RemoteRecipeValidationError(f"duplicate artifact path: {normalized}")
            paths.add(normalized)
        refs = extract_knowledge_artifact_refs(self.knowledge, paths)
        missing = refs - paths
        unreferenced = paths - refs
        if missing:
            raise RemoteRecipeValidationError(f"knowledge references missing artifacts: {sorted(missing)!r}")
        if unreferenced:
            raise RemoteRecipeValidationError(f"uploaded artifacts absent from knowledge: {sorted(unreferenced)!r}")


@dataclass(frozen=True)
class RemoteWriteResult:
    """Outcome of the optional CLOSE-time remote write."""

    status: str
    reason: str = ""
    canonical_id: str = ""
    session_id: str = ""
    # Compatibility field; AgentX stores its primary score here.
    optimized_throughput: float = 0.0


__all__ = [
    "Artifact",
    "CONFIG_SECTION",
    "KERNEL_SECTION",
    "KnowledgeBundle",
    "KBSelectionProfile",
    "MAX_FILE_BYTES",
    "MAX_FILES",
    "MAX_KNOWLEDGE_BYTES",
    "MAX_PATH_BYTES",
    "PATCH_SECTION",
    "RECIPE_SECTIONS",
    "RemoteRecipeValidationError",
    "RemoteWriteResult",
    "RecipeScope",
    "extract_knowledge_artifact_refs",
    "validate_relative_path",
]
