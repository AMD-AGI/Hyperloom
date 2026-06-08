# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""ActionRegistry

Loads action metadata from ``actions/_meta/<name>.yaml`` (one file per
action); the markdown body at ``actions/<name>.md`` is loaded lazily.

Operational fields gate execution/dispatch (``allowed_tools``,
``requires_lanes``, ``lease_ttl_sec``, ``preferred_backend`` /
``preferred_model``, ``max_turns``, ``side_effects``); all other fields are
prompt-advisory only (KB_design §3.9 Inv-9.1 enforced by construction here).

Schema (DESIGN §16.2)::

    name:                str  (required, must equal the filename stem)
    family:              one of {prep, analysis, shallow, deep_kernel,
                                 long, creative, resilience}
    cost_minutes_p50:    float    # prompt-advisory
    cost_minutes_p75:    float    # prompt-advisory
    expected_gain_pct:   [low, high]   # prompt-advisory prior, NOT a sort key
    accuracy_risk:       float    # 0..1, prompt-advisory
    crash_risk:          float    # 0..1, prompt-advisory
    prerequisites:       list[str]    # prompt-advisory ordering hint
    requires_lanes:      list[str]    # operational
    allowed_tools:       list[str]    # operational
    side_effects:        list[str]    # operational
    preferred_backend:   "claude" | "codex"   # operational
    preferred_model:     str          # operational
    max_turns:           int          # operational
    lease_ttl_sec:       int          # operational
    applicable_when:     list[str]    # prompt-advisory predicate list
    description:         str          # prompt rendering; defaults to name.
    pipeline_phase:      str          # one of VALID_PIPELINE_PHASES;
                                      # prompt grouping only.
    typical_runtime_min: float        # display-only; defaults to
                                      # cost_minutes_p50.
    verdict_class:       "archival" | "exploration" | "promotion"
                                      # routes Critic prompt rule set;
                                      # never a hidden gate (P3_20).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..paths import asset_actions_dir


# ``family`` is a prompt grouping label only; no runtime scheduler uses it.
VALID_FAMILIES: frozenset[str] = frozenset({
    "prep", "analysis", "shallow", "deep_kernel",
    "long", "creative", "resilience",
})

VALID_BACKENDS: frozenset[str] = frozenset({"claude", "codex"})

# Coarse-grained pipeline phase for prompt_builder grouping; prompt-advisory
# only (the real state machine lives in :mod:`phase_state`).
VALID_PIPELINE_PHASES: frozenset[str] = frozenset({
    "prep",        # target_analysis / baseline / warm replay
    "measure",     # baseline (gates explore)
    "explore",     # explore / specialists / patch integration
    "analysis",    # profile / roofline / deep_kernel_analysis
    "deep",        # kernel_opt / integrate / operator_tuning / vendor_kernel_config
    "validate",    # reserved; stack validation is inlined into explore
    "finalize",    # report
    "support",     # recover (the rest were retired)
})

# Per-action verdict policy class — selects which Critic prompt rule set
# applies; never a hidden hard gate (loosen P3_20).
VALID_VERDICT_CLASSES: frozenset[str] = frozenset({
    "archival", "exploration", "promotion",
})

# Default classifier for live actions; yaml ``verdict_class`` overrides.
# Unknown names fall back to ``"exploration"`` (safest non-deadlocking default).
_DEFAULT_VERDICT_CLASS: dict[str, str] = {
    # archival — transcribe state, no new measurement
    "report":                  "archival",
    "session_breakdown":       "archival",
    "target_analysis":         "archival",
    # promotion — mutate optimization_stack + claim gain
    "integrate":               "promotion",
    # exploration — everything else (run benchmarks to generate data)
    "baseline":                "exploration",
    "roofline":                "exploration",
    "sweep":                   "exploration",
    "conc_sweep":              "exploration",
    "kernel_opt":              "exploration",
    "gemm_tuning":             "exploration",
    "operator_tuning":         "exploration",
    "vendor_kernel_config":    "exploration",
    "deep_kernel_analysis":    "exploration",
    "recover":                 "exploration",
}
_DEFAULT_VERDICT_CLASS_FALLBACK: str = "exploration"


def default_verdict_class_for(action_name: str) -> str:
    """Look up the default ``verdict_class``; falls back to ``"exploration"``."""
    return _DEFAULT_VERDICT_CLASS.get(
        action_name, _DEFAULT_VERDICT_CLASS_FALLBACK,
    )


_REQUIRED_FIELDS: tuple[str, ...] = (
    "name", "family", "cost_minutes_p50", "cost_minutes_p75",
    "expected_gain_pct", "accuracy_risk", "crash_risk",
)


class ActionRegistryError(RuntimeError):
    """Raised on schema or filesystem problems while loading actions."""


@dataclass(frozen=True)
class ActionMetadata:
    """Mirrors ``actions/_meta/<name>.yaml`` (DESIGN §16.2).

    Operational fields gate execution; every other field is prompt-advisory
    only. See module docstring.
    """

    name: str
    family: str
    cost_minutes_p50: float
    cost_minutes_p75: float
    expected_gain_pct: tuple[float, float]
    accuracy_risk: float
    crash_risk: float
    prerequisites: tuple[str, ...] = ()
    requires_lanes: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()
    preferred_backend: str = "claude"
    preferred_model: str = "claude-opus-4-7"
    max_turns: int = 30
    lease_ttl_sec: int = 1800
    applicable_when: tuple[str, ...] = ()
    description: str = ""
    pipeline_phase: str = "explore"
    typical_runtime_min: float = 0.0
    # Routes Critic prompt rules only, never a hidden hard gate (loosen P3_20).
    verdict_class: str = ""

    @classmethod
    def from_yaml_dict(cls, data: dict[str, Any], expected_name: str) -> "ActionMetadata":
        for field_name in _REQUIRED_FIELDS:
            if field_name not in data:
                raise ActionRegistryError(
                    f"action {expected_name!r}: missing required field {field_name!r}"
                )
        if data["name"] != expected_name:
            raise ActionRegistryError(
                f"action filename stem {expected_name!r} does not match "
                f"yaml field name={data['name']!r}"
            )
        if data["family"] not in VALID_FAMILIES:
            raise ActionRegistryError(
                f"action {expected_name!r}: family {data['family']!r} not in "
                f"{sorted(VALID_FAMILIES)!r}"
            )
        gain = data["expected_gain_pct"]
        if not (isinstance(gain, (list, tuple)) and len(gain) == 2):
            raise ActionRegistryError(
                f"action {expected_name!r}: expected_gain_pct must be [low, high]"
            )
        for ratio_field in ("accuracy_risk", "crash_risk"):
            v = float(data[ratio_field])
            if not (0.0 <= v <= 1.0):
                raise ActionRegistryError(
                    f"action {expected_name!r}: {ratio_field}={v} not in 0..1"
                )
        backend = str(data.get("preferred_backend", "claude"))
        if backend not in VALID_BACKENDS:
            raise ActionRegistryError(
                f"action {expected_name!r}: preferred_backend={backend!r} not in "
                f"{sorted(VALID_BACKENDS)!r}"
            )
        # prompt-builder fields — all optional with safe defaults.
        cost_p50 = float(data["cost_minutes_p50"])
        description = str(data.get("description", "")).strip()
        pipeline_phase = str(data.get("pipeline_phase", "explore")).strip() or "explore"
        if pipeline_phase not in VALID_PIPELINE_PHASES:
            raise ActionRegistryError(
                f"action {expected_name!r}: pipeline_phase={pipeline_phase!r} not in "
                f"{sorted(VALID_PIPELINE_PHASES)!r}"
            )
        typical_runtime_min_raw = data.get("typical_runtime_min")
        try:
            typical_runtime_min = (
                float(typical_runtime_min_raw)
                if typical_runtime_min_raw is not None
                else cost_p50
            )
        except (TypeError, ValueError) as exc:
            raise ActionRegistryError(
                f"action {expected_name!r}: typical_runtime_min must be a "
                f"number, got {typical_runtime_min_raw!r}"
            ) from exc
        if typical_runtime_min < 0:
            raise ActionRegistryError(
                f"action {expected_name!r}: typical_runtime_min must be >= 0, "
                f"got {typical_runtime_min}"
            )
        # verdict_class — yaml override wins, else table default; validated against allowlist.
        verdict_class = str(
            data.get("verdict_class") or "",
        ).strip().lower()
        if not verdict_class:
            verdict_class = default_verdict_class_for(expected_name)
        if verdict_class not in VALID_VERDICT_CLASSES:
            raise ActionRegistryError(
                f"action {expected_name!r}: verdict_class={verdict_class!r} "
                f"not in {sorted(VALID_VERDICT_CLASSES)!r}"
            )
        return cls(
            name=str(data["name"]),
            family=str(data["family"]),
            cost_minutes_p50=cost_p50,
            cost_minutes_p75=float(data["cost_minutes_p75"]),
            expected_gain_pct=(float(gain[0]), float(gain[1])),
            accuracy_risk=float(data["accuracy_risk"]),
            crash_risk=float(data["crash_risk"]),
            prerequisites=tuple(data.get("prerequisites") or ()),
            requires_lanes=tuple(data.get("requires_lanes") or ()),
            allowed_tools=tuple(data.get("allowed_tools") or ()),
            side_effects=tuple(data.get("side_effects") or ()),
            preferred_backend=backend,
            preferred_model=str(data.get("preferred_model", "claude-opus-4-7")),
            max_turns=int(data.get("max_turns", 30)),
            lease_ttl_sec=int(data.get("lease_ttl_sec", 1800)),
            applicable_when=tuple(data.get("applicable_when") or ()),
            description=description or str(data["name"]),
            pipeline_phase=pipeline_phase,
            typical_runtime_min=typical_runtime_min,
            verdict_class=verdict_class,
        )


class ActionRegistry:
    """In-memory registry of loaded action metadata.

    Call :meth:`load` once at boot; it is idempotent and re-scans each call.
    """

    def __init__(self, actions_dir: Path | None = None) -> None:
        self.actions_dir = Path(actions_dir) if actions_dir else asset_actions_dir()
        self.meta_dir = self.actions_dir / "_meta"
        self._cache: dict[str, ActionMetadata] = {}
        self._loaded = False

    def load(self) -> "ActionRegistry":
        """Scan ``_meta/*.yaml``, validate, populate cache. Returns self."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ActionRegistryError(
                "PyYAML is required to load action metadata; pip install PyYAML"
            ) from exc

        if not self.meta_dir.is_dir():
            raise ActionRegistryError(
                f"actions meta directory not found: {self.meta_dir}"
            )

        cache: dict[str, ActionMetadata] = {}
        for path in sorted(self.meta_dir.glob("*.yaml")):
            stem = path.stem
            if stem.startswith("_"):
                continue
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            cache[stem] = ActionMetadata.from_yaml_dict(data, expected_name=stem)
        self._cache = cache
        self._loaded = True
        return self

    def get(self, name: str) -> ActionMetadata | None:
        if not self._loaded:
            self.load()
        return self._cache.get(name)

    def all(self) -> list[ActionMetadata]:
        if not self._loaded:
            self.load()
        return list(self._cache.values())

    def names(self) -> list[str]:
        if not self._loaded:
            self.load()
        return sorted(self._cache.keys())

    def by_family(self, family: str) -> list[ActionMetadata]:
        if family not in VALID_FAMILIES:
            raise ActionRegistryError(
                f"family={family!r} not in {sorted(VALID_FAMILIES)!r}"
            )
        return [a for a in self.all() if a.family == family]


__all__ = [
    "ActionMetadata",
    "ActionRegistry",
    "ActionRegistryError",
    "VALID_BACKENDS",
    "VALID_FAMILIES",
    "VALID_PIPELINE_PHASES",
    "VALID_VERDICT_CLASSES",
    "default_verdict_class_for",
]
