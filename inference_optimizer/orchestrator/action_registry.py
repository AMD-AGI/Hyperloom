"""ActionRegistry — DESIGN v0.6 §16.

Loads action metadata from ``actions/_meta/<name>.yaml`` (one file per
action). The corresponding markdown body at ``actions/<name>.md`` is
the agent-facing playbook (loaded lazily by SubAgentRunner / Coordinator
when composing a sub-agent prompt).

Used by:

* :class:`PolicyGate` — looks up ``allowed_tools`` for sub-agent dispatch
  and validates that proposed action names exist
* :class:`SubAgentRunner` — reads ``requires_lanes`` / ``lease_ttl_sec``
  / ``allowed_tools`` to gate sub-agent execution
* Budget-Aware Scheduler (P1+) — uses ``family`` / ``expected_gain_pct``
  / ``cost_minutes_p75`` / ``accuracy_risk`` / ``crash_risk`` for scoring

v0.6 schema (DESIGN §16.2)::

    name:                str  (required, must equal the filename stem)
    family:              one of {prep, analysis, shallow, deep_kernel,
                                 long, creative, resilience}
    cost_minutes_p50:    float
    cost_minutes_p75:    float
    expected_gain_pct:   [low, high]   # 2-element list of floats
    accuracy_risk:       float   # 0..1
    crash_risk:          float   # 0..1
    prerequisites:       list[str]
    requires_lanes:      list[str]
    allowed_tools:       list[str]
    side_effects:        list[str]
    preferred_backend:   "claude" | "codex"
    preferred_model:     str
    max_turns:           int
    lease_ttl_sec:       int
    applicable_when:     list[str]    # free-form predicates
    # v0.6.2 (Phase 1) prompt-builder additions:
    description:         str          # 1-line action brief consumed by
                                      # prompt_builder; defaults to name
                                      # when missing.
    pipeline_phase:      str          # one of VALID_PIPELINE_PHASES; used
                                      # by prompt_builder to group actions.
                                      # Defaults to "explore".
    typical_runtime_min: float        # display-only typical wallclock for
                                      # the action; defaults to
                                      # cost_minutes_p50.

v0.6 vs v0.5 schema:

* Removed ``allowed_modes`` (single full mode — DESIGN ADR-34)
* Family vocabulary unchanged (7 families)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..paths import asset_actions_dir


VALID_FAMILIES: frozenset[str] = frozenset({
    "prep", "analysis", "shallow", "deep_kernel",
    "long", "creative", "resilience",
})

VALID_BACKENDS: frozenset[str] = frozenset({"claude", "codex"})

# Phase 1 (prompt_builder) — coarse-grained pipeline phase used to group
# actions inside the Orchestration system prompt (e.g. "Run prep actions
# first, then move to explore..."). Kept intentionally small; map onto the
# DESIGN §11 timeline rather than the family taxonomy because family is
# scheduler-oriented (prep/analysis/...) while phases are LLM-oriented.
VALID_PIPELINE_PHASES: frozenset[str] = frozenset({
    "prep",        # setup / classify / target_analysis / baseline
    "measure",     # baseline (gates explore)
    "explore",     # backends / params / sweep
    "analysis",    # profile / pmc_roofline / deep_kernel_analysis
    "deep",        # kernel_opt / integrate / operator_tuning / vendor_kernel_config
    "validate",    # validate_stack — apply optimization_stack + rebench
    "finalize",    # report
    "support",     # dream / re_explore / recover / comm_optimization / compiler_tuning
})

_REQUIRED_FIELDS: tuple[str, ...] = (
    "name", "family", "cost_minutes_p50", "cost_minutes_p75",
    "expected_gain_pct", "accuracy_risk", "crash_risk",
)


class ActionRegistryError(RuntimeError):
    """Raised on schema or filesystem problems while loading actions."""


@dataclass(frozen=True)
class ActionMetadata:
    """Mirrors ``actions/_meta/<name>.yaml`` (DESIGN §16.2)."""

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
    # Phase 1 prompt-builder fields — see module docstring.
    description: str = ""
    pipeline_phase: str = "explore"
    typical_runtime_min: float = 0.0

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
        # Phase 1 prompt-builder fields. All optional; missing values fall back
        # to safe defaults so old yaml files keep parsing while new ones can
        # opt into richer prompts.
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
        )


class ActionRegistry:
    """In-memory registry of loaded action metadata.

    Construction is cheap; call :meth:`load` once at boot. ``load()``
    is idempotent and re-scans the meta directory each call.
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
]
