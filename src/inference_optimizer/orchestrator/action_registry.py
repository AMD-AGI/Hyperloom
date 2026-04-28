"""ActionRegistry — DESIGN §12.

Loads action metadata from ``actions/_meta/<name>.yaml`` (one file per action)
plus the corresponding markdown body from ``actions/<name>.md``.

Used by:

    * :class:`PolicyGate` to validate ``allowed_modes`` and look up
      ``allowed_tools`` for sub-agent dispatch.
    * :class:`SubAgentRunner` (F3b) to compose the system prompt for a
      delegated Claude / Codex sub-agent.
    * :class:`BudgetAwareScheduler` (Phase 9) to score candidate actions.

YAML schema (subset enforced in v0.6):

    name:                 str  (required, must equal the filename stem)
    family:               one of {prep, analysis, shallow, deep_kernel,
                                  long, creative, resilience}
    cost_minutes_p50:     float
    cost_minutes_p75:     float
    expected_gain_pct:    [low, high]   # 2-element list of floats
    accuracy_risk:        float   # 0..1
    crash_risk:           float   # 0..1
    prerequisites:        list[str]
    requires_lanes:       list[str]
    allowed_tools:        list[str]
    side_effects:         list[str]
    allowed_modes:        list[str]    # subset of ExecutionMode values
    preferred_backend:    "claude" | "codex"
    preferred_model:      str
    max_turns:            int
    lease_ttl_sec:        int
    applicable_when:      list[str]    # free-form predicates
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .execution_mode import ExecutionMode


# ---------------------------------------------------------------------------
class ActionRegistryError(RuntimeError):
    """Raised on schema or filesystem problems while loading actions."""


# ---------------------------------------------------------------------------
_VALID_FAMILIES: frozenset[str] = frozenset(
    {"prep", "analysis", "shallow", "deep_kernel", "long", "creative", "resilience"}
)
_REQUIRED_FIELDS: tuple[str, ...] = (
    "name",
    "family",
    "cost_minutes_p50",
    "cost_minutes_p75",
    "expected_gain_pct",
    "accuracy_risk",
    "crash_risk",
    "allowed_modes",
)


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionMetadata:
    """Mirrors ``actions/_meta/<name>.yaml`` schema (DESIGN §12.2)."""

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
    allowed_modes: tuple[ExecutionMode, ...] = ()
    preferred_backend: str = "claude"
    preferred_model: str = "claude-opus-4-7"
    max_turns: int = 30
    lease_ttl_sec: int = 1800
    applicable_when: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
class ActionRegistry:
    """In-memory registry of loaded action metadata + markdown bodies.

    Construction is cheap; call :meth:`load` once before serving requests.
    Re-loading is supported for hot-reload during dev.
    """

    def __init__(self, actions_dir: Path) -> None:
        self.actions_dir = Path(actions_dir)
        self.meta_dir = self.actions_dir / "_meta"
        self._cache: dict[str, ActionMetadata] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------
    def load(self) -> "ActionRegistry":
        """Scan ``_meta/*.yaml``, validate, populate cache. Returns ``self``."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - tested via dependency
            raise ActionRegistryError(
                "PyYAML is required to load action metadata; "
                "pip install PyYAML"
            ) from exc

        if not self.meta_dir.is_dir():
            raise ActionRegistryError(
                f"actions meta directory not found: {self.meta_dir}"
            )

        cache: dict[str, ActionMetadata] = {}
        for yaml_path in sorted(self.meta_dir.glob("*.yaml")):
            try:
                raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise ActionRegistryError(
                    f"failed to parse YAML at {yaml_path}: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ActionRegistryError(
                    f"action file {yaml_path} must contain a mapping at top level"
                )
            try:
                meta = self._validate(raw, expected_name=yaml_path.stem)
            except ActionRegistryError as exc:
                raise ActionRegistryError(
                    f"{yaml_path}: {exc}"
                ) from exc
            if meta.name in cache:
                raise ActionRegistryError(
                    f"duplicate action name {meta.name!r} (file={yaml_path})"
                )
            cache[meta.name] = meta

        self._cache = cache
        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # Lookup API
    # ------------------------------------------------------------------
    def get(self, name: str) -> ActionMetadata | None:
        """Look up an action by name. Returns ``None`` when unknown.

        ``None`` is preferred over raising so PolicyGate can short-circuit
        unknown-action checks without an exception detour.
        """
        return self._cache.get(name)

    def all(self) -> tuple[ActionMetadata, ...]:
        return tuple(self._cache.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._cache.keys())

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._cache

    def __iter__(self):  # convenience for tests
        return iter(self._cache.values())

    def allowed_for_mode(
        self, mode: ExecutionMode | str
    ) -> tuple[ActionMetadata, ...]:
        """Return the subset of actions that declare the mode as supported."""
        if isinstance(mode, str):
            try:
                mode = ExecutionMode(mode)
            except ValueError as exc:
                raise ActionRegistryError(
                    f"unknown execution mode: {mode!r}"
                ) from exc
        return tuple(a for a in self._cache.values() if mode in a.allowed_modes)

    def system_prompt_for(self, name: str) -> str:
        """Return ``actions/<name>.md`` body, or '' if missing."""
        path = self.actions_dir / f"{name}.md"
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    def _validate(
        self, raw: dict[str, Any], *, expected_name: str
    ) -> ActionMetadata:
        for field in _REQUIRED_FIELDS:
            if field not in raw:
                raise ActionRegistryError(f"missing required field {field!r}")

        name = str(raw["name"])
        if name != expected_name:
            raise ActionRegistryError(
                f"action name {name!r} does not match filename stem {expected_name!r}"
            )

        family = str(raw["family"])
        if family not in _VALID_FAMILIES:
            raise ActionRegistryError(
                f"invalid family {family!r}; must be one of {sorted(_VALID_FAMILIES)}"
            )

        gain_raw = raw["expected_gain_pct"]
        if not isinstance(gain_raw, (list, tuple)) or len(gain_raw) != 2:
            raise ActionRegistryError(
                "expected_gain_pct must be a 2-element list of floats"
            )
        gain_low, gain_high = float(gain_raw[0]), float(gain_raw[1])

        modes_raw = raw["allowed_modes"]
        if not isinstance(modes_raw, (list, tuple)) or not modes_raw:
            raise ActionRegistryError(
                "allowed_modes must be a non-empty list of mode names"
            )
        modes: list[ExecutionMode] = []
        for m in modes_raw:
            try:
                modes.append(ExecutionMode(str(m)))
            except ValueError as exc:
                raise ActionRegistryError(
                    f"unknown execution mode in allowed_modes: {m!r}"
                ) from exc

        return ActionMetadata(
            name=name,
            family=family,
            cost_minutes_p50=float(raw["cost_minutes_p50"]),
            cost_minutes_p75=float(raw["cost_minutes_p75"]),
            expected_gain_pct=(gain_low, gain_high),
            accuracy_risk=float(raw["accuracy_risk"]),
            crash_risk=float(raw["crash_risk"]),
            prerequisites=_to_str_tuple(raw.get("prerequisites", ())),
            requires_lanes=_to_str_tuple(raw.get("requires_lanes", ())),
            allowed_tools=_to_str_tuple(raw.get("allowed_tools", ())),
            side_effects=_to_str_tuple(raw.get("side_effects", ())),
            allowed_modes=tuple(modes),
            preferred_backend=str(raw.get("preferred_backend", "claude")),
            preferred_model=str(raw.get("preferred_model", "claude-opus-4-7")),
            max_turns=int(raw.get("max_turns", 30)),
            lease_ttl_sec=int(raw.get("lease_ttl_sec", 1800)),
            applicable_when=_to_str_tuple(raw.get("applicable_when", ())),
        )


# ---------------------------------------------------------------------------
def _to_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    raise ActionRegistryError(
        f"expected list/tuple, got {type(value).__name__}: {value!r}"
    )


__all__ = [
    "ActionMetadata",
    "ActionRegistry",
    "ActionRegistryError",
]
