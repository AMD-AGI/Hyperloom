# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist dispatch profile — the three orthogonal dials that parameterise a single ``specialist`` worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hyperloom.common.env import is_truthy

if TYPE_CHECKING:
    from .domains import SpecialistDomain


# scope
SCOPE_DOMAIN = "domain"
SCOPE_DOMAINS = "domains"
SCOPE_FREEFORM = "freeform"
SCOPE_VALUES: frozenset[str] = frozenset({SCOPE_DOMAIN, SCOPE_DOMAINS, SCOPE_FREEFORM})

# mode
MODE_RESEARCH = "research"
MODE_PATCH = "patch"
MODE_VALUES: frozenset[str] = frozenset({MODE_RESEARCH, MODE_PATCH})

# Defaults: an anchored dispatch resolves to single-domain, patch-authoring behaviour; a truly bare
# dispatch is inferred ``freeform`` and resolves to the cheap read-only research mode.
DEFAULT_SCOPE = SCOPE_DOMAIN
DEFAULT_MODE = MODE_PATCH
DEFAULT_BENCH = False


@dataclass(frozen=True)
class SpecialistProfile:
    """Resolved dispatch dials for one specialist task."""

    scope: str = DEFAULT_SCOPE
    mode: str = DEFAULT_MODE
    bench: bool = DEFAULT_BENCH

    @property
    def is_freeform(self) -> bool:
        """Whether this profile uses the free-form (unscoped) scope."""
        return self.scope == SCOPE_FREEFORM

    @property
    def reserves_benchmark_lane(self) -> bool:
        """True iff this dispatch should contend for the ``benchmark_lane``."""
        return self.mode == MODE_PATCH and self.bench


def _infer_scope(p: dict[str, Any]) -> str:
    """Infer the dispatch scope when none is explicitly given."""
    # Local import avoids a module-load cycle.
    from .domains import normalize_dispatch_tags

    tags = normalize_dispatch_tags(p)
    if len(tags) >= 2:
        return SCOPE_DOMAINS
    if tags:
        return SCOPE_DOMAIN
    return SCOPE_FREEFORM


def _whole_machine_available() -> bool:
    """Whether this single node exposes any card to a whole-machine lease."""
    from ..actions.executors._multi_node_env import is_multi_node
    from ..bus.gpu_pool import resolve_whole_machine_devices

    return not is_multi_node() and bool(resolve_whole_machine_devices())


def requires_gpu(params: dict[str, Any] | None) -> bool:
    """True for ``needs_gpu``, bench-capable, and enablement specialists (the latter boot a server)."""
    p = params or {}
    if is_truthy(p.get("needs_gpu")) or resolve_specialist_profile(p).reserves_benchmark_lane:
        return True
    return bool(p.get("enablement")) and _whole_machine_available()


def specialist_lanes(params: dict[str, Any] | None, base_lanes: list[str]) -> list[str]:
    """Lanes for a specialist: GPU specialists hold ``gpu_research_lane`` instead of ``research_lane``."""
    lanes = list(base_lanes)
    if requires_gpu(params):
        lanes = [lane for lane in lanes if lane != "research_lane"] + ["gpu_research_lane"]
    if resolve_specialist_profile(params).reserves_benchmark_lane:
        lanes.append("benchmark_lane")
    return list(dict.fromkeys(lanes))


def wall_budget_base_min(params: dict[str, Any] | None) -> float:
    """Base wall-clock budget in minutes: 60 for patch mode, 10 for research mode."""
    return 60.0 if resolve_specialist_profile(params).mode == MODE_PATCH else 10.0


def is_authoring_specialist(params: dict[str, Any] | None) -> bool:
    """True for a FRAMEWORK or ENABLEMENT authoring specialist."""
    p = params or {}
    return bool(p.get("framework_agent_authoring")) or bool(p.get("enablement"))


def uses_whole_machine_gpu_lane(params: dict[str, Any] | None) -> bool:
    """True when a GPU specialist should lease the *whole machine* (time-shared with serving via ``gpu_research_lane``) rather than the serving-disjoint ``gpu_specialist_pool``."""
    p = params or {}
    return bool(p.get("enablement")) or resolve_specialist_profile(p).reserves_benchmark_lane


def holds_serving_slot(params: dict[str, Any] | None) -> bool:
    """True when a GPU specialist must hold the whole-machine ``serving_slot`` Ray resource (mutually exclusive with production serving)."""
    return resolve_specialist_profile(params or {}).reserves_benchmark_lane


def resolve_specialist_profile(
    params: dict[str, Any] | None,
    domain: "SpecialistDomain | None" = None,
) -> SpecialistProfile:
    """Resolve scope/mode/bench from dispatch params, falling back to safe defaults."""
    p = params or {}

    scope = str(p.get("scope") or "").strip().lower()
    if scope not in SCOPE_VALUES:
        scope = _infer_scope(p)

    mode = str(p.get("mode") or "").strip().lower()
    if mode not in MODE_VALUES:
        domain_default = str(getattr(domain, "default_mode", "") or "").strip().lower()
        if domain_default in MODE_VALUES:
            mode = domain_default
        else:
            mode = MODE_RESEARCH if scope == SCOPE_FREEFORM else DEFAULT_MODE

    bench = is_truthy(p.get("bench"), default=DEFAULT_BENCH)
    if mode != MODE_PATCH:
        bench = False

    return SpecialistProfile(scope=scope, mode=mode, bench=bench)


__all__ = [
    "DEFAULT_BENCH",
    "DEFAULT_MODE",
    "DEFAULT_SCOPE",
    "MODE_PATCH",
    "MODE_RESEARCH",
    "MODE_VALUES",
    "SCOPE_DOMAIN",
    "SCOPE_DOMAINS",
    "SCOPE_FREEFORM",
    "SCOPE_VALUES",
    "SpecialistProfile",
    "holds_serving_slot",
    "is_authoring_specialist",
    "requires_gpu",
    "resolve_specialist_profile",
    "specialist_lanes",
    "uses_whole_machine_gpu_lane",
    "wall_budget_base_min",
]
