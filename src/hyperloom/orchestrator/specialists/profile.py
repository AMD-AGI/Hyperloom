# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist dispatch profile — the four orthogonal dials that parameterise a single ``specialist`` worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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

# lane
LANE_CPU = "cpu"
LANE_GPU = "gpu"
LANE_VALUES: frozenset[str] = frozenset({LANE_CPU, LANE_GPU})


# Defaults: an anchored dispatch resolves to single-domain, patch-authoring, GPU-leased behaviour; a truly bare
# dispatch is inferred ``freeform`` and resolves to the cheap read-only research/CPU lane.
DEFAULT_SCOPE = SCOPE_DOMAIN
DEFAULT_MODE = MODE_PATCH
DEFAULT_BENCH = False
DEFAULT_LANE = LANE_GPU


@dataclass(frozen=True)
class SpecialistProfile:
    """Resolved dispatch dials for one specialist task."""

    scope: str = DEFAULT_SCOPE
    mode: str = DEFAULT_MODE
    bench: bool = DEFAULT_BENCH
    lane: str = DEFAULT_LANE

    @property
    def is_freeform(self) -> bool:
        """Whether this profile uses the free-form (unscoped) scope."""
        return self.scope == SCOPE_FREEFORM

    @property
    def reserves_benchmark_lane(self) -> bool:
        """True iff this dispatch should contend for the ``benchmark_lane``."""
        return self.mode == MODE_PATCH and self.bench


def _coerce_bool(value: Any, default: bool) -> bool:
    """Coerce a loosely-typed value to a boolean."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


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


def uses_whole_machine_gpu_lane(params: dict[str, Any] | None) -> bool:
    """True when a GPU specialist should lease the *whole machine* (time-shared with serving via ``gpu_research_lane``) rather than the serving-disjoint ``gpu_specialist_pool``."""
    p = params or {}
    if bool(p.get("framework_agent_authoring")):
        return True
    return resolve_specialist_profile(p).reserves_benchmark_lane


def holds_serving_slot(params: dict[str, Any] | None) -> bool:
    """True when a GPU specialist must hold the whole-machine ``serving_slot`` Ray resource (mutually exclusive with production serving)."""
    return resolve_specialist_profile(params or {}).reserves_benchmark_lane


def resolve_specialist_profile(
    params: dict[str, Any] | None,
    domain: "SpecialistDomain | None" = None,
) -> SpecialistProfile:
    """Resolve scope/mode/bench/lane from dispatch params, falling back to safe defaults."""
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

    bench = _coerce_bool(p.get("bench"), DEFAULT_BENCH)
    if mode != MODE_PATCH:
        bench = False

    lane = str(p.get("lane") or "").strip().lower()
    if lane not in LANE_VALUES:
        lane = LANE_GPU if mode == MODE_PATCH else LANE_CPU

    return SpecialistProfile(scope=scope, mode=mode, bench=bench, lane=lane)


__all__ = [
    "DEFAULT_BENCH",
    "DEFAULT_LANE",
    "DEFAULT_MODE",
    "DEFAULT_SCOPE",
    "LANE_CPU",
    "LANE_GPU",
    "LANE_VALUES",
    "MODE_PATCH",
    "MODE_RESEARCH",
    "MODE_VALUES",
    "SCOPE_DOMAIN",
    "SCOPE_DOMAINS",
    "SCOPE_FREEFORM",
    "SCOPE_VALUES",
    "SpecialistProfile",
    "holds_serving_slot",
    "resolve_specialist_profile",
    "uses_whole_machine_gpu_lane",
]
