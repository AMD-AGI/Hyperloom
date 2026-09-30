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


def requires_gpu(params: dict[str, Any] | None) -> bool:
    """True when a specialist needs a GPU lease.

    Covers explicit ``needs_gpu``, bench-capable dispatches, and
    enablement authoring (which boots a server to validate the patch).
    FRAMEWORK authoring runs CPU-only; its integration benchmark goes
    through ``integrate_patch``.
    """
    p = params or {}
    if is_truthy(p.get("needs_gpu")):
        return True
    if resolve_specialist_profile(p).reserves_benchmark_lane:
        return True
    # Enablement specialists compile and boot a server; they need the whole machine.
    if bool(p.get("enablement")):
        return True
    return False


def specialist_lanes(params: dict[str, Any] | None, base_lanes: list[str]) -> list[str]:
    """Compute the lane list for a specialist dispatch.

    CPU specialists use only ``base_lanes`` (typically ``["research_lane"]``).
    GPU specialists replace ``research_lane`` with ``gpu_research_lane``.
    Bench specialists also add ``benchmark_lane``.
    """
    p = params or {}
    gpu = requires_gpu(p)
    profile = resolve_specialist_profile(p)
    lanes: list[str] = []
    for lane in base_lanes:
        if lane == "research_lane" and gpu:
            lanes.append("gpu_research_lane")
        else:
            lanes.append(lane)
    if gpu and "gpu_research_lane" not in lanes:
        lanes.append("gpu_research_lane")
    if profile.reserves_benchmark_lane and "benchmark_lane" not in lanes:
        lanes.append("benchmark_lane")
    return list(dict.fromkeys(lanes))


def wall_budget_base_min(params: dict[str, Any] | None) -> float:
    """Base wall-clock budget in minutes for a specialist, tiered by mode.

    Patch mode gets 60 minutes; research mode gets 10 minutes. The bench
    floor (rebench timeout + 10 min) is applied by the caller.
    """
    profile = resolve_specialist_profile(params or {})
    return 60.0 if profile.mode == MODE_PATCH else 10.0


def is_authoring_specialist(params: dict[str, Any] | None) -> bool:
    """True for an ENABLEMENT authoring specialist, which defaults to every GPU on the machine.

    FRAMEWORK authoring is no longer whole-machine GPU.
    """
    p = params or {}
    return bool(p.get("enablement"))


def uses_whole_machine_gpu_lane(params: dict[str, Any] | None) -> bool:
    """True when a GPU specialist should lease the *whole machine* (time-shared with serving via ``gpu_research_lane``) rather than the serving-disjoint ``gpu_specialist_pool``."""
    if is_authoring_specialist(params):
        return True
    return resolve_specialist_profile(params or {}).reserves_benchmark_lane


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
