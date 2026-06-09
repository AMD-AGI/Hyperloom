# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Specialist dispatch profile — the four orthogonal dials that replace the
former specialist / dynamic_action / dynamic_specialist split.

A single ``specialist`` worker is parameterised by:

* ``scope``  — ``domain`` (single catalogue domain), ``domains`` (cross-domain
  combination), or ``freeform`` (no domain lock; natural-language task).
* ``mode``   — ``research`` (read-only; produce findings) or ``patch``
  (worktree; produce a real unified diff).
* ``bench``  — whether the worker may run in-loop micro-benchmarks
  (only meaningful for ``mode == patch``).
* ``lane``   — ``cpu`` (research / freeform default) or ``gpu`` (patch + bench).

Defaults preserve the legacy single-domain patch-authoring behaviour so an
existing ``delegate{action='specialist', params={domain, gap, ...}}`` call is
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


# Backward-compatible defaults: a bare ``specialist`` dispatch keeps the legacy
# single-domain, patch-authoring, GPU-leased behaviour.
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
        return self.scope == SCOPE_FREEFORM

    @property
    def is_cross_domain(self) -> bool:
        return self.scope == SCOPE_DOMAINS

    @property
    def grants_bench_tool(self) -> bool:
        """True iff this dispatch may use the in-loop ``run_bench`` tool. Only
        patch-authoring specialists with ``bench=True`` qualify (bench has no
        meaning for read-only research)."""
        return self.mode == MODE_PATCH and self.bench


def _coerce_bool(value: Any, default: bool) -> bool:
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


def resolve_specialist_profile(params: dict[str, Any] | None) -> SpecialistProfile:
    """Read ``scope`` / ``mode`` / ``bench`` / ``lane`` from dispatch params.

    Unknown / missing values fall back to the legacy-compatible defaults so the
    resolver never raises; PolicyGate is the place that *rejects* malformed
    dispatches, this helper only normalises for the runtime.
    """
    p = params or {}

    scope = str(p.get("scope") or "").strip().lower()
    if scope not in SCOPE_VALUES:
        scope = DEFAULT_SCOPE

    mode = str(p.get("mode") or "").strip().lower()
    if mode not in MODE_VALUES:
        # Freeform recon defaults to read-only research; everything else keeps
        # the legacy patch-authoring default.
        mode = MODE_RESEARCH if scope == SCOPE_FREEFORM else DEFAULT_MODE

    bench = _coerce_bool(p.get("bench"), DEFAULT_BENCH)
    # bench only has meaning when the worker can write a patch.
    if mode != MODE_PATCH:
        bench = False

    lane = str(p.get("lane") or "").strip().lower()
    if lane not in LANE_VALUES:
        # CPU lane for read-only research / freeform recon; GPU lane when the
        # worker authors patches (and especially when it benches).
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
    "resolve_specialist_profile",
]
