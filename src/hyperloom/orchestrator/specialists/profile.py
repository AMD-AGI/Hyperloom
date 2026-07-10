# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Specialist dispatch profile — the four orthogonal dials that parameterise a
single ``specialist`` worker.

A single ``specialist`` worker is parameterised by:

* ``scope``  — ``domain`` (single catalogue domain), ``domains`` (cross-domain
  combination), or ``freeform`` (no domain lock; natural-language task).
* ``mode``   — ``research`` (read-only; produce findings) or ``patch``
  (worktree; produce a real unified diff).
* ``bench``  — whether the worker may run in-loop micro-benchmarks
  (only meaningful for ``mode == patch``).
* ``lane``   — ``cpu`` (research / freeform default) or ``gpu`` (patch + bench).

Defaults resolve a ``delegate{action='specialist', params={domain, gap, ...}}``
call to single-domain patch-authoring behaviour.
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


# Defaults. A dispatch that carries a domain/tag anchor resolves to the
# single-domain, patch-authoring, GPU-leased behaviour (DEFAULT_MODE/_LANE).
# A *truly bare* dispatch (no scope and no domain/tag anchor) is inferred to be
# ``freeform`` and therefore resolves to the cheap, read-only research/CPU lane
# — "safe & cheap first".
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
        """Whether this profile uses the free-form (unscoped) scope.

        Returns:
            ``True`` if the scope is free-form.
        """
        return self.scope == SCOPE_FREEFORM

    @property
    def is_cross_domain(self) -> bool:
        """Whether this profile spans multiple knowledge domains.

        Returns:
            ``True`` if the scope is the multi-domain scope.
        """
        return self.scope == SCOPE_DOMAINS

    @property
    def reserves_benchmark_lane(self) -> bool:
        """True iff this dispatch should contend for the ``benchmark_lane``.

        Bench-capable patch specialists (``mode=patch & bench=True``) run their
        own serving + benchmark loop on their leased cards, so they reserve the
        shared ``benchmark_lane`` to avoid oversubscribing benchmark resources.

        Returns:
            ``True`` when the profile is patch-mode with ``bench=True``.
        """
        return self.mode == MODE_PATCH and self.bench


def _coerce_bool(value: Any, default: bool) -> bool:
    """Coerce a loosely-typed value to a boolean.

    Accepts native bools, numbers, and common truthy/falsey strings
    (``"true"``/``"yes"``/``"on"`` and their negatives).

    Args:
        value: Value to interpret.
        default: Fallback returned when ``value`` is ``None`` or unrecognized.

    Returns:
        The interpreted boolean, or ``default`` when undecidable.
    """
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
    """Infer the dispatch scope when none is explicitly given.

    A dispatch that carries a domain/tag anchor is a real (cross-)domain
    specialist; one with no anchor at all is treated as ``freeform`` so a bare
    dispatch defaults to the cheap read-only lane instead of the expensive
    patch/GPU lane.

    Args:
        p: The dispatch params to inspect for domain/tag anchors.

    Returns:
        ``SCOPE_DOMAINS`` for two-or-more tags, ``SCOPE_DOMAIN`` for one, or
        ``SCOPE_FREEFORM`` when no anchor is present.
    """
    # Local import avoids a module-load cycle (specialist_domains is heavier).
    from .domains import normalize_dispatch_tags

    tags = normalize_dispatch_tags(p)
    if len(tags) >= 2:
        return SCOPE_DOMAINS
    if tags:
        return SCOPE_DOMAIN
    return SCOPE_FREEFORM


def uses_whole_machine_gpu_lane(params: dict[str, Any] | None) -> bool:
    """True when a GPU specialist should lease the *whole machine* (time-shared
    with serving via ``gpu_research_lane``) rather than the serving-disjoint
    ``gpu_specialist_pool``.

    Two dispatch shapes take the whole-machine, time-shared lane:

    * **framework-authoring** specialists (``framework_agent_authoring``) — the
      long-standing behaviour: they lease the whole node from
      ``framework_gpu_pool``.
    * **bench-capable** specialists (``mode=patch`` & ``bench=true``,
      i.e. :attr:`SpecialistProfile.reserves_benchmark_lane`) — they start a
      real TP-sharded server on the full serving-TP cards and run a benchmark
      loop, so they are already temporally mutually-exclusive with production
      serving via ``gpu_research_lane`` + ``benchmark_lane``. Because the
      serving process is torn down at the end of every explore/integrate round
      (and its cards freed), a bench specialist can safely take the whole
      machine in the gap between rounds. Critically, the serving-disjoint pool
      is *empty* whenever serving occupies the whole node (``TP == #GPUs``), so
      without this route bench specialists are structurally undispatchable on a
      whole-machine-serving session.

    Non-bench GPU probes (microbench / profiling, ``bench=false``) keep the
    serving-disjoint pool: they are designed to run on cards physically disjoint
    from serving.

    Args:
        params: The specialist dispatch params (carrying
            ``framework_agent_authoring`` / ``mode`` / ``bench`` / ``scope`` /
            ``lane``), or ``None``.

    Returns:
        ``True`` when the dispatch should draw from the whole-machine pool.
    """
    p = params or {}
    if bool(p.get("framework_agent_authoring")):
        return True
    return resolve_specialist_profile(p).reserves_benchmark_lane


def resolve_specialist_profile(params: dict[str, Any] | None) -> SpecialistProfile:
    """Read ``scope`` / ``mode`` / ``bench`` / ``lane`` from dispatch params.

    Unknown / missing values fall back to the legacy-compatible defaults so the
    resolver never raises; PolicyGate is the place that *rejects* malformed
    dispatches, this helper only normalises for the runtime.

    When ``scope`` is absent/unknown it is *inferred* from the presence of a
    domain/tag anchor: anchored dispatches resolve to ``domain``/``domains``
    (legacy patch/GPU default preserved), while a truly bare dispatch resolves
    to ``freeform`` → ``research`` → ``cpu`` (safe & cheap first).

    Args:
        params: The dispatch params carrying ``scope`` / ``mode`` / ``bench`` /
            ``lane``, or ``None``.

    Returns:
        The resolved :class:`SpecialistProfile` with legacy-compatible
        fallbacks applied.
    """
    p = params or {}

    scope = str(p.get("scope") or "").strip().lower()
    if scope not in SCOPE_VALUES:
        scope = _infer_scope(p)

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
    "uses_whole_machine_gpu_lane",
]
