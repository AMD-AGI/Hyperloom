"""Canonical ``roofline_source`` provenance enum, shared across trace routes.

``roofline_source`` records HOW a kernel's roofline bound was derived, so a
downstream consumer can weigh the number regardless of which backend produced
it (the TraceLens agent/deterministic route or the standalone bypass reader).
Both routes emit values from this single vocabulary:

    - ``placeholder``: no perf model. The ``bound_type`` is a structural default
      (e.g. shapes were not captured), NOT a modelled result -- treat as unknown.
    - ``analytical``: bound derived from an analytical roofline model. On bypass
      this is captured shapes + measured kernel time vs the achievable ceiling;
      on TraceLens it is the per-op perf model vs the arch-benchmark ceiling.
    - ``rocprof``: bound refined by a hardware measurement (rocprof-compute),
      i.e. the strongest provenance. Reserved for the opt-in enrichment stage.

The ladder is ``placeholder`` -> ``analytical`` -> ``rocprof`` (weakest to
strongest); a later stage only ever upgrades the source.

Aggregation-view note (device-kernel vs aten-op), documented here so the two
routes' numbers are not naively equated: the bypass reader ranks/aggregates by
the *device kernel* (full GPU coverage, robust under cudagraph replay), whereas
TraceLens aggregates by the *aten operation* (one perf-model row per op, which
may fan out to several device kernels). The ``roofline_source`` enum is shared,
but a per-``kernel_id`` row from one route is not guaranteed to correspond 1:1
to a row from the other -- compare workload-level roofline aggregates, not raw
per-row identities, when reconciling the two backends.

Kept dependency-free (stdlib only) so the bypass reader can import it without
pulling in TraceLens.
"""

from __future__ import annotations

#: No perf model; bound_type is a structural default (unestimable / no shapes).
PLACEHOLDER = "placeholder"
#: Bound derived from an analytical roofline model (shapes/op-model + ceiling).
ANALYTICAL = "analytical"
#: Bound refined by a hardware measurement (rocprof-compute); strongest source.
ROCPROF = "rocprof"

#: The full set of valid ``roofline_source`` values.
VALID = frozenset({PLACEHOLDER, ANALYTICAL, ROCPROF})
