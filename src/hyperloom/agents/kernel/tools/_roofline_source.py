"""Canonical ``roofline_source`` provenance enum, shared across trace routes.

Records HOW a kernel's roofline bound was derived. Vocabulary:

    - ``placeholder``: no perf model; ``bound_type`` is a structural default
      (e.g. shapes not captured), treat as unknown.
    - ``analytical``: bound derived from an analytical roofline model.
    - ``rocprof``: bound refined by a hardware measurement (strongest).

The ladder ``placeholder`` -> ``analytical`` -> ``rocprof`` only ever upgrades.
"""

from __future__ import annotations

#: No perf model; bound_type is a structural default (unestimable / no shapes).
PLACEHOLDER = "placeholder"
#: Bound derived from an analytical roofline model (shapes/op-model + ceiling).
ANALYTICAL = "analytical"
