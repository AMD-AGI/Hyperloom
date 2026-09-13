"""Canonical ``roofline_source`` provenance enum, shared across trace routes."""

from __future__ import annotations

#: No perf model; bound_type is a structural default (unestimable / no shapes).
PLACEHOLDER = "placeholder"
#: Bound derived from an analytical roofline model (shapes/op-model + ceiling).
ANALYTICAL = "analytical"
