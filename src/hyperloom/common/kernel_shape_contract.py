# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared kernel shape provenance contract for bypass analysis and kernel-opt."""

from __future__ import annotations

# Operand-dim provenances trusted by the kernel-opt dispatch gate.
DISPATCHABLE_SHAPE_PROVENANCE = frozenset({"torch_trace", "capture_backfill", "tuning_csv"})

# Alias used by the kernel-opt predispatch validator.
ALLOWED_SHAPE_PROVENANCE = DISPATCHABLE_SHAPE_PROVENANCE


def is_dispatchable_shape_provenance(provenance: str) -> bool:
    """Return whether ``provenance`` carries dispatch-grade operand dims."""
    return provenance in DISPATCHABLE_SHAPE_PROVENANCE
