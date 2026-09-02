# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared kernel shape provenance contract for bypass analysis and kernel-opt."""

from __future__ import annotations

# Operand dims read out of the trace.
MEASURED_SHAPE_PROVENANCE = frozenset({"torch_trace", "capture_backfill", "tuning_csv"})

# Provenances the kernel-opt dispatch gate accepts.
DISPATCHABLE_SHAPE_PROVENANCE = MEASURED_SHAPE_PROVENANCE

# Alias used by the kernel-opt predispatch validator.
ALLOWED_SHAPE_PROVENANCE = DISPATCHABLE_SHAPE_PROVENANCE
