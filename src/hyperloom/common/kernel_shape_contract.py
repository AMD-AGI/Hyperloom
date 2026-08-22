# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared kernel shape provenance contract for bypass analysis and kernel-opt."""

from __future__ import annotations

# Operand dims read out of the trace.
MEASURED_SHAPE_PROVENANCE = frozenset({"torch_trace", "capture_backfill", "tuning_csv"})

# Operand dims the candidate-review session supplied. Split by how it got them,
# because the two are worth different confidence when a tuned kernel later fails
# to move end-to-end throughput: a backfill is a recorded shape the deterministic
# lookup merely failed to join to this row, whereas a derivation is arithmetic
# over the model config and serving arguments and can be wrong in ways nothing
# detects until the integration benchmark runs.
REVIEW_BACKFILL_PROVENANCE = "review_backfill"
REVIEW_DERIVED_PROVENANCE = "review_derived"
REVIEW_SHAPE_PROVENANCE = frozenset({REVIEW_BACKFILL_PROVENANCE, REVIEW_DERIVED_PROVENANCE})

# Provenances the kernel-opt dispatch gate accepts. Review-supplied dims are
# admitted deliberately: under CUDA graph capture a replay has no cpu_op parent,
# so the trace records no arguments at all for the hottest kernels, and refusing
# the review's answer does not fall back to a measured shape -- it falls back to
# the tuning backend inventing one with no view of the serving configuration.
DISPATCHABLE_SHAPE_PROVENANCE = MEASURED_SHAPE_PROVENANCE | REVIEW_SHAPE_PROVENANCE

# Alias used by the kernel-opt predispatch validator.
ALLOWED_SHAPE_PROVENANCE = DISPATCHABLE_SHAPE_PROVENANCE
