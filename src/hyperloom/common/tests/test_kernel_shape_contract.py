# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shape-provenance set the dispatch gate tests membership against."""

from hyperloom.common.kernel_shape_contract import (
    ALLOWED_SHAPE_PROVENANCE,
    DISPATCHABLE_SHAPE_PROVENANCE,
    MEASURED_SHAPE_PROVENANCE,
    REVIEW_SHAPE_PROVENANCE,
)


def test_dispatchable_and_allowed_match():
    assert ALLOWED_SHAPE_PROVENANCE == DISPATCHABLE_SHAPE_PROVENANCE


def test_capture_backfill_is_dispatchable():
    assert "capture_backfill" in DISPATCHABLE_SHAPE_PROVENANCE


def test_geometry_provenance_not_dispatchable():
    assert "launch_grid" not in DISPATCHABLE_SHAPE_PROVENANCE


def test_reviewed_dims_are_dispatchable():
    """Under graph capture the trace records no arguments for the hottest
    kernels, so refusing the review's dims does not fall back to a measured
    shape -- it falls back to the tuning backend inventing one.
    """
    assert REVIEW_SHAPE_PROVENANCE
    assert REVIEW_SHAPE_PROVENANCE <= DISPATCHABLE_SHAPE_PROVENANCE


def test_measured_and_reviewed_stay_distinguishable():
    """Collapsing the two would remove the only signal that says whether a
    disappointing end-to-end result is worth blaming on the shape.
    """
    assert not MEASURED_SHAPE_PROVENANCE & REVIEW_SHAPE_PROVENANCE
    assert DISPATCHABLE_SHAPE_PROVENANCE == MEASURED_SHAPE_PROVENANCE | REVIEW_SHAPE_PROVENANCE
