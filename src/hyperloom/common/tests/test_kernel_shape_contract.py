# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shape-provenance set the dispatch gate tests membership against."""

from hyperloom.common.kernel_shape_contract import (
    ALLOWED_SHAPE_PROVENANCE,
    DISPATCHABLE_SHAPE_PROVENANCE,
    MEASURED_SHAPE_PROVENANCE,
)


def test_dispatchable_and_allowed_match():
    assert ALLOWED_SHAPE_PROVENANCE == DISPATCHABLE_SHAPE_PROVENANCE


def test_capture_backfill_is_dispatchable():
    assert "capture_backfill" in DISPATCHABLE_SHAPE_PROVENANCE


def test_geometry_provenance_not_dispatchable():
    assert "launch_grid" not in DISPATCHABLE_SHAPE_PROVENANCE


def test_only_measured_dims_are_dispatchable():
    assert DISPATCHABLE_SHAPE_PROVENANCE == MEASURED_SHAPE_PROVENANCE
