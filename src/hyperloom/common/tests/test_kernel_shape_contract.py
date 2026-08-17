# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shape-provenance contract.

The dispatch gate tests membership in ``DISPATCHABLE_SHAPE_PROVENANCE``
directly; the ``is_dispatchable_shape_provenance`` wrapper it used to call is
gone, so these assert the set the gate actually reads.
"""

from hyperloom.common.kernel_shape_contract import (
    ALLOWED_SHAPE_PROVENANCE,
    DISPATCHABLE_SHAPE_PROVENANCE,
)


def test_dispatchable_and_allowed_match():
    assert ALLOWED_SHAPE_PROVENANCE == DISPATCHABLE_SHAPE_PROVENANCE


def test_capture_backfill_is_dispatchable():
    assert "capture_backfill" in DISPATCHABLE_SHAPE_PROVENANCE


def test_geometry_provenance_not_dispatchable():
    assert "launch_grid" not in DISPATCHABLE_SHAPE_PROVENANCE
