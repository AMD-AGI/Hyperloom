# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from hyperloom.common.kernel_shape_contract import (
    ALLOWED_SHAPE_PROVENANCE,
    DISPATCHABLE_SHAPE_PROVENANCE,
    is_dispatchable_shape_provenance,
)


def test_dispatchable_and_allowed_match():
    assert ALLOWED_SHAPE_PROVENANCE == DISPATCHABLE_SHAPE_PROVENANCE


def test_capture_backfill_is_dispatchable():
    assert is_dispatchable_shape_provenance("capture_backfill")


def test_geometry_provenance_not_dispatchable():
    assert not is_dispatchable_shape_provenance("launch_grid")
