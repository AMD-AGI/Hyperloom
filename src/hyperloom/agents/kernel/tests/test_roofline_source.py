###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the shared roofline_source provenance values (_roofline_source).

Locks the values both trace-analysis routes emit and confirms the bypass
analytical roofline plus TraceLens kernel-roofline row stamp the shared strings.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _roofline_source as rs  # noqa: E402


def test_shared_values():
    assert rs.PLACEHOLDER == "placeholder"
    assert rs.ANALYTICAL == "analytical"


def test_bypass_roofline_uses_shared_enum():
    from _bypass_roofline import compute_roofline

    r = compute_roofline(
        category="GEMM",
        shape_str="(4096,4096) bf16<br>(4096,4096) bf16",
        gpu_time_us=500.0,
        call_count=1,
        gpu_type="mi300x",
    )
    assert r is not None
    assert r["roofline_source"] == rs.ANALYTICAL


def test_tracelens_row_emits_shared_enum():
    import tracelens_analysis as tl

    # A candidate whose per-op perf model produced numbers -> analytical.
    row = tl._kernel_roofline_row({"kernel_id": "k001", "name": "x", "efficiency_percent": 40.0})
    assert row["roofline_source"] == rs.ANALYTICAL
    # A bare candidate with no perf-model numbers -> placeholder.
    bare = tl._kernel_roofline_row({"kernel_id": "k002", "name": "y"})
    assert bare["roofline_source"] == rs.PLACEHOLDER
