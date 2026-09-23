# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Ceiling CSV seam: native write at the assembly sites + MAIDAS read-in-place.

These exercise the seam functions directly (no model/meta needed): the native-mode
``write_ceiling_arm`` publishes composed arms, and ``compute_roofline_breakdown_from_state``
reads an external (MAIDAS-authored) ``roofline_ceiling.csv`` in place of native compute.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hyperloom.orchestrator.kernel.roofline_csv as rc
from hyperloom.orchestrator.kernel.roofline_ceiling import (
    RooflineBreakdown,
    compute_roofline_breakdown_from_state,
    write_ceiling_arm,
)
from hyperloom.orchestrator.state.shared_state import SharedState


def _state(tmp_path: Path, **kw) -> SharedState:
    s = SharedState(**kw)
    s._session_dir = tmp_path
    return s


def test_native_write_ceiling_arm_publishes_arms(tmp_path: Path) -> None:
    s = _state(tmp_path)  # native mode (no external dir, not disabled)
    (tmp_path / "reports").mkdir()
    write_ceiling_arm(s, "baseline", RooflineBreakdown(600.0, 1800.0, 600.0, "memory"))
    write_ceiling_arm(s, "current_best", RooflineBreakdown(640.0, 1835.0, 640.0, "memory"))

    out = rc.read_ceiling(tmp_path / "reports" / "roofline_ceiling.csv")
    assert out["baseline"]["peak_tok_per_sec"] == 600.0
    assert out["baseline"]["bound_kind"] == "memory"
    # second arm did not clobber the first (upsert)
    assert out["current_best"]["peak_tok_per_sec"] == 640.0


def test_write_ceiling_arm_noop_in_maidas_mode(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    ext.mkdir()
    s = _state(tmp_path, roofline_csv_dir=str(ext))  # MAIDAS mode -> HL writes nothing
    write_ceiling_arm(s, "baseline", RooflineBreakdown(1.0, 2.0, 1.0, "memory"))
    assert not (tmp_path / "reports" / "roofline_ceiling.csv").exists()
    assert not (ext / "roofline_ceiling.csv").exists()


def test_write_ceiling_arm_noop_when_disabled(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir()
    s = _state(tmp_path, roofline_csv_disabled=True)
    write_ceiling_arm(s, "baseline", RooflineBreakdown(1.0, 2.0, 1.0, "memory"))
    assert not (tmp_path / "reports" / "roofline_ceiling.csv").exists()


def test_maidas_read_replaces_native_compute(tmp_path: Path) -> None:
    # External MAIDAS authored the composed ceiling; HL must read it instead of computing.
    ext = tmp_path / "ext"
    ext.mkdir()
    rc.write_ceiling(
        [{"row_type": "arm", "arm": "baseline", "mem_tok_per_sec": 626.96, "cmp_tok_per_sec": 1820.5, "peak_tok_per_sec": 626.96, "bound_kind": "memory"}],
        ext / "roofline_ceiling.csv",
    )
    s = _state(tmp_path, roofline_csv_dir=str(ext))
    bd = compute_roofline_breakdown_from_state(s, arm="baseline")
    assert bd.peak_tok_per_sec == 626.96
    assert bd.mem_tok_per_sec == 626.96
    assert bd.cmp_tok_per_sec == 1820.5
    assert bd.bound_kind == "memory"


def test_maidas_read_derives_bound_kind_when_blank(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    ext.mkdir()
    # bound_kind blank -> reader derives memory if mem <= cmp else compute.
    rc.write_ceiling(
        [{"row_type": "arm", "arm": "baseline", "mem_tok_per_sec": 500.0, "cmp_tok_per_sec": 900.0, "peak_tok_per_sec": 500.0, "bound_kind": ""}],
        ext / "roofline_ceiling.csv",
    )
    s = _state(tmp_path, roofline_csv_dir=str(ext))
    bd = compute_roofline_breakdown_from_state(s, arm="baseline")
    assert bd.bound_kind == "memory"  # 500 <= 900


def test_maidas_strict_missing_arm_raises(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    ext.mkdir()
    rc.write_ceiling(
        [{"row_type": "arm", "arm": "baseline", "peak_tok_per_sec": 100.0, "bound_kind": "memory"}],
        ext / "roofline_ceiling.csv",
    )
    s = _state(tmp_path, roofline_csv_dir=str(ext), roofline_csv_strict=True)
    # current_best arm is absent -> strict fails hard.
    with pytest.raises(FileNotFoundError):
        compute_roofline_breakdown_from_state(s, arm="current_best")
