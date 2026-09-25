# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Ceiling CSV seam: native write at the assembly sites + external read-in-place.

These exercise the seam functions directly (no model/meta needed): the native-mode
``write_ceiling_arm`` publishes composed arms, and ``compute_roofline_breakdown_from_state``
reads an external (externally authored) ``roofline_ceiling.csv`` in place of native compute.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hyperloom.orchestrator.kernel.roofline_csv as rc
from hyperloom.orchestrator.kernel.roofline_ceiling import (
    RooflineBreakdown,
    ceiling_config_columns,
    ceiling_config_key,
    compute_roofline_breakdown_from_state,
    write_ceiling_arm,
)
from hyperloom.orchestrator.state.shared_state import SharedState


def _state(tmp_path: Path, **kw) -> SharedState:
    s = SharedState(**kw)
    s._session_dir = tmp_path
    return s


def test_native_write_ceiling_arm_publishes_arms(tmp_path: Path) -> None:
    # Rows are keyed by the CONFIG that produced them; two DISTINCT configs never clobber.
    s_a = _state(tmp_path, precision="fp8")
    s_b = _state(tmp_path, precision="mxfp4")
    key_a, _, _ = ceiling_config_key(s_a, "baseline")
    key_b, _, _ = ceiling_config_key(s_b, "current_best")
    assert key_a != key_b  # distinct precision -> distinct content key
    write_ceiling_arm(s_a, "baseline", RooflineBreakdown(600.0, 1800.0, 600.0, "memory"))
    write_ceiling_arm(s_b, "current_best", RooflineBreakdown(640.0, 1835.0, 640.0, "memory"))

    # All CSVs live under the dedicated <session>/roofline_csv subfolder.
    out = rc.read_ceiling(tmp_path / "roofline_csv" / "roofline_ceiling.csv")
    assert out[key_a]["peak_tok_per_sec"] == 600.0
    assert out[key_a]["bound_kind"] == "memory"
    assert out[key_b]["peak_tok_per_sec"] == 640.0  # distinct config -> not clobbered


def test_write_ceiling_arm_noop_in_external_mode(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    ext.mkdir()
    s = _state(tmp_path, roofline_csv_dir=str(ext))  # external-CSV mode -> HL writes nothing
    write_ceiling_arm(s, "baseline", RooflineBreakdown(1.0, 2.0, 1.0, "memory"))
    assert not (tmp_path / "roofline_csv" / "roofline_ceiling.csv").exists()
    assert not (ext / "roofline_ceiling.csv").exists()


def test_write_ceiling_arm_noop_when_disabled(tmp_path: Path) -> None:
    s = _state(tmp_path, roofline_csv_disabled=True)
    write_ceiling_arm(s, "baseline", RooflineBreakdown(1.0, 2.0, 1.0, "memory"))
    assert not (tmp_path / "roofline_csv" / "roofline_ceiling.csv").exists()


def test_external_read_replaces_native_compute(tmp_path: Path) -> None:
    # An external program authored the composed ceiling; HL must read it instead of computing.
    # The external row is keyed by the same CONFIG key HL builds at read time.
    ext = tmp_path / "ext"
    ext.mkdir()
    s = _state(tmp_path, roofline_csv_dir=str(ext))
    rc.write_ceiling(
        [{"row_type": "arm", **ceiling_config_columns(s, "baseline"), "mem_tok_per_sec": 626.96, "cmp_tok_per_sec": 1820.5, "peak_tok_per_sec": 626.96, "bound_kind": "memory"}],
        ext / "roofline_ceiling.csv",
    )
    bd = compute_roofline_breakdown_from_state(s, arm="baseline")
    assert bd.peak_tok_per_sec == 626.96
    assert bd.mem_tok_per_sec == 626.96
    assert bd.cmp_tok_per_sec == 1820.5
    assert bd.bound_kind == "memory"


def test_external_read_derives_bound_kind_when_blank(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    ext.mkdir()
    s = _state(tmp_path, roofline_csv_dir=str(ext))
    # bound_kind blank -> reader derives memory if mem <= cmp else compute.
    rc.write_ceiling(
        [{"row_type": "arm", **ceiling_config_columns(s, "baseline"), "mem_tok_per_sec": 500.0, "cmp_tok_per_sec": 900.0, "peak_tok_per_sec": 500.0, "bound_kind": ""}],
        ext / "roofline_ceiling.csv",
    )
    bd = compute_roofline_breakdown_from_state(s, arm="baseline")
    assert bd.bound_kind == "memory"  # 500 <= 900


def test_external_strict_missing_arm_raises(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    ext.mkdir()
    # A row for a DIFFERENT config exists, but not the one this run resolves to.
    rc.write_ceiling(
        [{"row_type": "arm", **rc.ceiling_key_columns({"fw": "vllm", "conc": 999}), "peak_tok_per_sec": 100.0, "bound_kind": "memory"}],
        ext / "roofline_ceiling.csv",
    )
    s = _state(tmp_path, roofline_csv_dir=str(ext), roofline_csv_strict=True)
    # this run's config is absent from the CSV -> strict fails hard.
    with pytest.raises(FileNotFoundError):
        compute_roofline_breakdown_from_state(s, arm="current_best")


# --------------------------------------------------------------------------- #
# A consumer — arch-peak swap from gpu_arch_peaks.csv (case-insensitive).
# --------------------------------------------------------------------------- #

from hyperloom.orchestrator.kernel.roofline_ceiling import (  # noqa: E402
    _arch_peak_resolver,
    _hbm_bw_gbps,
    _resolve_achievable_tflops,
)


def test_resolver_case_insensitive_device_match(tmp_path: Path) -> None:
    # Producer writes upper-case normalize_platform names; consumers key off lower-case gpu_type.
    rc.write_arch_peaks(
        [{"name": "MI355X", "mem_bw_gbps": 8000.0, "matrix_bf16_tflops": 1686.0, "matrix_mx4_tflops": 5663.0}],
        tmp_path / "gpu_arch_peaks.csv",
    )
    res = rc.RooflineResolver(tmp_path)
    assert res.arch_peak("mi355x", "bf16") == 1686.0  # lower-case device hits upper-case row
    assert res.arch_peak("mi355x", "fp4") == 5663.0  # fp4 synonym -> mx4 column
    assert res.mem_bw("MI355X") == 8000.0


def test_achievable_tflops_peak_swaps_when_resolver_has_device(tmp_path: Path) -> None:
    rc.write_arch_peaks(
        [{"name": "MI355X", "mem_bw_gbps": 8000.0, "matrix_bf16_tflops": 1234.0}],
        tmp_path / "gpu_arch_peaks.csv",
    )
    res = rc.RooflineResolver(tmp_path)
    # CSV value wins over the hardcoded HW_SPECS_ACHIEVABLE table.
    assert _resolve_achievable_tflops("mi355x", "bf16", res) == 1234.0
    # No resolver -> falls back to the table (non-zero for a known device/precision).
    assert _resolve_achievable_tflops("mi355x", "bf16", None) > 0
    # hbm swap likewise.
    spec = {"hbm_bw_gbps": 5300.0}
    assert _hbm_bw_gbps(spec, "mi355x", res) == 8000.0
    assert _hbm_bw_gbps(spec, "mi355x", None) == 5300.0


def test_arch_peak_resolver_inert_in_native_mode(tmp_path: Path) -> None:
    # Native mode: the peak-swap is intentionally OFF even if a self-written CSV exists,
    # so the ceiling denominator stays consistent (table) across baseline + per-cycle.
    (tmp_path / "roofline_csv").mkdir()
    rc.write_arch_peaks(
        [{"name": "MI355X", "mem_bw_gbps": 8000.0, "matrix_bf16_tflops": 1234.0}],
        tmp_path / "roofline_csv" / "gpu_arch_peaks.csv",
    )
    s = _state(tmp_path)  # native (no external roofline_csv_dir)
    assert _arch_peak_resolver(s) is None


def test_arch_peak_resolver_live_in_external_mode(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    ext.mkdir()
    rc.write_arch_peaks(
        [{"name": "MI355X", "mem_bw_gbps": 8000.0, "matrix_bf16_tflops": 1234.0}],
        ext / "gpu_arch_peaks.csv",
    )
    s = _state(tmp_path, roofline_csv_dir=str(ext))  # external mode: authoritative peaks supplied externally
    res = _arch_peak_resolver(s)
    assert res is not None
    assert res.arch_peak("mi355x", "bf16") == 1234.0
