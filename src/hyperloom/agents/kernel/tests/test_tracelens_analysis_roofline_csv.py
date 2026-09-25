# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Tool-side roofline CSV interface: the K/A producers and the K consumer overlay.

``tracelens_analysis.py`` runs as a subprocess; these exercise its CSV helpers in
process. The producers publish the per-kernel analytical (``kernel_roofline.csv``) and
the arch peaks (``gpu_arch_peaks.csv``); the consumer overlays the per-kernel analytical
back from the CSV onto the loaded PerfModel rows. MEASURED/DERIVED columns never appear.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"

# The tool imports its siblings by bare name (``from _task_group_contract import ...``),
# so the tools dir must be on sys.path before the module executes.
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import tracelens_analysis as t  # noqa: E402


class _Args:
    def __init__(self, write_dir: str = "", read_dir: str = "") -> None:
        self.roofline_csv_dir = write_dir
        self.roofline_csv_read_dir = read_dir


def test_write_dir_and_read_dir_resolution(tmp_path: Path) -> None:
    a = _Args(write_dir=str(tmp_path / "rc"))
    assert t._roofline_csv_write_dir(a) == tmp_path / "rc"
    # read falls back to the write dir when not set separately
    assert t._roofline_csv_read_dir(a) == tmp_path / "rc"
    a2 = _Args(write_dir=str(tmp_path / "w"), read_dir=str(tmp_path / "r"))
    assert t._roofline_csv_read_dir(a2) == tmp_path / "r"
    assert t._roofline_csv_write_dir(_Args()) is None


def test_produce_kernel_csv_keeps_only_analytical(tmp_path: Path) -> None:
    csv_dir = tmp_path / "roofline_csv"
    payload = {
        "kernels": [
            {
                "name": "triton_gemm_x",
                "bound_type": "compute_bound",
                "arithmetic_intensity": 42.0,
                "kernel_category": "gemm",
                # measured / derived — must be dropped:
                "duration_us": 9.0,
                "gpu_pct": 5.0,
                "efficiency_percent": 88.0,
            }
        ]
    }
    t.publish_kernel_roofline_csv(csv_dir, payload, None)
    import hyperloom.orchestrator.kernel.roofline_csv as rc

    rows = rc.read_kernel_roofline(csv_dir / "kernel_roofline.csv")
    row = rows[rc.canonical_key("triton_gemm_x")]
    assert row["bound_type"] == "compute"
    assert row["arithmetic_intensity"] == 42.0
    assert row["kernel_category"] == "gemm"
    # measured/derived not columns of kernel_roofline.csv at all
    assert "duration_us" not in row and "gpu_pct" not in row and "efficiency_percent" not in row


def test_produce_arch_csv_renames_fp4(tmp_path: Path) -> None:
    csv_dir = tmp_path / "roofline_csv"
    arch = tmp_path / "MI355X.json"
    arch.write_text(
        json.dumps(
            {"name": "MI355X", "mem_bw_gbps": 8000.0, "max_achievable_tflops": {"matrix_bf16": 1686.0, "matrix_fp4": 5663.0}}
        )
    )
    t.publish_arch_peaks_csv(csv_dir, arch, None)
    import hyperloom.orchestrator.kernel.roofline_csv as rc

    peaks = rc.read_arch_peaks(csv_dir / "gpu_arch_peaks.csv")
    assert peaks["MI355X"]["matrix_bf16_tflops"] == 1686.0
    assert peaks["MI355X"]["matrix_mx4_tflops"] == 5663.0  # fp4 -> mx4


def test_consume_kernel_csv_overlays_bottleneck_and_ai(tmp_path: Path) -> None:
    csv_dir = tmp_path / "roofline_csv"
    t.publish_kernel_roofline_csv(
        csv_dir,
        {"kernels": [{"name": "triton_gemm_x", "bound_type": "memory_bound", "arithmetic_intensity": 7.0}]},
        None,
    )
    # No PerfModel JSON; the analytical comes entirely from the CSV.
    out = t.load_roofline_results("", read_dir=csv_dir)
    row = out[t.roofline_match_key("triton_gemm_x")]
    assert row["bottleneck"] == "memory"
    assert row["arithmetic_intensity"] == 7.0


def test_load_roofline_intra_cycle_round_trip(tmp_path: Path) -> None:
    # Native round-trip: a PerfModel JSON is loaded, PUBLISHED to kernel_roofline.csv, then
    # READ BACK — so the returned analytical is CSV-sourced on the very first cycle (no bypass).
    csv_dir = tmp_path / "roofline_csv"
    pm = tmp_path / "perfmodel.json"
    pm.write_text(
        json.dumps([{"name": "triton_gemm_x", "bottleneck": "compute_bound", "arithmetic_intensity": 55.0}])
    )
    assert not (csv_dir / "kernel_roofline.csv").exists()
    out = t.load_roofline_results(str(pm), write_dir=csv_dir, read_dir=csv_dir)
    # The CSV was written this call (producer) ...
    assert (csv_dir / "kernel_roofline.csv").exists()
    # ... and the returned analytical came back through it (consumer overlay).
    row = out[t.roofline_match_key("triton_gemm_x")]
    assert row["bottleneck"] == "compute"
    assert row["arithmetic_intensity"] == 55.0


def test_producers_noop_without_dir(tmp_path: Path) -> None:
    # None dir -> no file, no raise.
    t.publish_kernel_roofline_csv(None, {"kernels": [{"name": "k"}]}, None)
    t.publish_arch_peaks_csv(None, tmp_path / "x.json", None)
    assert not (tmp_path / "kernel_roofline.csv").exists()
