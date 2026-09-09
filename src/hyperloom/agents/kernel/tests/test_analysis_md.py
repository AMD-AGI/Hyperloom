###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Tests for the canonical analysis.md renderer (_analysis_md) and the report
structure it guarantees for the bypass route that consumes it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _analysis_md as am  # noqa: E402
import _bypass_report as br  # noqa: E402
from test_bypass_report import _KERNELS, _analyze  # noqa: E402


def _sample_kwargs():
    return dict(
        route="bypass",
        model_name="M",
        provenance_detail="detail.",
        exec_summary={
            "total_gpu_time_ms": 10.0,
            "gpu_busy_pct": 80.0,
            "gpu_idle_pct": 20.0,
            "gpu_memcpy_ms": 1.0,
            "top_bottleneck_category": "GEMM",
            "attribution_pct": 90.0,
        },
        system_signals={"idle_pct": 20.0, "exposed_comm_pct": None, "exposed_memcpy_pct": 5.0},
        idle_threshold=80.0,
        hot_kernels=[
            {
                "name": "k",
                "time_us": 100.0,
                "gpu_pct": 50.0,
                "efficiency_percent": 40.0,
                "arithmetic_intensity": 12.0,
                "bound_type": "compute_bound",
                "category": "GEMM",
                "source_file": "f.py",
            }
        ],
        p_items=[
            {
                "rank": 0,
                "category": "GEMM",
                "rows": [
                    {
                        "name": "k",
                        "time_us": 100.0,
                        "gpu_pct": 50.0,
                        "e2e_pct": None,
                        "call_count": 3,
                        "flops_per_byte": None,
                        "efficiency_percent": 40.0,
                        "bound_type": "compute_bound",
                        "args": ["(4,4)"],
                        "source_file": "f.py",
                        "kernel_path": None,
                    }
                ],
            }
        ],
    )


def test_render_report_canonical_sections():
    md = am.render_report(**_sample_kwargs())
    assert md.startswith("# Performance Analysis Report \u2014 M")
    assert "> Generated via bypass route (HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass). detail." in md
    assert "## Executive Summary" in md and "| Metric | Value |" in md
    assert "| Total GPU Time | 10.000 ms |" in md
    assert "| GPU Busy % | 80.00% |" in md
    assert "## System-Level Signals" in md and "| Signal | % of total GPU time | Note |" in md
    assert "| GPU idle | 20.00% | within 80% idle gate |" in md
    assert "## Top Hot Kernels" in md and am.TOP_HOT_KERNELS_COLUMNS in md
    assert "### P0: GEMM kernels" in md
    assert "<!-- reasoning-candidate tier=compute rank=0 -->" in md
    assert am.P_ITEM_COLUMNS in md


def test_render_report_missing_values_render_dash():
    kw = _sample_kwargs()
    kw.update(
        model_name="",
        provenance_detail="",
        exec_summary={k: None for k in kw["exec_summary"]},
        system_signals={"idle_pct": None, "exposed_comm_pct": None, "exposed_memcpy_pct": None},
        hot_kernels=[],
        p_items=[],
    )
    kw["exec_summary"]["top_bottleneck_category"] = ""
    md = am.render_report(**kw)
    # no model -> plain title
    assert md.splitlines()[0] == "# Performance Analysis Report"
    assert f"| Total GPU Time | {am.DASH} ms |" in md
    assert f"| GPU idle | {am.DASH} | - |" in md
    assert "_No GPU kernels found in trace._" in md


def test_extra_sections_appended_under_divider():
    kw = _sample_kwargs()
    kw["extra_sections"] = "## Route Extra\n\nbody"
    md = am.render_report(**kw)
    assert "Additional route-specific detail below" in md
    assert "## Route Extra" in md
    # extras come AFTER the shared Top Hot Kernels section
    assert md.index("## Top Hot Kernels") < md.index("## Route Extra")


# The canonical spine: section headings + table-header rows every report
# rendered through _analysis_md must carry.
_CANONICAL_SPINE = (
    "# Performance Analysis Report",
    "> Generated via ",
    "HYPERLOOM_TRACE_ANALYSIS_ROUTE=",
    "## Executive Summary",
    "| Metric | Value |",
    "## System-Level Signals",
    "| Signal | % of total GPU time | Note |",
    "## Top Hot Kernels",
    am.TOP_HOT_KERNELS_COLUMNS,
)


def _bypass_md():
    analyze = _analyze([dict(k) for k in _KERNELS])
    cands = br.build_candidates(analyze, framework="vllm", target_platform="MI300X")
    return br.render_analysis_md(cands, analyze, model_name="M", framework="vllm", target_platform="MI300X")


def test_bypass_report_carries_the_canonical_spine():
    bypass_md = _bypass_md()
    for marker in _CANONICAL_SPINE:
        assert marker in bypass_md, f"bypass missing: {marker}"
    assert "route=bypass" in bypass_md.replace("HYPERLOOM_TRACE_ANALYSIS_ROUTE=", "route=")
    # Route-specific detail lands under the divider, after the shared sections.
    assert "Additional route-specific detail below" in bypass_md


def test_category_vocabulary_is_canonical():
    # Category display uses one canonical vocabulary; the raw upstream spelling
    # (lowercase ``gemm``, ``Others``) must not reach the report.
    bypass_spine = _bypass_md().split("Additional route-specific detail below")[0]
    assert "| Others |" not in bypass_spine
    assert "| gemm |" not in bypass_spine
    assert "| GEMM |" in bypass_spine and "| SDPA |" in bypass_spine
