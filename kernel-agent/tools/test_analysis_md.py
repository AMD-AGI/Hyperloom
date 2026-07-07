###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the shared canonical analysis.md renderer (_analysis_md) and the
cross-route consistency it guarantees (bypass vs TraceLens deterministic).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _analysis_md as am  # noqa: E402
import _bypass_report as br  # noqa: E402
import tracelens_analysis as tl  # noqa: E402
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
                "name": "k", "time_us": 100.0, "gpu_pct": 50.0, "efficiency_percent": 40.0,
                "arithmetic_intensity": 12.0, "bound_type": "compute_bound",
                "category": "GEMM", "source_file": "f.py",
            }
        ],
        p_items=[
            {
                "rank": 0, "category": "GEMM",
                "rows": [
                    {
                        "name": "k", "time_us": 100.0, "gpu_pct": 50.0, "e2e_pct": None,
                        "call_count": 3, "flops_per_byte": None, "efficiency_percent": 40.0,
                        "bound_type": "compute_bound", "args": ["(4,4)"], "source_file": "f.py",
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
        route="deterministic",
        model_name="",
        provenance_detail="",
        exec_summary={k: None for k in kw["exec_summary"]},
        system_signals={"idle_pct": None, "exposed_comm_pct": None, "exposed_memcpy_pct": None},
        hot_kernels=[],
        p_items=[],
    )
    kw["exec_summary"]["top_bottleneck_category"] = ""
    md = am.render_report(**kw)
    # no model -> plain title (no em-dash suffix)
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


# ── cross-route consistency: identical canonical spine ───────────────────────

# The section headings + table-header rows both routes MUST emit identically.
_SHARED_SPINE = (
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


def _deterministic_md(tmp_path):
    cands = [
        {
            "name": "gemm_kernel", "duration_us": 500.0, "gpu_pct": 60.0, "efficiency_percent": 45.0,
            "tracelens_category": "GEMM", "bound_type": "compute_bound", "source_file": "f.py",
            "tracelens_pitem_rank": 0, "impact_score": 1.2, "call_count": 3, "shapes": ["(4,4)"],
            "kernel_path": "launch.py",
        }
    ]
    path = tl.generate_minimal_analysis_md(tmp_path, cands, idle_pct=20.0, model_name="M")
    return path.read_text(encoding="utf-8")


def test_both_routes_share_canonical_spine(tmp_path):
    bypass_md = _bypass_md()
    det_md = _deterministic_md(tmp_path)
    for marker in _SHARED_SPINE:
        assert marker in bypass_md, f"bypass missing: {marker}"
        assert marker in det_md, f"deterministic missing: {marker}"
    # route ids differ in the provenance line, titles match format
    assert "route=bypass" in bypass_md.replace("HYPERLOOM_TRACE_ANALYSIS_ROUTE=", "route=")
    assert "route=deterministic" in det_md.replace("HYPERLOOM_TRACE_ANALYSIS_ROUTE=", "route=")
    # bypass carries its richer extras under the divider; deterministic does not
    assert "Additional route-specific detail below" in bypass_md
    assert "Additional route-specific detail below" not in det_md


def test_category_vocabulary_canonical_and_consistent(tmp_path):
    # Category display uses ONE canonical vocabulary on both routes: no raw
    # lowercase leaks (e.g. "gemm"), and the shared spine before the bypass
    # divider carries canonical TitleCase labels.
    det_md = _deterministic_md(tmp_path)
    # deterministic fed raw tracelens_category "gemm" -> must render canonical
    assert "| GEMM |" in det_md
    assert "| gemm |" not in det_md
    assert "### P0: GEMM kernels" in det_md
    # bypass Top Hot Kernels categories are canonical TitleCase too (fixture is
    # SDPA + GEMM); "Others" would have collapsed to "Other" had it appeared.
    bypass_spine = _bypass_md().split("Additional route-specific detail below")[0]
    assert "| Others |" not in bypass_spine
    assert "| GEMM |" in bypass_spine and "| SDPA |" in bypass_spine
