# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the high-GPU-time ``other``-bucket candidate recovery (#514).

Hyperloom builds candidates only from analysis.md reasoning-candidate (P-item)
blocks; TraceLens never emits such a block for a kernel it files under the
un-roofline'd ``other`` category, so a dominant editable kernel (the Triton
fused-MoE GEMM at ~67% GPU time) was silently dropped. The defense-in-depth
fallback recovers it from the per-op ranking sidecar so it flows through
classify_patchability and reaches GEAK.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

# tools/ is not a package — stick its dir on sys.path so we can import.
_TOOL_DIR = Path(__file__).resolve().parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

import tracelens_analysis as tla  # noqa: E402
import tracelens_skill_runner as tlr  # noqa: E402

_TRITON_MOE_SRC = (
    "/sgl-workspace/sglang/python/sglang/srt/layers/moe/"
    "fused_moe_triton/fused_moe.py"
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _ops_summary_csv(rows: str) -> str:
    return "name,op category,gpu time (ms)\n" + rows


# load_ops_ranking — schema-tolerant sidecar parsing
def test_load_ops_ranking_reads_ops_summary_csv(tmp_path):
    _write(
        tmp_path / "ops_summary.csv",
        _ops_summary_csv(
            "fused_moe_kernel,other,6700.0\n"
            "aten::mm,GEMM,2300.0\n"
            "rmsnorm,elementwise,1000.0\n"
        ),
    )
    ranking = tla.load_ops_ranking(tmp_path)
    by_name = {r["name"]: r for r in ranking}
    assert by_name["fused_moe_kernel"]["category"] == "other"
    # ms column scaled to microseconds.
    assert by_name["fused_moe_kernel"]["gpu_us"] == 6700.0 * 1000.0


def test_load_ops_ranking_reads_unified_perf_summary_in_perf_report_csvs(tmp_path):
    _write(
        tmp_path / "perf_report_csvs" / "unified_perf_summary.csv",
        "name,op category,gpu_time_us\nfused_moe_kernel,other,6700000\n",
    )
    ranking = tla.load_ops_ranking(tmp_path)
    assert ranking and ranking[0]["name"] == "fused_moe_kernel"
    assert ranking[0]["gpu_us"] == 6700000.0


def test_load_ops_ranking_reads_priority_data_json(tmp_path):
    _write(
        tmp_path / "priority_data.json",
        json.dumps({"findings": [
            {"name": "fused_moe_kernel", "category": "other", "gpu_pct": 67.0},
            {"name": "aten::mm", "category": "GEMM", "gpu_pct": 23.0},
        ]}),
    )
    ranking = tla.load_ops_ranking(tmp_path)
    by_name = {r["name"]: r for r in ranking}
    assert by_name["fused_moe_kernel"]["gpu_pct"] == 67.0


def test_load_ops_ranking_empty_when_absent(tmp_path):
    assert tla.load_ops_ranking(tmp_path) == []
    assert tla.load_ops_ranking(None) == []


# recover_other_bucket_candidates — the fallback gate
def test_recover_surfaces_high_time_other_bucket_op(tmp_path):
    _write(
        tmp_path / "ops_summary.csv",
        _ops_summary_csv(
            "fused_moe_kernel,other,6700.0\n"
            "aten::mm,GEMM,2300.0\n"
            "rmsnorm,elementwise,1000.0\n"
        ),
    )
    # analysis.md already surfaced aten::mm; the 67% other-bucket op was dropped.
    recovered = tla.recover_other_bucket_candidates(
        tmp_path, [{"name": "aten::mm"}], top_k=10,
    )
    assert len(recovered) == 1
    cand = recovered[0]
    assert cand["name"] == "fused_moe_kernel"
    assert cand["candidate_source"] == "other_bucket_fallback"
    assert cand["source_file"] == ""            # resolved later in _finalize
    assert cand["duration_us"] == 6700.0 * 1000.0
    assert round(cand["gpu_pct"], 0) == 67.0


def test_recover_skips_op_already_in_candidates(tmp_path):
    _write(
        tmp_path / "ops_summary.csv",
        _ops_summary_csv("fused_moe_kernel,other,6700.0\naten::mm,GEMM,3300.0\n"),
    )
    # The op is already an analysis.md candidate -> not duplicated.
    recovered = tla.recover_other_bucket_candidates(
        tmp_path, [{"name": "fused_moe_kernel"}], top_k=10,
    )
    assert recovered == []


def test_recover_skips_below_threshold(tmp_path):
    # other-bucket op at only ~5% of GPU time, default threshold is 10%.
    _write(
        tmp_path / "ops_summary.csv",
        _ops_summary_csv("tiny_other,other,500.0\naten::mm,GEMM,9500.0\n"),
    )
    recovered = tla.recover_other_bucket_candidates(
        tmp_path, [{"name": "aten::mm"}], top_k=10,
    )
    assert recovered == []


def test_recover_skips_rooflined_category(tmp_path):
    # A high-time GEMM op missing from candidates is NOT recovered: it is not
    # an "other"-bucket op (it would have had a reasoning-candidate block).
    _write(
        tmp_path / "ops_summary.csv",
        _ops_summary_csv("big_gemm,GEMM,9000.0\nrmsnorm,elementwise,1000.0\n"),
    )
    recovered = tla.recover_other_bucket_candidates(
        tmp_path, [], top_k=10,
    )
    assert recovered == []


def test_recover_threshold_env_override(tmp_path, monkeypatch):
    _write(
        tmp_path / "ops_summary.csv",
        _ops_summary_csv("tiny_other,other,500.0\naten::mm,GEMM,9500.0\n"),
    )
    monkeypatch.setenv("HYPERLOOM_OTHER_BUCKET_MIN_GPU_PCT", "1.0")
    recovered = tla.recover_other_bucket_candidates(
        tmp_path, [{"name": "aten::mm"}], top_k=10,
    )
    assert [c["name"] for c in recovered] == ["tiny_other"]


def test_recover_empty_when_no_sidecar(tmp_path):
    assert tla.recover_other_bucket_candidates(tmp_path, [], top_k=10) == []


def test_recover_from_priority_data_json(tmp_path):
    _write(
        tmp_path / "priority_data.json",
        json.dumps({"findings": [
            {"name": "fused_moe_kernel", "category": "other", "gpu_pct": 67.0},
        ]}),
    )
    recovered = tla.recover_other_bucket_candidates(tmp_path, [], top_k=10)
    assert [c["name"] for c in recovered] == ["fused_moe_kernel"]


# Headline #514 scenario: a synthetic analysis.md MISSING the high-time op.
_SYNTHETIC_ANALYSIS_MD = textwrap.dedent(
    """\
    # Synthetic Analysis

    ## Detailed Analysis

    ### Compute Kernel Insights

    <a id="detailed-analysis-compute-p1"></a>
    <!-- reasoning-candidate tier=compute rank=1 -->
    #### 🔴 P1: Compute-bound BF16 GEMMs underrunning roofline (Tensile)

    **Identification:** synthetic single GEMM candidate.

    **Data:**

    | Operation |  Args  |            Kernel Path                  | Time (ms) | %E2E | Count |FLOPS/Byte| Efficiency | Bound |
    |-----------|--------|-----------------------------------------|-----------|------|-------|----------|------------|-------|
    | aten::mm | (24576,8192) bf16 | — | 2300.0 | 23.0 | 320 | 5059.76 | 68.74% of 708 TFLOPS | compute-bound |
    """
)


def test_synthetic_analysis_md_missing_high_time_op_is_recovered(tmp_path):
    analysis_md = _write(tmp_path / "analysis.md", _SYNTHETIC_ANALYSIS_MD)
    report_cands = tlr.parse_analysis_md(analysis_md, top_k=10)
    assert [c["name"] for c in report_cands] == ["aten::mm"], (
        "synthetic analysis.md should yield exactly the P1 GEMM candidate"
    )
    # The #1 kernel (67% other-bucket Triton fused-MoE GEMM) has no
    # reasoning-candidate block, so it is absent from report_cands.
    _write(
        tmp_path / "ops_summary.csv",
        _ops_summary_csv(
            "fused_moe_kernel,other,6700.0\n"
            "aten::mm,GEMM,2300.0\n"
            "rmsnorm,elementwise,1000.0\n"
        ),
    )
    recovered = tla.recover_other_bucket_candidates(
        tmp_path, report_cands, top_k=10,
    )
    names = {c["name"] for c in recovered}
    assert names == {"fused_moe_kernel"}, (
        "the dropped high-time other-bucket op must be recovered, and the "
        "already-surfaced GEMM must not be duplicated"
    )


def test_recovered_other_bucket_kernel_routes_to_geak(tmp_path, monkeypatch):
    """End-to-end intent: a recovered raw candidate (source_file unset) flows
    through _finalize_candidates -> classify_patchability and a Triton kernel
    under /sgl-workspace/sglang/ is marked reusable_native_kernel=True (so it
    is routable to GEAK) instead of being silently dropped."""
    _write(
        tmp_path / "ops_summary.csv",
        _ops_summary_csv("fused_moe_kernel,other,6700.0\naten::mm,GEMM,3300.0\n"),
    )
    monkeypatch.setattr(
        tla, "_reusable_roots", lambda: ("/sgl-workspace/sglang/",),
    )
    monkeypatch.setattr(
        tla, "locate_source_via_grep", lambda _name: _TRITON_MOE_SRC,
    )
    recovered = tla.recover_other_bucket_candidates(
        tmp_path, [{"name": "aten::mm"}], top_k=10,
    )
    finalized = tla._finalize_candidates(recovered, total_dur=None)
    item = finalized[0]
    assert item["name"] == "fused_moe_kernel"
    # No longer dropped: it was classified (key present) AND routed to GEAK.
    assert "reusable_native_kernel" in item
    assert item["reusable_native_kernel"] is True, item.get("skip_reason")
    assert item["source_file"] == _TRITON_MOE_SRC


def test_classify_patchability_marks_triton_sglang_kernel_reusable(monkeypatch):
    monkeypatch.setattr(
        tla, "_reusable_roots", lambda: ("/sgl-workspace/sglang/",),
    )
    reusable, reason = tla.classify_patchability({
        "name": "fused_moe_kernel",
        "source_file": _TRITON_MOE_SRC,
        "source_type": "triton",
    })
    assert reusable is True, reason
