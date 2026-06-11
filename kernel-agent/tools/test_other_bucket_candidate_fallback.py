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

import pytest

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
            "rmsnorm,elementwise,500.0\n"  # ~5%, below the 10% floor
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
    # gpu_pct is over the full ranking total (6700+2300+500 = 9500 ms): ~70.5%.
    assert round(cand["gpu_pct"], 0) == 71.0


def test_recover_skips_op_already_in_candidates(tmp_path):
    _write(
        tmp_path / "ops_summary.csv",
        _ops_summary_csv("fused_moe_kernel,other,6700.0\naten::mm,GEMM,3300.0\n"),
    )
    # The op already surfaced in analysis.md (fused_moe_kernel) is NOT duplicated;
    # the broadened gate still recovers the high-time GEMM that was missing.
    recovered = tla.recover_other_bucket_candidates(
        tmp_path, [{"name": "fused_moe_kernel"}], top_k=10,
    )
    names = {c["name"] for c in recovered}
    assert "fused_moe_kernel" not in names           # dedup against existing
    assert names == {"aten::mm"}                      # broadened gate recovers it


def test_recover_skips_below_threshold(tmp_path):
    # op at only ~5% of GPU time, default threshold is 10%.
    _write(
        tmp_path / "ops_summary.csv",
        _ops_summary_csv("tiny_other,other,500.0\naten::mm,GEMM,9500.0\n"),
    )
    recovered = tla.recover_other_bucket_candidates(
        tmp_path, [{"name": "aten::mm"}], top_k=10,
    )
    assert recovered == []


def test_recover_surfaces_high_time_non_other_category(tmp_path):
    # Gate broadened (#515): a high-time op in a *rooflined* category that is
    # nonetheless missing from analysis.md candidates IS recovered (previously
    # this was skipped because it was not an "other"-bucket op).
    _write(
        tmp_path / "ops_summary.csv",
        _ops_summary_csv("big_gemm,GEMM,9000.0\nrmsnorm,elementwise,1000.0\n"),
    )
    recovered = tla.recover_other_bucket_candidates(
        tmp_path, [], top_k=10,
    )
    names = {c["name"] for c in recovered}
    assert "big_gemm" in names


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
            "rmsnorm,elementwise,500.0\n"  # ~5%, below the 10% floor
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


# ---------------------------------------------------------------------------
# REAL ops_summary.csv schema (#515): the columns TraceLens actually emits.
# The simplified `name,op category,gpu time (ms)` schema above does NOT exercise
# the real `Categories` (list-repr) / `total_direct_kernel_time_ms` /
# `Percentage (%)` columns, which is exactly the gap that left load_ops_ranking
# returning pct=None and dropping the dominant MoE_fused row.
# ---------------------------------------------------------------------------

_REAL_OPS_SUMMARY_HEADER = (
    "name,parent_module,total_direct_kernel_time_sum,"
    "total_subtree_kernel_time_sum,total_subtree_kernel_time_count,"
    "total_direct_kernel_time_ms,Count,Categories,call_stack_first,"
    "Percentage (%),Cumulative Percentage (%)"
)

# The actual dominant Triton fused-MoE row from the Qwen3-30B-A3B run.
_REAL_MOE_NAME = (
    "sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel_427"
)


def _real_ops_summary_csv() -> str:
    return "\n".join(
        [
            _REAL_OPS_SUMMARY_HEADER,
            # MoE_fused row: Categories is a python-list-repr string; quote the
            # call_stack cell because it contains commas/=>.
            f"{_REAL_MOE_NAME}, nn.Module: FusedMoE ,302875.90625,302875.90625,"
            "96,302.87590625,96,['MoE_fused'],"
            f'"{_REAL_MOE_NAME} => nn.Module: FusedMoE_0",'
            "67.44388342277517,67.44388342277517",
            # A GEMM row that analysis.md already surfaced as aten::mm.
            "aten::mm, nn.Module: QKVParallelLinear ,27301.67,27301.67,48,"
            "27.30167,48,['GEMM'],aten::mm => nn.Module: QKVParallelLinear_0,"
            "6.0794902,73.523373",
        ]
    ) + "\n"


def test_load_ops_ranking_reads_real_ops_summary_schema(tmp_path):
    _write(tmp_path / "ops_summary.csv", _real_ops_summary_csv())
    ranking = tla.load_ops_ranking(tmp_path)
    by_name = {r["name"]: r for r in ranking}
    moe = by_name[_REAL_MOE_NAME]
    # Categories list-repr -> bare label.
    assert moe["category"] == "MoE_fused"
    # total_direct_kernel_time_ms (ms) scaled to microseconds.
    assert moe["gpu_us"] == pytest.approx(302.87590625 * 1000.0)
    # Percentage (%) column parsed directly.
    assert moe["gpu_pct"] == pytest.approx(67.4438834, abs=1e-4)
    assert by_name["aten::mm"]["category"] == "GEMM"


def test_clean_category_label_handles_list_repr():
    assert tla._clean_category_label("['MoE_fused']") == "MoE_fused"
    assert tla._clean_category_label("['GEMM', 'Reduce']") == "GEMM"
    assert tla._clean_category_label("MoE_fused") == "MoE_fused"
    assert tla._clean_category_label("[]") == ""
    assert tla._clean_category_label("") == ""


def test_recover_moe_fused_real_schema(tmp_path):
    """The dominant MoE_fused row (67% GPU time) is recovered from the REAL
    ops_summary.csv schema even though it is NOT an "other"-bucket op."""
    _write(tmp_path / "ops_summary.csv", _real_ops_summary_csv())
    # analysis.md surfaced the GEMM; the 67% MoE_fused kernel had no block.
    recovered = tla.recover_other_bucket_candidates(
        tmp_path, [{"name": "aten::mm"}], top_k=10,
    )
    assert [c["name"] for c in recovered] == [_REAL_MOE_NAME]
    cand = recovered[0]
    assert cand["tracelens_category"] == "MoE_fused"
    assert cand["gpu_pct"] == pytest.approx(67.4438834, abs=1e-3)
    assert cand["candidate_source"] == "other_bucket_fallback"


def test_compound_subwindow_keywords_extracts_function():
    """The embedded function symbol is recoverable from the profiler-wrapped
    op name so source resolution can grep for it."""
    windows = tla._compound_subwindow_keywords(_REAL_MOE_NAME)
    assert "invoke_fused_moe_kernel" in windows
    # The full compound token (which never appears verbatim in source) is not
    # the only keyword we try.
    assert any("fused_moe_kernel" in w for w in windows)


def _make_fake_sglang_tree(root: Path) -> Path:
    """Create a minimal sglang-like source tree with the fused-MoE kernel."""
    src = (
        root / "sglang" / "srt" / "layers" / "moe" / "moe_runner"
        / "triton_utils" / "fused_moe.py"
    )
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        "import triton\n"
        "import triton.language as tl\n\n"
        "def invoke_fused_moe_kernel(a, b, c):\n"
        "    # launches the fused-MoE expert grouped-GEMM Triton kernel\n"
        "    return None\n",
        encoding="utf-8",
    )
    return src


def test_locate_source_resolves_profiler_wrapped_name(tmp_path, monkeypatch):
    """Fix 2(c): locate_source_via_grep resolves a profiler-wrapped op name to
    its kernel source via trailing sub-window keywords (hermetic — uses a fake
    source tree, not /sgl-workspace)."""
    src = _make_fake_sglang_tree(tmp_path / "src")
    monkeypatch.setattr(tla, "KNOWN_SEARCH_ROOTS", (str(tmp_path / "src"),))
    tla._GREP_CACHE.clear()
    resolved = tla.locate_source_via_grep(_REAL_MOE_NAME)
    assert resolved == str(src)


def test_recovered_moe_fused_real_schema_routes_to_geak(tmp_path, monkeypatch):
    """End-to-end (hermetic): real-schema MoE_fused row -> recovered ->
    _finalize_candidates -> source resolved -> reusable_native_kernel=True."""
    _write(tmp_path / "ops_summary.csv", _real_ops_summary_csv())
    src = _make_fake_sglang_tree(tmp_path / "src")
    fake_root = str(tmp_path / "src") + "/"
    monkeypatch.setattr(tla, "KNOWN_SEARCH_ROOTS", (str(tmp_path / "src"),))
    monkeypatch.setattr(tla, "_reusable_roots", lambda: (fake_root,))
    tla._GREP_CACHE.clear()

    recovered = tla.recover_other_bucket_candidates(
        tmp_path, [{"name": "aten::mm"}], top_k=10,
    )
    finalized = tla._finalize_candidates(recovered, total_dur=None)
    item = finalized[0]
    assert item["name"] == _REAL_MOE_NAME
    assert item["source_file"] == str(src)
    # Triton .py under a reusable root -> routable to GEAK.
    assert item["source_type"] == "triton"
    assert item["reusable_native_kernel"] is True, item.get("skip_reason")
