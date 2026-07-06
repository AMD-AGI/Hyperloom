###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the bypass fusion-opportunity analysis (_bypass_fusion)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bypass_fusion import adjacent_pairs, analyze_fusion, fusable_clusters  # noqa: E402


def _seq():
    # GEMM | (Norm, Elem, Elem) | GEMM | (Elem, Quant)
    return [
        {"name": "mm", "op_name": "aten::mm", "category": "GEMM", "ts": 0.0, "dur": 10.0},
        {"name": "rms", "op_name": "aiter::rmsnorm", "category": "Normalization", "ts": 10.0, "dur": 2.0},
        {"name": "add", "op_name": "aten::add", "category": "Elementwise", "ts": 12.5, "dur": 1.0},
        {"name": "mul", "op_name": "aten::mul", "category": "Elementwise", "ts": 13.5, "dur": 1.0},
        {"name": "mm2", "op_name": "aten::mm", "category": "GEMM", "ts": 15.0, "dur": 10.0},
        {"name": "silu", "op_name": "aten::silu", "category": "Elementwise", "ts": 25.0, "dur": 1.0},
        {"name": "q", "op_name": "aiter::quant", "category": "Quantization", "ts": 26.0, "dur": 1.0},
    ]


def test_clusters_group_consecutive_fusable_runs():
    clusters = fusable_clusters(_seq())
    assert len(clusters) == 2
    # ranked by aggregate time: (rms+add+mul = 4us) before (silu+q = 2us).
    top = clusters[0]
    assert top["launch_count"] == 3
    assert top["aggregate_dur_us"] == 4.0
    assert set(top["categories"]) == {"Normalization", "Elementwise"}
    assert clusters[1]["launch_count"] == 2
    assert set(clusters[1]["categories"]) == {"Elementwise", "Quantization"}


def test_lone_fusable_kernel_is_not_a_cluster():
    seq = [
        {"name": "mm", "category": "GEMM", "ts": 0.0, "dur": 5.0},
        {"name": "add", "category": "Elementwise", "ts": 5.0, "dur": 1.0},  # lone
        {"name": "mm2", "category": "GEMM", "ts": 6.0, "dur": 5.0},
    ]
    assert fusable_clusters(seq) == []


def test_cluster_reports_inter_kernel_gap():
    # two elementwise with a 0.5us gap between them -> gap surfaced.
    seq = [
        {"name": "a", "category": "Elementwise", "ts": 0.0, "dur": 1.0},
        {"name": "b", "category": "Elementwise", "ts": 1.5, "dur": 1.0},
    ]
    c = fusable_clusters(seq)[0]
    assert c["aggregate_dur_us"] == 2.0
    assert c["span_us"] == 2.5
    assert c["inter_kernel_gap_us"] == 0.5


def test_adjacent_pairs_counts_transitions():
    pairs = adjacent_pairs(_seq())
    d = {(p["from"], p["to"]): p["count"] for p in pairs}
    assert d[("Elementwise", "Elementwise")] == 1  # add -> mul
    assert d[("GEMM", "Normalization")] == 1
    assert all(p["count"] >= 1 for p in pairs)


def test_analyze_fusion_payload():
    out = analyze_fusion(_seq())
    assert out["launch_count"] == 7
    assert out["fusable_cluster_count"] == 2
    assert out["fusable_time_us"] == 6.0  # 4 + 2
    assert out["adjacent_pairs"]
    assert out["fusable_clusters"][0]["launch_count"] == 3


def test_fusable_time_and_count_cover_all_clusters_not_just_top_k():
    # 3 fusable clusters (2 Elementwise each), separated by GEMM. With top_k=1 the
    # LIST is capped to 1, but the total count/time must reflect ALL 3 clusters.
    seq = []
    ts = 0.0
    for i in range(3):
        seq.append({"name": f"mm{i}", "category": "GEMM", "ts": ts, "dur": 5.0}); ts += 5
        seq.append({"name": f"a{i}", "category": "Elementwise", "ts": ts, "dur": 1.0 + i}); ts += 1 + i
        seq.append({"name": f"b{i}", "category": "Elementwise", "ts": ts, "dur": 1.0 + i}); ts += 1 + i
    out = analyze_fusion(seq, top_k_clusters=1)
    assert out["fusable_cluster_count"] == 3          # total, not truncated
    assert len(out["fusable_clusters"]) == 1          # list capped to top_k
    assert out["fusable_time_us"] == round(sum((1.0 + i) * 2 for i in range(3)), 3)  # 2+4+6=12


def test_empty_and_all_nonfusable():
    assert analyze_fusion([])["fusable_cluster_count"] == 0
    only_gemm = [{"name": "mm", "category": "GEMM", "ts": 0.0, "dur": 5.0}]
    assert fusable_clusters(only_gemm) == []
