###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the variant-discriminating TraceShapeManifest producer.

Covers the frozen P0-A / WP-1 contract with synthetic ``analyze_trace``-shaped
inputs (no GPU, no trace files needed) plus one reader->producer end-to-end
check that the ``kernel_launches`` enrichment feeds the manifest.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _bypass_trace_reader as reader  # noqa: E402
import _trace_shape_manifest as tsm  # noqa: E402


def _launch(name, *, op_name="", ts=0.0, dur=100.0, shapes=None, dtypes=None, backend="", kfile=""):
    """Build one enriched kernel_launches record (reader output shape)."""
    return {
        "name": name,
        "op_name": op_name,
        "ts": ts,
        "dur": dur,
        "shapes": shapes or [],
        "dtypes": dtypes or [],
        "kernel_file": kfile,
        "kernel_backend": backend,
        "correlation": None,
    }


def _analysis(launches, *, kernels=None, steady_window=None, scope="steady_state"):
    """Build a synthetic analyze_trace result dict."""
    return {
        "status": "ok",
        "trace_file": "/tmp/fake.trace.json.gz",
        "kernel_launches": launches,
        "kernels": kernels or [],
        "steady_window": steady_window,
        "aggregation_scope": scope,
    }


# --- op classification / tags -------------------------------------------------


def test_classify_op_covers_common_families():
    assert tsm.classify_op("Cijk_Alik_Bljk_HHS", "aten::mm") == "gemm"
    assert tsm.classify_op("fused_moe_kernel", "") == "moe"
    assert tsm.classify_op("paged_attention_v1", "") == "attention"
    assert tsm.classify_op("rms_norm_kernel", "") == "norm"
    assert tsm.classify_op("some_unknown_thing", "") == "other"


def test_dual_gemm_tags():
    # bf16 GEMM -> both tags true.
    r_bf16 = tsm.build_row(
        _launch("gemm_bf16", op_name="aten::mm", dtypes=["c10::BFloat16", "c10::BFloat16"]),
        graph_variant="eager", node_ordinal=0, phase="decode", bucket="eager", capture_only=False,
    )
    assert r_bf16["is_gemm"] is True and r_bf16["is_target_gemm"] is True
    # GEMM with no dtype info -> gemm but NOT target (unaddressable == uncovered).
    r_nodtype = tsm.build_row(
        _launch("gemm_unknown", op_name="aten::mm"),
        graph_variant="eager", node_ordinal=1, phase="decode", bucket="eager", capture_only=False,
    )
    assert r_nodtype["is_gemm"] is True and r_nodtype["is_target_gemm"] is False
    # Non-GEMM -> neither.
    r_norm = tsm.build_row(
        _launch("rms_norm", op_name="", dtypes=["c10::BFloat16"]),
        graph_variant="eager", node_ordinal=2, phase="decode", bucket="eager", capture_only=False,
    )
    assert r_norm["is_gemm"] is False and r_norm["is_target_gemm"] is False


def test_dims_extraction_from_shapes():
    r = tsm.build_row(
        _launch("gemm", op_name="aten::mm", shapes=[[128, 4096], [4096, 8192]], dtypes=["fp8"]),
        graph_variant="eager", node_ordinal=0, phase="decode", bucket="eager", capture_only=False,
    )
    assert r["dims"] == {"M": 128, "N": 8192, "K": 4096, "batch": None, "groups": None}


# --- signature discrimination -------------------------------------------------


def test_signature_discriminates_variant():
    """Same math shape + same ordinal but different graph_variant -> distinct."""
    launch = _launch("gemm", op_name="aten::mm", shapes=[[16, 4096], [4096, 4096]], dtypes=["bf16"])
    r16 = tsm.build_row(launch, graph_variant="bs_16", node_ordinal=0, phase="decode", bucket="bs_16", capture_only=True)
    r32 = tsm.build_row(launch, graph_variant="bs_32", node_ordinal=0, phase="decode", bucket="bs_32", capture_only=True)
    assert r16["signature_key"] != r32["signature_key"]


def test_node_ordinal_distinguishes_same_kernel():
    """Two launches of an identical kernel at different ordinals -> two rows."""
    launches = [
        _launch("gemm", op_name="aten::mm", ts=1.0, shapes=[[16, 4096], [4096, 4096]], dtypes=["bf16"]),
        _launch("gemm", op_name="aten::mm", ts=2.0, shapes=[[16, 4096], [4096, 4096]], dtypes=["bf16"]),
    ]
    rows = tsm.build_variant_rows(
        graph_variant="bs_16", analysis=_analysis(launches), phase="decode", bucket="bs_16", capture_only=True
    )
    assert len(rows) == 2
    assert {r["node_ordinal"] for r in rows} == {0, 1}


# --- multi-variant manifest ---------------------------------------------------


def test_multi_variant_not_merged_and_replay_unresolved():
    launch = _launch("gemm", op_name="aten::mm", shapes=[[16, 4096], [4096, 4096]], dtypes=["bf16"])
    cap16 = _analysis([launch])
    cap32 = _analysis([launch])
    manifest = tsm.build_shape_manifest(
        main_analysis=_analysis([], kernels=[{"name": "hipGraphLaunch", "count": 118}]),
        capture_variants=[("bs_16", cap16), ("bs_32", cap32)],
        provenance={"_provenance_source": "wp1_stub"},
        main_trace_hash="deadbeef",
    )
    variants = {r["graph_variant"] for r in manifest["rows"]}
    assert variants == {"bs_16", "bs_32"}
    assert "multi_variant_replay_unresolved" in manifest["warnings"]
    assert manifest["workload"]["variant_steady_replay"] == {"bs_16": None, "bs_32": None}
    # capture-derived rows must be flagged capture_only (structure, not savings).
    assert all(r["capture_only"] for r in manifest["rows"])


def test_single_variant_replay_attributed():
    launch = _launch("gemm", op_name="aten::mm", shapes=[[16, 4096], [4096, 4096]], dtypes=["bf16"])
    manifest = tsm.build_shape_manifest(
        main_analysis=_analysis([], kernels=[{"name": "hipGraphLaunch", "count": 118}]),
        capture_variants=[("bs_16", _analysis([launch]))],
        provenance={},
        main_trace_hash="abc",
    )
    assert manifest["workload"]["variant_steady_replay"] == {"bs_16": 118}
    assert manifest["workload"]["total_graph_replays"] == 118


def test_eager_fallback_when_no_capture():
    launches = [
        _launch("gemm", op_name="aten::mm", ts=1.0, dur=200.0, shapes=[[1, 4096], [4096, 4096]], dtypes=["bf16"]),
        _launch("rms_norm", op_name="", ts=2.0, dur=50.0, dtypes=["bf16"]),
    ]
    manifest = tsm.build_shape_manifest(
        main_analysis=_analysis(launches),
        capture_variants=[],
        provenance={},
        main_trace_hash="x",
    )
    assert {r["graph_variant"] for r in manifest["rows"]} == {"eager"}
    assert all(r["capture_only"] is False for r in manifest["rows"])
    assert manifest["workload"]["variant_steady_replay"] == {"eager": 1}
    # totals: one target GEMM (200us) out of 250us total.
    assert manifest["workload"]["total_target_gemm_us"] == 200.0
    assert manifest["workload"]["total_gpu_kernel_us"] == 250.0


# --- manifest hash & schema ---------------------------------------------------


def test_manifest_hash_deterministic_and_sensitive():
    launch = _launch("gemm", op_name="aten::mm", shapes=[[16, 4096], [4096, 4096]], dtypes=["bf16"])
    m1 = tsm.build_shape_manifest(
        main_analysis=_analysis([]), capture_variants=[("bs_16", _analysis([launch]))],
        provenance={}, main_trace_hash="h",
    )
    m2 = tsm.build_shape_manifest(
        main_analysis=_analysis([]), capture_variants=[("bs_16", _analysis([launch]))],
        provenance={}, main_trace_hash="h",
    )
    assert m1["manifest_hash"] == m2["manifest_hash"]
    other = _launch("gemm", op_name="aten::mm", shapes=[[32, 4096], [4096, 4096]], dtypes=["bf16"])
    m3 = tsm.build_shape_manifest(
        main_analysis=_analysis([]), capture_variants=[("bs_16", _analysis([other]))],
        provenance={}, main_trace_hash="h",
    )
    assert m3["manifest_hash"] != m1["manifest_hash"]


def test_manifest_schema_shape():
    manifest = tsm.build_shape_manifest(
        main_analysis=_analysis([]), capture_variants=[], provenance={"_provenance_source": "wp1_stub"},
        main_trace_hash="h", generated_at="2026-07-23T00:00:00Z",
    )
    assert manifest["schema_version"] == tsm.SCHEMA_VERSION
    assert manifest["manifest_kind"] == tsm.MANIFEST_KIND
    assert manifest["generated_from"]["generated_at"] == "2026-07-23T00:00:00Z"
    assert "manifest_hash" in manifest
    # manifest must be JSON-serializable.
    json.dumps(manifest)


def test_empty_launches_produce_no_rows():
    manifest = tsm.build_shape_manifest(
        main_analysis=_analysis([]), capture_variants=[("bs_16", _analysis([]))],
        provenance={}, main_trace_hash="h",
    )
    assert manifest["rows"] == []


# --- reader -> producer end-to-end (enrichment present) -----------------------


def test_reader_enriches_launches_and_feeds_producer(tmp_path):
    """A tiny real trace flows through the reader and its enriched launches build
    a manifest with resolved shapes/dtypes (proves the additive reader change)."""
    events = [
        {"cat": "cpu_op", "name": "aten::mm", "args": {"External id": 100, "Input Dims": [[64, 4096], [4096, 4096]], "Input type": ["c10::BFloat16", "c10::BFloat16"]}},
        {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 5, "External id": 100}},
        {"cat": "kernel", "ph": "X", "name": "Cijk_gemm_bf16", "ts": 1000, "dur": 200, "args": {"correlation": 5}},
    ]
    trace = tmp_path / "rank_0.trace.json.gz"
    with gzip.open(trace, "wt", encoding="utf-8") as fh:
        json.dump({"traceEvents": events}, fh)

    analysis = reader.analyze_trace(trace, top_k=0, emit_launches=True)
    launches = analysis["kernel_launches"]
    assert launches and launches[0]["shapes"] == [[64, 4096], [4096, 4096]]
    assert launches[0]["dtypes"] == ["c10::BFloat16", "c10::BFloat16"]

    manifest = tsm.build_shape_manifest(
        main_analysis=analysis, capture_variants=[], provenance={}, main_trace_hash="h",
    )
    row = next(r for r in manifest["rows"] if r["op"] == "gemm")
    assert row["dims"] == {"M": 64, "N": 4096, "K": 4096, "batch": None, "groups": None}
    assert row["is_target_gemm"] is True
