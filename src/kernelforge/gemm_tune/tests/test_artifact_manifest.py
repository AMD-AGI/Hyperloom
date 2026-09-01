# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the TuningArtifactManifest (WP-4)."""

from __future__ import annotations

import json

import pytest

from kernelforge.gemm_tune.artifact_manifest import (
    TUNING_ARTIFACT_SCHEMA_VERSION,
    build_artifact_manifest,
    write_artifact_manifest,
)
from kernelforge.gemm_tune.report import TuneReport
from kernelforge.gemm_tune.shape_manifest import MANIFEST_KIND
from kernelforge.gemm_tune.tuners.base import TuneResult


def _manifest_dict() -> dict:
    return {
        "schema_version": 1,
        "manifest_kind": MANIFEST_KIND,
        "manifest_hash": "cafef00d",
        "generated_from": {
            "tracelens_revision": "tl-1.2.3",
            "main_trace_hash": "abc123",
            "capture_trace_hashes": {"bs_512_piecewise": "deadbeef"},
        },
        "workload": {
            "total_target_gemm_us": 12498.0,
            "variant_steady_replay": {"eager": 1},
        },
        "rows": [
            {
                "dims": {"M": 8192, "N": 5120, "K": 5120},
                "is_target_gemm": True,
                "cum_gpu_us": 7995.0,
                "capture_only": False,
                "graph_variant": "eager",
                "in_dtype": "fp8",
            },
            {
                "dims": {"M": 8192, "N": 34816, "K": 5120},
                "is_target_gemm": True,
                "cum_gpu_us": 3503.0,
                "capture_only": False,
                "graph_variant": "eager",
                "in_dtype": "fp8",
            },
            {
                "dims": {"M": 1, "N": 5120, "K": 5120},
                "is_target_gemm": True,
                "cum_gpu_us": 1000.0,
                "capture_only": False,
                "graph_variant": "eager",
                "in_dtype": "fp8",
            },
        ],
    }


def _candidate_result(csv_path: str) -> TuneResult:
    # o_proj + gate_up improved; (1,5120,5120) NOT improved -> not covered.
    return TuneResult(
        tuner_name="a8w8_blockscale",
        status="ok",
        artifact_path=csv_path,
        env_var="AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
        env_value=csv_path,
        total_shapes=3,
        improved_shapes=2,
        best_micro_speedup=4.75,
        avg_micro_speedup=4.3,
        shape_results=[
            {
                "M": 8192,
                "N": 5120,
                "K": 5120,
                "default_us": 1037.7,
                "tuned_us": 271.9,
                "speedup": 3.82,
                "improved": True,
            },
            {
                "M": 8192,
                "N": 34816,
                "K": 5120,
                "default_us": 6955.5,
                "tuned_us": 1464.5,
                "speedup": 4.75,
                "improved": True,
            },
            {"M": 1, "N": 5120, "K": 5120, "default_us": 50.0, "tuned_us": 49.0, "speedup": 1.02, "improved": False},
        ],
    )


def _report() -> TuneReport:
    return TuneReport(
        status="ok",
        micro_decision="candidate",
        requires_e2e_validation=True,
        recommended_env={"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "/tmp/c.csv"},
        finished_at="2026-07-23T00:00:00Z",
    )


def test_schema_and_provenance(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("M,N,K\n8192,5120,5120\n")
    mf = tmp_path / "m.json"
    mf.write_text(json.dumps(_manifest_dict()))
    am = build_artifact_manifest(
        _report(),
        [_candidate_result(str(csv))],
        shape_manifest_path=mf,
        gpu_type="mi355x",
        framework="vllm-aiter",
        precision="fp8",
        quant_type="blockscale",
        tp=1,
        generated_at="2026-07-23T00:00:00Z",
    )
    assert am["schema_version"] == TUNING_ARTIFACT_SCHEMA_VERSION
    assert am["provenance"]["gpu_type"] == "mi355x"
    assert am["micro_decision"] == "candidate"
    # source manifest linkage
    src = am["source_manifest"]
    assert src["present"] is True
    assert src["manifest_hash"] == "cafef00d"
    assert src["main_trace_hash"] == "abc123"
    assert src["tracelens_revision"] == "tl-1.2.3"
    # per-tuner csv hash present
    assert am["tuners"][0]["csv_sha256"]  # non-empty sha256 of the real file
    assert am["invalidation"]["keys"]


def test_weighted_coverage(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("x")
    mf = tmp_path / "m.json"
    mf.write_text(json.dumps(_manifest_dict()))
    am = build_artifact_manifest(_report(), [_candidate_result(str(csv))], shape_manifest_path=mf)
    cov = am["coverage"]
    # improved: 7995 + 3503 = 11498 covered; total 12498 -> 0.9200
    assert cov["covered_target_weight"] == pytest.approx(11498.0)
    assert cov["total_target_weight"] == pytest.approx(12498.0)
    assert cov["shape_coverage_factor"] == pytest.approx(round(11498.0 / 12498.0, 4))
    assert cov["improved_shape_count"] == 2
    assert cov["target_shape_count"] == 3


def test_no_source_manifest_coverage_null():
    am = build_artifact_manifest(_report(), [], shape_manifest_path=None)
    assert am["source_manifest"] == {"present": False}
    assert am["coverage"]["shape_coverage_factor"] is None


def test_invalid_manifest_degrades(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"manifest_kind": "nope"}))
    am = build_artifact_manifest(_report(), [], shape_manifest_path=bad)
    assert am["source_manifest"]["present"] is False
    assert am["source_manifest"]["error"] == "unreadable_or_invalid"


def test_write_artifact_manifest(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("M,N,K\n")
    mf = tmp_path / "m.json"
    mf.write_text(json.dumps(_manifest_dict()))
    out = write_artifact_manifest(
        _report(),
        [_candidate_result(str(csv))],
        tmp_path,
        shape_manifest_path=mf,
    )
    assert out.name == "tuning_artifact_manifest.json"
    data = json.loads(out.read_text())
    assert data["tool"] == "forge-gemm-tune"
    json.dumps(data)  # serializable
