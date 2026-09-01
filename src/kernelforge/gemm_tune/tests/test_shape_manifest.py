# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for kernelforge.gemm_tune.shape_manifest (TraceShapeManifest consumer)."""

from __future__ import annotations

import json

import pytest

from kernelforge.gemm_tune.shape_manifest import (
    MANIFEST_KIND,
    load_manifest,
    manifest_to_shapes,
    write_manifest_untuned_csv,
)


def _manifest() -> dict:
    """A synthetic TraceShapeManifest with Qwen3-14B-like FP8 GEMM rows."""
    return {
        "schema_version": 1,
        "manifest_kind": MANIFEST_KIND,
        "manifest_hash": "deadbeef",
        "workload": {
            "total_gpu_kernel_us": 14000.0,
            "total_gemm_us": 12943.0,
            "total_target_gemm_us": 12943.0,
            "variant_steady_replay": {"bs_512_piecewise": 100, "eager": 1},
        },
        "rows": [
            {
                "dims": {"M": 8192, "N": 5120, "K": 5120},
                "is_gemm": True,
                "is_target_gemm": True,
                "cum_gpu_us": 6954.0,
                "capture_only": False,
                "graph_variant": "eager",
                "in_dtype": "c10::Float8_e4m3fn",
            },
            {
                "dims": {"M": 8192, "N": 34816, "K": 5120},
                "is_gemm": True,
                "is_target_gemm": True,
                "cum_gpu_us": 3503.0,
                "capture_only": False,
                "graph_variant": "eager",
                "in_dtype": "c10::Float8_e4m3fn",
            },
            # dedups with row 0 (same M,N,K) -> weights sum
            {
                "dims": {"M": 8192, "N": 5120, "K": 5120},
                "is_gemm": True,
                "is_target_gemm": True,
                "cum_gpu_us": 1041.0,
                "capture_only": False,
                "graph_variant": "eager",
                "in_dtype": "c10::Float8_e4m3fn",
            },
            # capture_only -> weight scaled by variant_steady_replay (10 * 100 = 1000)
            {
                "dims": {"M": 1, "N": 5120, "K": 5120},
                "is_gemm": True,
                "is_target_gemm": True,
                "cum_gpu_us": 10.0,
                "capture_only": True,
                "graph_variant": "bs_512_piecewise",
                "in_dtype": "fp8",
            },
            # is_gemm but NOT target -> excluded
            {
                "dims": {"M": 8192, "N": 5120, "K": 5120},
                "is_gemm": True,
                "is_target_gemm": False,
                "cum_gpu_us": 9999.0,
                "capture_only": False,
                "graph_variant": "eager",
                "in_dtype": "bf16",
            },
            # missing M -> dropped (cannot tune)
            {
                "dims": {"M": None, "N": 5120, "K": 5120},
                "is_gemm": True,
                "is_target_gemm": True,
                "cum_gpu_us": 5.0,
                "capture_only": False,
                "graph_variant": "eager",
                "in_dtype": "fp8",
            },
        ],
    }


class TestManifestToShapes:
    def test_dedup_weight_and_order(self):
        shapes = manifest_to_shapes(_manifest())
        # 3 distinct target shapes (non-target + missing-M dropped)
        assert [(s["M"], s["N"], s["K"]) for s in shapes] == [
            (8192, 5120, 5120),  # 6954 + 1041 = 7995 (deduped, highest)
            (8192, 34816, 5120),  # 3503
            (1, 5120, 5120),  # 10 * 100 (capture_only x steady replay) = 1000
        ]
        assert shapes[0]["weight"] == pytest.approx(7995.0)
        assert shapes[2]["weight"] == pytest.approx(1000.0)

    def test_target_only_false_includes_nontarget(self):
        shapes = manifest_to_shapes(_manifest(), target_only=False)
        # now the is_gemm-but-not-target 5120x5120 row folds into that shape too
        assert any((s["M"], s["N"], s["K"]) == (8192, 5120, 5120) for s in shapes)
        # non-target-only distinct count grows vs target_only
        assert len(shapes) >= 3

    def test_top_k_caps(self):
        shapes = manifest_to_shapes(_manifest(), top_k=2)
        assert len(shapes) == 2
        assert (shapes[0]["M"], shapes[0]["N"], shapes[0]["K"]) == (8192, 5120, 5120)

    def test_missing_dims_dropped(self):
        shapes = manifest_to_shapes(_manifest())
        assert all(isinstance(s["M"], int) and s["M"] > 0 for s in shapes)


class TestWriteCsv:
    def test_mnk_csv(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps(_manifest()))
        out = write_manifest_untuned_csv(p, tmp_path)
        lines = out.read_text().splitlines()
        assert lines[0] == "M,N,K"
        assert lines[1] == "8192,5120,5120"  # highest weight first
        assert lines[2] == "8192,34816,5120"
        assert lines[3] == "1,5120,5120"

    def test_mnk_q_dtype_csv(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps(_manifest()))
        out = write_manifest_untuned_csv(p, tmp_path, needs_q_dtype_w=True)
        lines = out.read_text().splitlines()
        assert lines[0] == "M,N,K,q_dtype_w"
        assert lines[1] == "8192,5120,5120,torch.float8_e4m3fnuz"

    def test_no_target_shapes_returns_none(self, tmp_path):
        m = _manifest()
        for r in m["rows"]:
            r["is_target_gemm"] = False
        p = tmp_path / "m.json"
        p.write_text(json.dumps(m))
        assert write_manifest_untuned_csv(p, tmp_path) is None


class TestLoadAndCoverage:
    def test_load_rejects_non_manifest(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"manifest_kind": "something_else", "rows": []}))
        with pytest.raises(ValueError):
            load_manifest(p)

    def test_load_accepts_manifest(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps(_manifest()))
        assert load_manifest(p)["manifest_kind"] == MANIFEST_KIND
