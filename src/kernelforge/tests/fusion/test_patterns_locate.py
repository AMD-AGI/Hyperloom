# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for pattern matching, shape resolution, recipe assembly, manifest."""

from __future__ import annotations

import json

from kernelforge.fusion.diagnose import diagnose_from_shares
from kernelforge.fusion.locate import build_recipes
from kernelforge.fusion.patterns import match_patterns
from kernelforge.fusion.report import build_manifest
from kernelforge.fusion.shapes import resolve_decode_shapes


def _candidate_diag(shares, busy=0.21):
    return diagnose_from_shares(shares, busy_fraction_of_wall=busy)


class TestMatchPatterns:
    def test_residual_add_rmsnorm_triggers(self):
        # launch_bound_share = 0.14+0.10+0.05 = 0.29 >= 0.25 -> candidate.
        d = _candidate_diag({"gemm": 0.5, "add": 0.14, "rmsnorm": 0.10, "activation": 0.05})
        matched = match_patterns(d, "sglang")
        ids = [p.id for p, _ in matched]
        assert "residual_add_rmsnorm" in ids
        # ranked by trigger share (add+rmsnorm = 0.22 is the strongest here).
        assert matched[0][0].id == "residual_add_rmsnorm"

    def test_qk_norm_rope_triggers(self):
        d = _candidate_diag({"gemm": 0.5, "rmsnorm": 0.12, "rope": 0.06, "add": 0.09})
        ids = [p.id for p, _ in match_patterns(d, "sglang")]
        assert "qk_norm_rope" in ids

    def test_non_candidate_yields_nothing(self):
        d = diagnose_from_shares({"gemm": 0.8, "attention": 0.15, "add": 0.02}, busy_fraction_of_wall=0.72)
        assert match_patterns(d, "sglang") == []

    def test_framework_filter(self):
        d = _candidate_diag({"add": 0.2, "rmsnorm": 0.1})
        assert match_patterns(d, "sglang")  # sglang in every pattern's frameworks
        assert match_patterns(d, "tensorrt") == []  # unknown framework -> nothing


class TestResolveShapes:
    def test_reads_config(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "lfm2",
                    "hidden_size": 2048,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "intermediate_size": 8192,
                    "rms_norm_eps": 1e-5,
                }
            ),
            encoding="utf-8",
        )
        s = resolve_decode_shapes(tmp_path, decode_batch=16)
        assert s["model_type"] == "lfm2"
        assert s["hidden_size"] == 2048
        assert s["head_dim"] == 128  # 2048 / 16
        assert s["gqa_groups"] == 2  # 16 / 8
        assert s["T"] == 16

    def test_missing_config_safe(self, tmp_path):
        s = resolve_decode_shapes(tmp_path)
        assert s["model_type"] == "" and s["T"] == 16


class TestBuildRecipesAndManifest:
    def test_build_recipes_for_candidate(self, tmp_path):
        # Synthetic model_type -> source unresolved (hermetic; no source filtering).
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "toylm",
                    "hidden_size": 2048,
                    "num_attention_heads": 16,
                }
            ),
            encoding="utf-8",
        )
        d = _candidate_diag({"gemm": 0.45, "add": 0.18, "rmsnorm": 0.12})
        recipes = build_recipes(d, model_path=str(tmp_path), framework="sglang")
        assert recipes
        residual = next((r for r in recipes if r.pattern_id == "residual_add_rmsnorm"), None)
        assert residual is not None
        assert residual.shapes["hidden_size"] == 2048
        assert residual.env_flag == "TOYLM_FUSED_RESIDUAL"  # model-prefixed
        assert residual.source_confirmed is None  # source unresolved

    def test_manifest_verdict_candidate(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "toylm", "hidden_size": 2048, "num_attention_heads": 16}), encoding="utf-8"
        )
        d = _candidate_diag({"gemm": 0.45, "add": 0.18, "rmsnorm": 0.12})
        recipes = build_recipes(d, model_path=str(tmp_path), framework="sglang")
        m = build_manifest(
            framework="sglang",
            model_path=str(tmp_path),
            model_type="toylm",
            diagnosis=d,
            recipe=recipes[0],
            candidates=recipes,
        )
        assert m["verdict"] == "candidate"
        assert m["fusion"]["pattern"] == "residual_add_rmsnorm"
        assert m["validation"] is None and m["artifacts"] is None

    def test_build_recipes_empty_for_non_candidate(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
        d = diagnose_from_shares({"gemm": 0.8, "attention": 0.15}, busy_fraction_of_wall=0.72)
        assert build_recipes(d, model_path=str(tmp_path), framework="sglang") == []

    def test_manifest_verdict_no_opportunity(self, tmp_path):
        d = diagnose_from_shares({"gemm": 0.8, "attention": 0.15}, busy_fraction_of_wall=0.72)
        m = build_manifest(framework="sglang", model_path=str(tmp_path), model_type="qwen3", diagnosis=d, recipe=None)
        assert m["verdict"] == "no_opportunity"
        assert m["fusion_candidates"] == []
