# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for model_analyzer module."""

import json
import pytest

from kernelforge.gemm_tune.model_analyzer import analyze_model, _extract_quant_info


class TestExtractQuantInfo:
    def test_empty_config(self):
        assert _extract_quant_info({}) == ("", 0, 0)

    def test_none_quantization_config(self):
        assert _extract_quant_info({"quantization_config": None}) == ("", 0, 0)

    def test_non_dict_quantization_config(self):
        assert _extract_quant_info({"quantization_config": "invalid"}) == ("", 0, 0)

    def test_null_bits_and_group_size(self):
        config = {"quantization_config": {"bits": None, "group_size": None}}
        assert _extract_quant_info(config) == ("", 0, 0)

    def test_string_bits(self):
        config = {"quantization_config": {"bits": "8", "group_size": "128"}}
        assert _extract_quant_info(config) == ("", 8, 128)

    def test_awq_config(self):
        config = {"quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128}}
        assert _extract_quant_info(config) == ("awq", 4, 128)

    def test_gptq_config(self):
        config = {"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 32}}
        assert _extract_quant_info(config) == ("gptq", 4, 32)

    def test_compressed_tensors(self):
        config = {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {"group_0": {"weights": {"num_bits": 8, "group_size": 32}}},
            }
        }
        assert _extract_quant_info(config) == ("compressed-tensors", 8, 32)

    def test_compressed_tensors_null_fields(self):
        config = {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {"group_0": {"weights": {"num_bits": None, "group_size": None}}},
            }
        }
        assert _extract_quant_info(config) == ("compressed-tensors", 0, 0)

    def test_invalid_bits_type(self):
        config = {"quantization_config": {"bits": "not_a_number"}}
        assert _extract_quant_info(config) == ("", 0, 0)


class TestAnalyzeModel:
    def test_missing_config(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            analyze_model(str(tmp_path / "nonexistent"))

    def test_invalid_json(self, tmp_path):
        (tmp_path / "config.json").write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            analyze_model(str(tmp_path))

    def test_minimal_config(self, tmp_path):
        config = {"hidden_size": 4096, "intermediate_size": 11008}
        (tmp_path / "config.json").write_text(json.dumps(config))
        profile = analyze_model(str(tmp_path))
        assert profile.hidden_size == 4096
        assert profile.intermediate_size == 11008
        assert profile.is_moe is False

    def test_moe_model(self, tmp_path):
        config = {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_local_experts": 128,
            "num_experts_per_tok": 8,
            "hidden_act": "silu",
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        profile = analyze_model(str(tmp_path))
        assert profile.is_moe is True
        assert profile.num_experts == 128
        assert profile.num_experts_per_tok == 8
        assert profile.moe_intermediate_size == 768
        assert profile.activation_type_str == "ActivationType.Silu"

    def test_num_experts_field_variant(self, tmp_path):
        config = {"hidden_size": 2048, "num_experts": 64, "num_experts_per_tok": 4}
        (tmp_path / "config.json").write_text(json.dumps(config))
        profile = analyze_model(str(tmp_path))
        assert profile.is_moe is True
        assert profile.num_experts == 64
