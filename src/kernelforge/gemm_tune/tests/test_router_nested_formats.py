# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A precision that only names the quantization toolkit routes on the formats the checkpoint nests."""

from kernelforge.gemm_tune.model_analyzer import ModelProfile
from kernelforge.gemm_tune.router import select_tuners


def _quark_moe_profile(layer_quant_config=None):
    """A Quark MXFP4 MoE checkpoint, optionally overriding some layers' format, shaped like GLM-5.3-MXFP4."""
    quant_cfg = {
        "quant_method": "quark",
        "global_quant_config": {"weight": {"dtype": "fp4", "group_size": 32}},
        "exclude": ["lm_head"],
    }
    if layer_quant_config:
        quant_cfg["layer_quant_config"] = layer_quant_config
    return ModelProfile(
        model_path="/fake/GLM-5.3-MXFP4",
        is_moe=True,
        num_experts=256,
        num_experts_per_tok=8,
        hidden_size=6144,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        num_hidden_layers=78,
        num_attention_heads=64,
        num_key_value_heads=64,
        quant_method="quark",
        raw_config={"quantization_config": quant_cfg},
    )


def _fp8_attention():
    return {
        f"model.layers.{layer}.self_attn.{proj}": {"weight": {"dtype": "fp8_e4m3", "group_size": 128}}
        for layer in range(3)
        for proj in ("q_a_proj", "kv_b_proj", "o_proj")
    }


def _routed(profile, precision, tmp_path, quant_type="auto"):
    log = tmp_path / "kernels.log"
    log.write_text("gemm_a8w8_blockscale_bpreshuffle\n")
    return select_tuners(
        profile,
        framework="sglang",
        precision=precision,
        quant_type=quant_type,
        gpu_type="mi355x",
        kernel_signature_log=str(log),
    )


def _running(specs):
    return {s.name for s in specs if not s.skip_reason}


def test_mxfp4_experts_and_fp8_attention_each_get_their_tuner(tmp_path):
    specs = _routed(_quark_moe_profile(_fp8_attention()), "quark", tmp_path)
    running = _running(specs)
    assert "fmoe_ck" in running
    assert "a8w8_blockscale_bpreshuffle" in running
    # The dense layers the checkpoint keeps fp8 are not fp4 GEMMs.
    assert "a4w4_blockscale" not in {s.name for s in specs}


def test_the_toolkit_name_routes_like_the_format_it_wraps(tmp_path):
    profile = _quark_moe_profile()
    as_toolkit = [(s.name, s.skip_reason) for s in _routed(profile, "quark", tmp_path)]
    as_format = [(s.name, s.skip_reason) for s in _routed(profile, "mxfp4", tmp_path)]
    assert as_toolkit == as_format


def test_a_format_precision_is_trusted_over_the_nested_formats(tmp_path):
    specs = _routed(_quark_moe_profile(_fp8_attention()), "mxfp4", tmp_path)
    running = _running(specs)
    assert {"fmoe_ck", "a4w4_blockscale"} <= running
    assert not any(name.startswith("a8w8") for name in running)


def test_compressed_tensors_fp8_reaches_the_moe_and_fp8_tuners(tmp_path):
    profile = ModelProfile(
        model_path="/fake/Qwen3-30B-A3B-FP8-dynamic",
        is_moe=True,
        num_experts=128,
        num_experts_per_tok=8,
        hidden_size=2048,
        intermediate_size=6144,
        moe_intermediate_size=768,
        quant_method="compressed-tensors",
        raw_config={
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {"targets": ["Linear"], "weights": {"num_bits": 8, "type": "float"}},
                },
            }
        },
    )
    running = _running(_routed(profile, "compressed-tensors", tmp_path))
    assert {"fmoe_ck", "a8w8_blockscale_bpreshuffle"} <= running
