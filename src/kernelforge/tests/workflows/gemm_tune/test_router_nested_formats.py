# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A precision that only names the quantization toolkit routes on what the serving log and the checkpoint say runs."""

from kernelforge.workflows.gemm_tune.model_analyzer import ModelProfile
from kernelforge.workflows.gemm_tune.router import RoutingEvidence, RoutingTarget, select_tuners


_MXFP4 = {"weight": {"dtype": "fp4", "group_size": 32}}


def _quark_moe_profile(global_quant_config=_MXFP4, **layer_configs):
    """A Quark MoE checkpoint shaped like GLM-5.3-MXFP4, optionally overriding some layers' format."""
    quant_cfg = {"quant_method": "quark", "exclude": ["lm_head"], **layer_configs}
    if global_quant_config:
        quant_cfg["global_quant_config"] = global_quant_config
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


def _fp8_moe_dispatch(quant_type):
    return (
        "[aiter] [fused_moe] using 2stage default for ('gfx950', 256, 64, 6144, 2048, 256, 8, <ActivationType.Silu: 0>, "
        f"'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float8_e4m3fn', 'QuantType.{quant_type}', True, False)"
    )


_FP8_PER_TOKEN_MOE_DISPATCH = _fp8_moe_dispatch("per_Token")


def _routed(profile, precision, tmp_path, quant_type="auto", log_line="gemm_a8w8_blockscale_bpreshuffle"):
    log = tmp_path / "kernels.log"
    log.write_text(log_line + "\n")
    return select_tuners(
        profile,
        RoutingTarget(
            framework="sglang",
            precision=precision,
            quant_type=quant_type,
            gpu_type="mi355x",
        ),
        RoutingEvidence(
            kernel_signature_log=str(log),
        ),
    )


def _running(specs):
    return {s.name for s in specs if not s.skip_reason}


def test_mxfp4_experts_and_fp8_attention_each_get_their_tuner(tmp_path):
    specs = _routed(_quark_moe_profile(layer_quant_config=_fp8_attention()), "quark", tmp_path)
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
    specs = _routed(_quark_moe_profile(layer_quant_config=_fp8_attention()), "mxfp4", tmp_path)
    running = _running(specs)
    assert {"fmoe_ck", "a4w4_blockscale"} <= running
    assert not any(name.startswith("a8w8") for name in running)


def test_the_logged_moe_dispatch_outranks_the_declared_expert_format(tmp_path):
    specs = _routed(_quark_moe_profile(), "quark", tmp_path, log_line=_FP8_PER_TOKEN_MOE_DISPATCH)
    (moe,) = [s for s in specs if s.name == "fmoe_ck"]
    assert moe.skip_reason and "per_Token" in moe.skip_reason
    # The dense lane is not what that line describes, so it keeps the checkpoint's format.
    assert "a4w4_blockscale" in _running(specs)


def test_a_format_precision_is_trusted_over_the_logged_moe_dispatch(tmp_path):
    specs = _routed(_quark_moe_profile(), "mxfp4", tmp_path, log_line=_FP8_PER_TOKEN_MOE_DISPATCH)
    assert "fmoe_ck" in _running(specs)


def test_a_logged_quant_type_no_moe_tuner_handles_leaves_the_declared_format_to_decide(tmp_path):
    specs = _routed(_quark_moe_profile(), "quark", tmp_path, log_line=_fp8_moe_dispatch("per_128x128"))
    assert "fmoe_ck" in _running(specs)


def test_experts_declared_only_by_their_moe_module_class_get_the_moe_tuner(tmp_path):
    profile = _quark_moe_profile(None, layer_type_quant_config={"FusedMoE": _MXFP4})
    assert "fmoe_ck" in _running(_routed(profile, "quark", tmp_path))


def test_a_moe_module_class_entry_overrides_the_whole_checkpoint_format_for_the_experts_only():
    fp8 = {"weight": {"dtype": "fp8_e4m3", "qscheme": "per_tensor"}}
    profile = _quark_moe_profile(fp8, layer_type_quant_config={"FusedMoE": _MXFP4})
    assert (profile.expert_precision, profile.dense_precision) == ("fp4", "fp8")


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
