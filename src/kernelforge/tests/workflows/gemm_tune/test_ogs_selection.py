# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Route only the GPT-OSS BF16 × MXFP4 serving path to matmul_ogs tuning."""

from kernelforge.workflows.gemm_tune.model_analyzer import ModelProfile
from kernelforge.workflows.gemm_tune.router import RoutingEvidence, RoutingTarget, select_tuners
from kernelforge.workflows.gemm_tune.tuners.ogs_tuning import cases, flag_search_space, projection_shapes


def _profile(**updates) -> ModelProfile:
    values = dict(
        model_path="/models/gpt-oss-20b",
        architecture="GptOssForCausalLM",
        is_moe=True,
        num_experts=32,
        num_experts_per_tok=4,
        hidden_size=2880,
        intermediate_size=2880,
        quant_method="mxfp4",
        model_dtype="bfloat16",
    )
    values.update(updates)
    return ModelProfile(**values)


def _names(profile: ModelProfile, gpu_type: str) -> list:
    return select_tuners(
        profile,
        RoutingTarget(framework="vllm", precision="mxfp4", quant_type="auto", gpu_type=gpu_type),
        RoutingEvidence(has_shapes_json=True),
    )


def test_r9700_gpt_oss_mxfp4_selects_the_kernel_it_serves():
    specs = _names(_profile(), "r9700")
    assert specs[0].name == "vllm_mxfp4_matmul_ogs"
    assert specs[0].should_run
    assert not any(spec.name == "vllm_moe_triton" for spec in specs)
    assert next(spec for spec in specs if spec.name == "vllm_dense_tunableop").fallback


def test_other_gpus_and_moe_models_keep_the_quantized_skip():
    for profile, gpu_type in [
        (_profile(), "mi355x"),
        (_profile(), "r9600d"),
        (_profile(architecture="OtherMoE"), "r9700"),
    ]:
        specs = _names(profile, gpu_type)
        (moe,) = [spec for spec in specs if spec.name == "vllm_moe_triton"]
        assert not moe.should_run
        assert "mxfp4" in moe.skip_reason


def test_gpt_oss_projection_shapes_match_rocm_mxfp4_padding():
    assert projection_shapes(_profile()) == ((6144, 3072), (3072, 3072))
    planned = cases(_profile(), [1, 15, 512], conc=15, thorough=False)
    assert {(case.n, case.k) for case in planned} == {(6144, 3072), (3072, 3072)}
    assert any(case.min_m <= 2 <= case.max_m for case in planned)
    assert any(case.min_m <= 64 <= case.max_m for case in planned)


def test_flag_search_space_is_bounded_and_covers_the_rdna_levers():
    normal = flag_search_space(thorough=False)
    thorough = flag_search_space(thorough=True)
    assert len(normal) == 4
    assert len(thorough) > len(normal)
    assert all(
        set(flags) == {"block_m", "block_n", "block_k", "num_warps", "num_stages", "split_k", "xcd_swizzle"}
        for flags in thorough
    )
    assert {flags["xcd_swizzle"] for flags in thorough} == {0, 8}
