# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The dense GEMM the runtime dispatches is not the one ``precision`` names.

Regression cover for a quantized MoE model whose dense traffic is bf16. Five
production sessions tuned ``a4w4_blockscale`` for two hours apiece while the
serving process looked up ``bf16_tuned_gemm.csv`` twenty thousand times and
found nothing, because the dense branch was an if/elif chain on one scalar and
the bf16 arm sat behind the fp4 arm.
"""

from kernelforge.gemm_tune.model_analyzer import ModelProfile
from kernelforge.gemm_tune.router import select_tuners


def _mxfp4_moe_profile(**kwargs):
    """A quark mxfp4 MoE checkpoint, shaped like MiniMax-M3-MXFP4.

    The ``exclude`` list is the real one, collapsed: quark leaves lm_head and
    every attention projection at bf16, so ~99% of the weight bytes are fp4 and
    ~100% of the dense GEMM calls are not.
    """
    defaults = {
        "model_path": "/fake/MiniMax-M3-MXFP4",
        "is_moe": True,
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "hidden_size": 6144,
        "intermediate_size": 3072,
        "moe_intermediate_size": 3072,
        "num_hidden_layers": 60,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "model_dtype": "bfloat16",
        "quant_method": "quark",
        "raw_config": {
            "quantization_config": {
                "quant_method": "quark",
                "global_quant_config": {"weight": {"dtype": "fp4", "group_size": 32}},
                "exclude": [
                    "language_model.lm_head",
                    "language_model.model.layers.0.self_attn.q_proj",
                    "language_model.model.layers.0.self_attn.k_proj",
                    "language_model.model.layers.0.self_attn.v_proj",
                    "language_model.model.layers.0.self_attn.o_proj",
                    "language_model.model.layers.0.input_layernorm",
                ],
            }
        },
    }
    defaults.update(kwargs)
    return ModelProfile(**defaults)


def _demand(tuner, table, *, misses=21824, keys=2732):
    """A demand.json naming one table the serving run consulted."""
    return {
        "demands": [
            {
                "table": table,
                "tuner": tuner,
                "env_var": "AITER_CONFIG_GEMM_BF16",
                "miss_count": misses,
                "distinct_keys": keys,
                "keys": [{"M": 1, "N": 6144, "K": 6144, "requests": misses}],
            }
        ],
    }


class TestExclusionListIsRead:
    """``precision`` describes the majority; ``exclude`` describes the rest."""

    def test_unquantized_linears_are_listed_without_the_norms(self):
        profile = _mxfp4_moe_profile()
        modules = profile.unquantized_linear_modules
        assert "language_model.lm_head" in modules
        assert "language_model.model.layers.0.self_attn.q_proj" in modules
        # A layernorm is in the same list and is not a GEMM.
        assert not any("layernorm" in m for m in modules)

    def test_a_checkpoint_with_no_exclusions_keeps_nothing_dense(self):
        profile = _mxfp4_moe_profile(raw_config={})
        assert not profile.keeps_dense_layers_at_model_dtype

    def test_awq_style_key_is_understood(self):
        profile = _mxfp4_moe_profile(
            raw_config={
                "quantization_config": {"modules_to_not_convert": ["lm_head"]},
            }
        )
        assert profile.unquantized_linear_modules == ["lm_head"]


class TestQuantizedModelsStillTuneBf16Dense:
    """The bug: one ``elif`` made the bf16 tuner unreachable once quantized."""

    def test_mxfp4_moe_selects_both_dense_tuners(self):
        profile = _mxfp4_moe_profile()
        specs = select_tuners(
            profile,
            framework="sglang",
            precision="mxfp4",
            quant_type="auto",
            gpu_type="mi355x",
        )
        names = [s.name for s in specs if s.should_run]
        # a4w4 stays: this is only ever additive.
        assert "a4w4_blockscale" in names
        assert "sglang_dense_bf16" in names

    def test_fp8_model_with_excluded_layers_also_tunes_bf16_dense(self):
        profile = _mxfp4_moe_profile()
        specs = select_tuners(
            profile,
            framework="sglang",
            precision="fp8",
            quant_type="blockscale",
            gpu_type="mi300x",
        )
        names = [s.name for s in specs if s.should_run]
        assert "a8w8_blockscale" in names
        assert "sglang_dense_bf16" in names

    def test_fully_quantized_checkpoint_does_not_get_a_bf16_pass(self):
        """No exclusion list means no bf16 dense to tune -- do not spend the time."""
        profile = _mxfp4_moe_profile(raw_config={})
        specs = select_tuners(
            profile,
            framework="sglang",
            precision="mxfp4",
            quant_type="auto",
            gpu_type="mi355x",
        )
        names = [s.name for s in specs if s.should_run]
        assert "a4w4_blockscale" in names
        assert "sglang_dense_bf16" not in names

    def test_dense_fp8_with_only_lm_head_excluded_does_not_get_a_bf16_pass(self):
        profile = _mxfp4_moe_profile(
            is_moe=False,
            num_experts=0,
            raw_config={
                "quantization_config": {
                    "modules_to_not_convert": ["lm_head"],
                }
            },
        )
        specs = select_tuners(
            profile,
            framework="sglang",
            precision="fp8",
            quant_type="blockscale",
            gpu_type="mi300x",
        )
        names = [s.name for s in specs if s.should_run]
        assert "a8w8_blockscale" in names
        assert "sglang_dense_bf16" not in names

    def test_dense_fp8_with_attention_excluded_keeps_the_bf16_pass(self):
        profile = _mxfp4_moe_profile(
            is_moe=False,
            num_experts=0,
            raw_config={
                "quantization_config": {
                    "modules_to_not_convert": ["lm_head", "model.layers.0.self_attn.q_proj"],
                }
            },
        )
        specs = select_tuners(
            profile,
            framework="sglang",
            precision="fp8",
            quant_type="blockscale",
            gpu_type="mi300x",
        )
        assert "sglang_dense_bf16" in [s.name for s in specs if s.should_run]

    def test_fp32_weights_are_not_handed_to_the_bf16_tuner(self):
        profile = _mxfp4_moe_profile(model_dtype="float32")
        specs = select_tuners(
            profile,
            framework="sglang",
            precision="mxfp4",
            quant_type="auto",
            gpu_type="mi355x",
        )
        assert "sglang_dense_bf16" not in [s.name for s in specs if s.should_run]


class TestDemandOverridesTheGuess:
    """A consulted table is measurement; a precision label is inference."""

    def test_demand_adds_the_tuner_the_router_missed(self):
        profile = _mxfp4_moe_profile(raw_config={})  # exclusion list unavailable
        specs = select_tuners(
            profile,
            framework="sglang",
            precision="mxfp4",
            quant_type="auto",
            gpu_type="mi355x",
            demand_report=_demand("sglang_dense_bf16", "bf16_tuned_gemm.csv"),
        )
        names = [s.name for s in specs if s.should_run]
        assert "sglang_dense_bf16" in names
        assert "a4w4_blockscale" in names

    def test_demand_never_removes_what_the_framework_branch_chose(self):
        profile = _mxfp4_moe_profile()
        without = {
            s.name
            for s in select_tuners(
                profile,
                framework="sglang",
                precision="mxfp4",
                quant_type="auto",
                gpu_type="mi355x",
            )
        }
        with_demand = {
            s.name
            for s in select_tuners(
                profile,
                framework="sglang",
                precision="mxfp4",
                quant_type="auto",
                gpu_type="mi355x",
                demand_report=_demand("sglang_dense_bf16", "bf16_tuned_gemm.csv"),
            )
        }
        assert without <= with_demand

    def test_demand_does_not_overturn_a_skip_reason(self):
        """A skip is a capability statement (wrong arch), not a selection miss."""
        profile = _mxfp4_moe_profile()
        specs = select_tuners(
            profile,
            framework="sglang",
            precision="mxfp4",
            quant_type="auto",
            gpu_type="mi300x",  # fp4 unsupported on gfx942
            demand_report=_demand("a4w4_blockscale", "a4w4_blockscale_tuned_gemm.csv"),
        )
        a4w4 = [s for s in specs if s.name == "a4w4_blockscale"]
        assert len(a4w4) == 1
        assert not a4w4[0].should_run

    def test_an_unowned_demand_is_left_to_the_coverage_report(self):
        profile = _mxfp4_moe_profile(raw_config={})
        report = {"demands": [{"table": "something_tuned_gemm.csv", "tuner": None}]}
        specs = select_tuners(
            profile,
            framework="sglang",
            precision="mxfp4",
            quant_type="auto",
            gpu_type="mi355x",
            demand_report=report,
        )
        assert all(s.name for s in specs)

    def test_no_demand_file_changes_nothing(self):
        profile = _mxfp4_moe_profile()
        args = dict(framework="sglang", precision="mxfp4", quant_type="auto", gpu_type="mi355x")
        assert [s.name for s in select_tuners(profile, **args)] == [
            s.name for s in select_tuners(profile, demand_report=None, **args)
        ]

    def test_fmoe_ck_from_demand_keeps_its_moe_priority(self):
        """Ordering is a budget decision; a demand-added tuner must not jump it."""
        profile = _mxfp4_moe_profile(is_moe=False, raw_config={})
        specs = select_tuners(
            profile,
            framework="sglang",
            precision="mxfp4",
            quant_type="auto",
            gpu_type="mi355x",
            demand_report=_demand("fmoe_ck", "tuned_fmoe.csv"),
        )
        names = [s.name for s in specs]
        assert names.index("fmoe_ck") < names.index("a4w4_blockscale")
