# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A vLLM run whose MoE is partly served by aiter needs the CK tuner too.

Routing by framework assumes vLLM's Triton path owns the MoE. aiter's CK
fused-MoE can serve some or all of the token range in the same process, and its
table is written by ``fmoe_ck``, which the vLLM branch never selects. When both
appear in one log the answer is not to pick a side: each serves the range it
serves, and dropping either forfeits that range -- the same mistake as letting a
single 1-stage sighting disable CK tuning for the tokens 2-stage was serving.
"""

from __future__ import annotations

from kernelforge.gemm_tune.model_analyzer import ModelProfile
from kernelforge.gemm_tune.router import select_tuners

_CK = "[aiter] [fused_moe] using 2stage ck for (256, {tok}, 4096, 2048, 8, 2)"
_ASM = "[aiter] [fused_moe] using 1stage asm for (256, {tok}, 4096, 2048, 8, 2)"
_MISS = (
    "[aiter] [fused_moe] no tuned FlyDSL config for "
    "('gfx950', 256, {tok}, 4096, 2048, 8, 2, <ActivationType.Swiglu: 2>, "
    "'torch.bfloat16', 'torch.float4_e2m1fn_x2', 'torch.float4_e2m1fn_x2', "
    "'QuantType.per_1x32', True, False), using heuristic FlyDSL fallback"
)
_TRITON = "Using configuration from /x/E=8,N=14336.json for MoE layer"


def _moe_profile() -> ModelProfile:
    return ModelProfile(
        model_path="/fake",
        architecture="MixtralForCausalLM",
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        hidden_size=4096,
        intermediate_size=14336,
        moe_intermediate_size=14336,
    )


def _log(tmp_path, lines) -> str:
    p = tmp_path / "server.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def _names(specs) -> list[str]:
    return [s.name for s in specs if not s.skip_reason]


def _select(tmp_path, lines, **kw):
    return select_tuners(
        _moe_profile(),
        framework="vllm",
        precision="bf16",
        quant_type="none",
        gpu_type="mi355x",
        kernel_signature_log=_log(tmp_path, lines) if lines is not None else None,
        **kw,
    )


class TestVllmMoeRouting:
    def test_ck_in_the_log_adds_the_ck_tuner(self, tmp_path):
        names = _names(
            _select(
                tmp_path,
                [_CK.format(tok=16), _CK.format(tok=64), _MISS.format(tok=16)],
            )
        )
        assert "fmoe_ck" in names
        # And the Triton tuner stays: the log does not say Triton served nothing.
        assert "vllm_moe_triton" in names

    def test_a_mixed_log_keeps_both(self, tmp_path):
        # Part of the token range is CK-served and part is Triton-served, so
        # both tables need tuning.
        names = _names(
            _select(
                tmp_path,
                [
                    _ASM.format(tok=4096),
                    _CK.format(tok=16),
                    _MISS.format(tok=16),
                    _TRITON,
                ],
            )
        )
        assert {"fmoe_ck", "vllm_moe_triton"} <= set(names)

    def test_the_ck_tuner_is_given_only_the_tokens_ck_served(self, tmp_path):
        # Selecting both tuners is not enough on its own: a CK table keyed on
        # the token counts Triton served is one nothing ever reads.
        specs = _select(
            tmp_path,
            [
                _CK.format(tok=16),
                _CK.format(tok=64),
                _MISS.format(tok=16),
                _ASM.format(tok=4096),
                _TRITON,
            ],
        )
        (ck,) = [s for s in specs if s.name == "fmoe_ck"]
        assert ck.token_hint == [16, 64]

    def test_no_token_detail_means_the_runs_full_coverage(self, tmp_path):
        # A log naming the stage but no token count says nothing about which
        # part of the range CK served, so narrowing would be a guess.
        line = "[aiter] [fused_moe] using 2stage ck for (x, y, z)"
        (ck,) = [s for s in _select(tmp_path, [line, _MISS.format(tok=16)]) if s.name == "fmoe_ck"]
        assert ck.token_hint is None

    def test_ck_dispatches_with_no_misses_do_not_add_the_ck_tuner(self, tmp_path):
        names = _names(_select(tmp_path, [_CK.format(tok=16), _CK.format(tok=64)]))
        assert "fmoe_ck" not in names
        assert "vllm_moe_triton" in names

    def test_misses_only_on_asm_tokens_do_not_add_the_ck_tuner(self, tmp_path):
        names = _names(
            _select(
                tmp_path,
                [
                    _CK.format(tok=16),
                    _CK.format(tok=64),
                    _ASM.format(tok=4096),
                    _ASM.format(tok=8192),
                    _MISS.format(tok=4096),
                    _MISS.format(tok=8192),
                ],
            )
        )
        assert "fmoe_ck" not in names
        assert "vllm_moe_triton" in names

    def test_a_triton_only_log_does_not_add_the_ck_tuner(self, tmp_path):
        names = _names(_select(tmp_path, [_TRITON]))
        assert "fmoe_ck" not in names
        assert "vllm_moe_triton" in names

    def test_a_1stage_only_log_does_not_add_the_ck_tuner(self, tmp_path):
        # 1-stage ASM is not what fmoe_ck tunes; adding it would burn a tuner
        # on a path it cannot write a table for.
        names = _names(_select(tmp_path, [_ASM.format(tok=4096)]))
        assert "fmoe_ck" not in names

    def test_no_log_leaves_routing_exactly_as_before(self, tmp_path):
        assert _names(_select(tmp_path, None)) == _names(_select(tmp_path, []))

    def test_a_dense_model_is_untouched(self, tmp_path):
        dense = ModelProfile(
            model_path="/fake",
            hidden_size=4096,
            intermediate_size=14336,
        )
        specs = select_tuners(
            dense,
            framework="vllm",
            precision="bf16",
            quant_type="none",
            gpu_type="mi355x",
            kernel_signature_log=_log(tmp_path, [_CK.format(tok=16)]),
        )
        assert "fmoe_ck" not in [s.name for s in specs]

    def test_the_ck_tuner_is_never_added_twice(self, tmp_path):
        names = [s.name for s in _select(tmp_path, [_CK.format(tok=16), _MISS.format(tok=16)])]
        assert names.count("fmoe_ck") == 1

    def test_sglang_routing_is_unchanged(self, tmp_path):
        # sglang already selects fmoe_ck through its own branch; the vLLM-side
        # addition must not double it or reorder anything.
        specs = select_tuners(
            _moe_profile(),
            framework="sglang",
            precision="bf16",
            quant_type="none",
            gpu_type="mi355x",
            kernel_signature_log=_log(tmp_path, [_CK.format(tok=16), _MISS.format(tok=16)]),
        )
        names = [s.name for s in specs]
        assert names.count("fmoe_ck") == 1
        assert names == sorted(names, key=lambda n: 10 if n == "fmoe_ck" else 20)
