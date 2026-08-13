# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for routing aiter-served BF16 vLLM runs to Forge's AITER dense tuner."""

from __future__ import annotations

import json

from hyperloom.orchestrator.kernel import request_handlers as krh

AITER_BF16_MISS = (
    "(EngineCore pid=1) [aiter] shape is M:1024, N:151936, K:5120 dtype='torch.bfloat16' "
    "otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, not found tuned "
    "config in /tmp/aiter_configs/bf16_tuned_gemm.csv, will use default config! "
    "using torch solution:0"
)


def _model(tmp_path, *, moe: bool = False) -> str:
    root = tmp_path / ("moe_model" if moe else "dense_model")
    root.mkdir(exist_ok=True)
    config = {
        "architectures": ["Qwen3MoeForCausalLM" if moe else "Qwen3ForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "torch_dtype": "bfloat16",
    }
    if moe:
        config.update({"num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 1536})
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(root)


def _log(tmp_path, text: str) -> str:
    path = tmp_path / "server.log"
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestDenseBf16ServesThroughAiter:
    def test_detects_aiter_bf16_lookup_on_a_dense_model(self, tmp_path):
        assert krh._vllm_dense_bf16_serves_through_aiter(
            _model(tmp_path), _log(tmp_path, AITER_BF16_MISS)
        )

    def test_moe_models_are_left_to_their_own_routing(self, tmp_path):
        assert not krh._vllm_dense_bf16_serves_through_aiter(
            _model(tmp_path, moe=True), _log(tmp_path, AITER_BF16_MISS)
        )

    def test_no_aiter_evidence_in_log(self, tmp_path):
        assert not krh._vllm_dense_bf16_serves_through_aiter(
            _model(tmp_path), _log(tmp_path, "INFO server started\n")
        )

    def test_missing_log_is_not_evidence(self, tmp_path):
        assert not krh._vllm_dense_bf16_serves_through_aiter(_model(tmp_path), "")
        assert not krh._vllm_dense_bf16_serves_through_aiter(
            _model(tmp_path), str(tmp_path / "absent.log")
        )


class TestForgeFrameworkRouting:
    def test_bf16_with_aiter_evidence_routes_to_aiter_family(self):
        assert (
            krh._forge_framework_for_vllm(
                framework="vllm",
                precision="bf16",
                quant_type="none",
                tunableop_input="",
                aiter_bf16_dense=True,
            )
            == "vllm-aiter"
        )

    def test_bf16_without_evidence_keeps_tunableop_path(self):
        assert (
            krh._forge_framework_for_vllm(
                framework="vllm",
                precision="bf16",
                quant_type="none",
                tunableop_input="",
                aiter_bf16_dense=False,
            )
            == "vllm"
        )

    def test_block_fp8_routing_is_unchanged(self):
        assert (
            krh._forge_framework_for_vllm(
                framework="vllm",
                precision="fp8",
                quant_type="blockscale",
                tunableop_input="",
            )
            == "vllm-aiter"
        )

    def test_explicit_tunableop_input_wins(self):
        assert (
            krh._forge_framework_for_vllm(
                framework="vllm",
                precision="bf16",
                quant_type="none",
                tunableop_input="/tmp/untuned.csv",
                aiter_bf16_dense=True,
            )
            == "vllm"
        )

    def test_sglang_is_untouched(self):
        assert (
            krh._forge_framework_for_vllm(
                framework="sglang",
                precision="bf16",
                quant_type="none",
                tunableop_input="",
                aiter_bf16_dense=True,
            )
            == "sglang"
        )

    def test_fp8_per_token_still_uses_the_vllm_branch(self):
        assert (
            krh._forge_framework_for_vllm(
                framework="vllm",
                precision="fp8",
                quant_type="per_token",
                tunableop_input="",
                aiter_bf16_dense=False,
            )
            == "vllm"
        )


class TestTunableOpCaptureIsSkippedWhenRouted:
    def test_routed_framework_skips_the_recording_pass(self, tmp_path):
        """A vllm-aiter run must not pay for a TunableOp server boot."""
        assert not krh._vllm_dense_shape_capture_required(
            framework="vllm-aiter",
            model_path=_model(tmp_path),
            shapes_json="",
            tunableop_input="",
            dry_run=False,
        )
        assert krh._vllm_dense_shape_capture_required(
            framework="vllm",
            model_path=_model(tmp_path),
            shapes_json="",
            tunableop_input="",
            dry_run=False,
        )
