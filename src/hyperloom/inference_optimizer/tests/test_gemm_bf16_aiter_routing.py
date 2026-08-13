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


AITER_FUSED_MOE = (
    "(Worker_TP0 pid=1) [aiter] [fused_moe] using 2stage default for "
    "(256, 8192, 3072, 1536, 128, 4, 'ActivationType.Swiglu', 'torch.bfloat16', "
    "'torch.float8_e4m3fn', 'torch.float4_e2m1fn_x2', 'QuantType.per_1x32', True, False)"
)


class TestAiterServingEvidence:
    def test_detects_bf16_dense(self, tmp_path):
        assert krh._aiter_serving_evidence(_log(tmp_path, AITER_BF16_MISS)) == {"bf16_dense"}

    def test_detects_fused_moe(self, tmp_path):
        assert krh._aiter_serving_evidence(_log(tmp_path, AITER_FUSED_MOE)) == {"fused_moe"}

    def test_detects_mxfp4_moe_backend_line(self, tmp_path):
        line = "INFO [mxfp4.py:513] Using 'AITER_MXFP4_BF16' Mxfp4 MoE backend."
        assert krh._aiter_serving_evidence(_log(tmp_path, line)) == {"fused_moe"}

    def test_no_evidence(self, tmp_path):
        assert krh._aiter_serving_evidence(_log(tmp_path, "INFO server started\n")) == set()

    def test_missing_log(self, tmp_path):
        assert krh._aiter_serving_evidence("") == set()
        assert krh._aiter_serving_evidence(str(tmp_path / "absent.log")) == set()


class TestResolveVllmAiterRouting:
    def test_dense_bf16_model(self, tmp_path):
        flags = krh._resolve_vllm_aiter_routing(
            model_path=_model(tmp_path), server_log=_log(tmp_path, AITER_BF16_MISS), tp=1
        )
        assert flags == {"aiter_bf16_dense": True, "aiter_fused_moe": False}

    def test_moe_model_on_aiter_fused_moe(self, tmp_path):
        flags = krh._resolve_vllm_aiter_routing(
            model_path=_model(tmp_path, moe=True),
            server_log=_log(tmp_path, AITER_FUSED_MOE),
            tp=1,
        )
        assert flags["aiter_fused_moe"] is True
        # A MoE checkpoint's dense side rides along with its MoE routing.
        assert flags["aiter_bf16_dense"] is False

    def test_moe_blocked_when_ck_cannot_serve_the_shard(self, tmp_path):
        """moe_intermediate_size=1536 shards to 192 at tp=8, which CK rejects."""
        flags = krh._resolve_vllm_aiter_routing(
            model_path=_model(tmp_path, moe=True),
            server_log=_log(tmp_path, AITER_FUSED_MOE),
            tp=8,
        )
        assert flags["aiter_fused_moe"] is False

    def test_no_evidence_routes_nothing(self, tmp_path):
        flags = krh._resolve_vllm_aiter_routing(
            model_path=_model(tmp_path), server_log=_log(tmp_path, "INFO up\n"), tp=1
        )
        assert flags == {"aiter_bf16_dense": False, "aiter_fused_moe": False}


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

    def test_fp8_per_token_dense_still_uses_the_vllm_branch(self):
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

    def test_aiter_fused_moe_routes_regardless_of_precision(self):
        for precision in ("mxfp4", "fp8", "bf16"):
            assert (
                krh._forge_framework_for_vllm(
                    framework="vllm",
                    precision=precision,
                    quant_type="auto",
                    tunableop_input="",
                    aiter_fused_moe=True,
                )
                == "vllm-aiter"
            ), precision

    def test_triton_moe_runtime_keeps_the_vllm_branch(self):
        assert (
            krh._forge_framework_for_vllm(
                framework="vllm",
                precision="mxfp4",
                quant_type="auto",
                tunableop_input="",
                aiter_fused_moe=False,
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
