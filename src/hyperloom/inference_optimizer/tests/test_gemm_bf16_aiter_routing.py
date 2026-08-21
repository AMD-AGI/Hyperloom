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
    "('gfx950', 256, 256, 8192, 1536, 128, 4, 'ActivationType.Swiglu', 'torch.bfloat16', "
    "'torch.float8_e4m3fn', 'torch.float4_e2m1fn_x2', 'QuantType.per_1x32', True, False)"
)
AITER_FUSED_MOE_BF16_FP4 = (
    "(Worker_TP0 pid=1) [aiter] [fused_moe] using 2stage default for "
    "('gfx950', 256, 256, 8192, 3072, 128, 4, 'ActivationType.Swiglu', 'torch.bfloat16', "
    "'torch.bfloat16', 'torch.float4_e2m1fn_x2', 'QuantType.per_1x32', True, False)"
)

# Verbatim lines from a production server.log, one per wording aiter emits.
# Kept literal because the previous hand-written fixtures dropped the leading
# gfx field, which let a regex that could never match a real log pass its tests.
REAL_2STAGE_DEFAULT = (
    "[aiter] [fused_moe] using 2stage default for ('gfx950', 256, 256, 4096, 512, 256, 6, "
    "'ActivationType.Silu', 'torch.bfloat16', 'torch.float8_e4m3fn', "
    "'torch.float4_e2m1fn_x2', 'QuantType.per_1x32', True, False)"
)
REAL_NO_TUNED_FLYDSL = (
    "[aiter] [fused_moe] no tuned FlyDSL config for ('gfx950', 256, 256, 4096, 512, 256, 6, "
    "'ActivationType.Silu', 'torch.bfloat16', 'torch.float8_e4m3fn', "
    "'torch.float4_e2m1fn_x2', 'QuantType.per_1x32', True, False), using heuristic "
    "FlyDSL fallback (kn1='flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w2_gui', "
    "kn2='flydsl_moe2_afp8_wfp4_bf16_t32x128x256_atomic_bnt2')"
)
REAL_2STAGE_WITH_KERNEL_NAMES = (
    "(Worker_TP7 pid=26394) [aiter] [fused_moe] using 2stage "
    "(kernelName1='flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w4_bnt0_gui_fp8', "
    "kernelName2='opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn3584') for "
    "('gfx950', 256, 8192, 7168, 512, 384, 6, 'ActivationType.Silu', 'torch.bfloat16', "
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


def _moe_tuple(
    q_a: str,
    q_w: str,
    q_type: str = "QuantType.per_1x32",
    *,
    inter_dim: int = 3072,
    expert: int = 128,
    topk: int = 4,
) -> str:
    return (
        "(Worker_TP0 pid=1) [aiter] [fused_moe] using 2stage default for "
        f"('gfx950', 256, 8192, 3072, {inter_dim}, {expert}, {topk}, "
        f"'ActivationType.Swiglu', 'torch.bfloat16', "
        f"'{q_a}', '{q_w}', '{q_type}', True, False)"
    )


class TestCkMoeTunerSupport:
    def test_bf16_activation_with_fp4_weights_is_unsupported(self, tmp_path):
        """Measured: candidate generation raises 'Unsupported data type combination'."""
        log = _log(tmp_path, _moe_tuple("torch.bfloat16", "torch.float4_e2m1fn_x2"))
        assert not krh._aiter_ck_moe_tuner_supports(log)

    def test_fp8_activation_with_fp4_weights_is_supported(self, tmp_path):
        log = _log(tmp_path, _moe_tuple("torch.float8_e4m3fn", "torch.float4_e2m1fn_x2"))
        assert krh._aiter_ck_moe_tuner_supports(log)

    def test_fp4_activation_and_weights_is_supported(self, tmp_path):
        log = _log(tmp_path, _moe_tuple("torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2"))
        assert krh._aiter_ck_moe_tuner_supports(log)

    def test_unquantised_bf16_moe_is_supported(self, tmp_path):
        log = _log(tmp_path, _moe_tuple("torch.bfloat16", "torch.bfloat16", "QuantType.No"))
        assert krh._aiter_ck_moe_tuner_supports(log)

    def test_a_mixed_log_stays_tunable_because_rows_are_filtered(self, tmp_path):
        """One checkpoint dispatches several pairs; the tunable ones still count.

        Measured in production: the same model logs both a BF16-activation and an
        FP8-activation problem. Blocking the whole model on the unsupported one
        would forfeit the tunable half, so the untunable rows are dropped when the
        tuning input is written instead.
        """
        log = _log(
            tmp_path,
            _moe_tuple("torch.float8_e4m3fn", "torch.float4_e2m1fn_x2")
            + "\n"
            + _moe_tuple("torch.bfloat16", "torch.float4_e2m1fn_x2", expert=257, topk=5),
        )
        assert krh._aiter_ck_moe_tuner_supports(log)

    def test_an_entirely_unsupported_log_is_rejected(self, tmp_path):
        log = _log(
            tmp_path,
            _moe_tuple("torch.bfloat16", "torch.float4_e2m1fn_x2")
            + "\n"
            + _moe_tuple("torch.bfloat16", "torch.float4_e2m1fn_x2", expert=257, topk=5),
        )
        assert not krh._aiter_ck_moe_tuner_supports(log)

    def test_moe_evidence_without_a_parseable_tuple_defers_to_forge(self, tmp_path):
        log = _log(tmp_path, "INFO Using 'AITER_MXFP4_BF16' Mxfp4 MoE backend.\n")
        assert krh._aiter_ck_moe_tuner_supports(log)


class TestDispatchKeyExtraction:
    """The regex must match every wording aiter actually emits.

    A hand-written fixture previously omitted the leading gfx field, so a regex
    that could not match a single real log line passed its tests while silently
    disabling the dtype gate in production.
    """

    def test_matches_the_plain_default_wording(self, tmp_path):
        keys = krh._aiter_fused_moe_dispatch_keys(_log(tmp_path, REAL_2STAGE_DEFAULT))
        assert len(keys) == 1
        assert keys[0]["inter_dim"] == "512"
        assert keys[0]["q_dtype_a"] == "torch.float8_e4m3fn"
        assert keys[0]["q_dtype_w"] == "torch.float4_e2m1fn_x2"
        assert keys[0]["expert"] == "256"
        assert keys[0]["topk"] == "6"

    def test_matches_the_flydsl_fallback_wording(self, tmp_path):
        keys = krh._aiter_fused_moe_dispatch_keys(_log(tmp_path, REAL_NO_TUNED_FLYDSL))
        assert len(keys) == 1
        assert keys[0]["inter_dim"] == "512"

    def test_matches_the_wording_that_interposes_kernel_names(self, tmp_path):
        """This form puts its own parenthesised group before the tuple."""
        keys = krh._aiter_fused_moe_dispatch_keys(
            _log(tmp_path, REAL_2STAGE_WITH_KERNEL_NAMES)
        )
        assert len(keys) == 1
        assert keys[0]["model_dim"] == "7168"
        assert keys[0]["expert"] == "384"

    def test_dedupes_on_everything_but_the_token_count(self, tmp_path):
        a = REAL_2STAGE_DEFAULT
        b = REAL_2STAGE_DEFAULT.replace("256, 256, 4096", "256, 512, 4096")
        keys = krh._aiter_fused_moe_dispatch_keys(_log(tmp_path, a + "\n" + b))
        assert len(keys) == 1

    def test_keeps_distinct_problems_from_one_model(self, tmp_path):
        """The EP path inflates expert/topk by one; that is a separate problem."""
        a = _moe_tuple("torch.float8_e4m3fn", "torch.float4_e2m1fn_x2")
        b = _moe_tuple("torch.float8_e4m3fn", "torch.float4_e2m1fn_x2", expert=129, topk=5)
        keys = krh._aiter_fused_moe_dispatch_keys(_log(tmp_path, a + "\n" + b))
        assert len(keys) == 2

    def test_missing_log(self, tmp_path):
        assert krh._aiter_fused_moe_dispatch_keys("") == []
        assert krh._aiter_fused_moe_dispatch_keys(str(tmp_path / "absent.log")) == []


class TestDtypePairSupport:
    """Mirrors the four kernel families in aiter's CK MoE codegen."""

    def test_supported_pairs(self):
        for act, weight in (
            ("torch.bfloat16", "torch.bfloat16"),
            ("torch.float16", "torch.float16"),
            ("torch.float8_e4m3fn", "torch.float8_e4m3fn"),
            ("torch.float8_e4m3fn", "torch.float4_e2m1fn_x2"),
            ("torch.float8_e4m3fnuz", "torch.float4_e2m1fn_x2"),
            ("torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2"),
        ):
            assert krh._aiter_moe_dtype_pair_supported(act, weight), (act, weight)

    def test_bf16_activation_with_fp4_weight_is_the_known_rejection(self):
        assert not krh._aiter_moe_dtype_pair_supported(
            "torch.bfloat16", "torch.float4_e2m1fn_x2"
        )

    def test_int8_activation_does_not_qualify_for_the_a8w4_family(self):
        """The a8w4 branch requires an FP8 activation specifically."""
        assert not krh._aiter_moe_dtype_pair_supported(
            "torch.int8", "torch.float4_e2m1fn_x2"
        )


class TestWriteFmoeUntunedCsvFromLog:
    def test_writes_one_row_per_problem_and_token(self, tmp_path):
        path, report = krh._write_fmoe_untuned_csv_from_log(
            _log(tmp_path, REAL_2STAGE_DEFAULT), [4, 512], tmp_path / "ws"
        )
        rows = [
            line for line in open(path, encoding="utf-8").read().splitlines() if line
        ]
        assert rows[0].split(",") == [
            "token", "model_dim", "inter_dim", "expert", "topk", "act_type", "dtype",
            "q_dtype_a", "q_dtype_w", "q_type", "use_g1u1", "doweight_stage1",
        ]
        assert len(rows) == 3  # header + 2 tokens
        assert rows[1] == (
            "4,4096,512,256,6,ActivationType.Silu,torch.bfloat16,"
            "torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0"
        )
        assert report["observed"] == 1
        assert report["tunable"] == 1
        assert report["dropped_unsupported"] == []

    def test_drops_pairs_the_tuner_would_reject(self, tmp_path):
        """One unsupported row aborts the whole aiter tuner run, so filter first."""
        log = _log(
            tmp_path,
            _moe_tuple("torch.float8_e4m3fn", "torch.float4_e2m1fn_x2")
            + "\n"
            + _moe_tuple("torch.bfloat16", "torch.float4_e2m1fn_x2", expert=129),
        )
        path, report = krh._write_fmoe_untuned_csv_from_log(log, [8], tmp_path / "ws")

        body = open(path, encoding="utf-8").read()
        assert "torch.bfloat16,torch.float4_e2m1fn_x2" not in body
        assert report["observed"] == 2
        assert report["tunable"] == 1
        assert report["dropped_unsupported"] == ["torch.bfloat16/torch.float4_e2m1fn_x2"]

    def test_no_tunable_problem_yields_no_csv(self, tmp_path):
        log = _log(tmp_path, _moe_tuple("torch.bfloat16", "torch.float4_e2m1fn_x2"))
        path, report = krh._write_fmoe_untuned_csv_from_log(log, [8], tmp_path / "ws")

        assert path == ""
        assert report["observed"] == 1
        assert report["tunable"] == 0

    def test_unwritable_workspace_costs_only_the_moe_input(self, tmp_path, monkeypatch):
        """A full disk must not take the dense tuners down with the MoE one."""
        log = _log(tmp_path, _moe_tuple("torch.float8_e4m3fn", "torch.float4_e2m1fn_x2"))

        def _boom(*_args, **_kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(krh.Path, "write_text", _boom)

        path, report = krh._write_fmoe_untuned_csv_from_log(log, [8], tmp_path / "ws")

        assert path == ""
        assert report["tunable"] == 1
        assert "No space left on device" in report["write_error"]

    def test_no_moe_evidence_yields_no_csv(self, tmp_path):
        path, report = krh._write_fmoe_untuned_csv_from_log(
            _log(tmp_path, "INFO server started\n"), [8], tmp_path / "ws"
        )
        assert path == ""
        assert report["observed"] == 0


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

    def test_moe_blocked_when_the_tuner_rejects_the_dtype_pair(self, tmp_path):
        flags = krh._resolve_vllm_aiter_routing(
            model_path=_model(tmp_path, moe=True),
            server_log=_log(tmp_path, AITER_FUSED_MOE_BF16_FP4),
            tp=1,
        )
        assert flags["aiter_fused_moe"] is False

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
