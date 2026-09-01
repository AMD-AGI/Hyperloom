# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The record type comes from the shape's dtype, not from the checkpoint precision.

``ctx.precision`` describes how the WEIGHTS are stored; it says nothing about
what the dense GEMM runs in. On the fleet's ``Qwen3.8-2.4T-A95B-Quark-MXFP4``
every one of the 21056 dense aiter lookups in a real serving log is
``dtype='torch.bfloat16'`` while ``ctx.precision`` is ``mxfp4``. Keying the
TunableOp record type off the precision alone found nothing and threw the whole
demand file away, then reported it as ``input_missing`` -- which reads as "the
upstream never sent data" when the data was there and understood.
"""

from __future__ import annotations

import json

from kernelforge.gemm_tune.model_analyzer import ModelProfile
from kernelforge.gemm_tune.tuners.base import TuneContext
from kernelforge.gemm_tune.tuners.vllm_dense_tunableop import VllmDenseTunableopTuner

BF16 = "GemmTunableOp_BFloat16_TN"
FP16 = "GemmTunableOp_Half_TN"


def _demand_file(tmp_path, keys):
    from kernelforge.gemm_tune.evidence import SCHEMA_VERSION

    path = tmp_path / "demand.json"
    path.write_text(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "demands": [
                    {
                        "table": "tunableop",
                        "tuner": "vllm_dense_tunableop",
                        "env_var": "PYTORCH_TUNABLEOP_FILENAME",
                        "key_schema": ["M", "N", "K"],
                        "logged_fields": ["M", "N", "K", "dtype"],
                        "miss_count": len(keys),
                        "keys": keys,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _key(m, n, k, dtype=None, **extra):
    key = {"M": m, "N": n, "K": k, "requests": 1}
    if dtype is not None:
        key["dtype"] = dtype
    key.update(extra)
    return key


def _tuner(tmp_path, demand, precision="bf16"):
    return VllmDenseTunableopTuner(
        TuneContext(
            profile=ModelProfile(model_path="/fake", hidden_size=4096),
            framework="vllm",
            precision=precision,
            quant_type="none",
            gpu_type="mi355x",
            tp=1,
            conc=8,
            tokens=[8],
            mp=1,
            output_dir=tmp_path,
            iters=20,
            warmup=5,
            min_improvement_pct=1.0,
            timeout_s=3600,
            demand_json=demand,
        )
    )


def _records(tmp_path, keys, precision="bf16"):
    path = _tuner(tmp_path, _demand_file(tmp_path, keys), precision)._resolve_input()
    if path is None:
        return None
    return path.read_text(encoding="utf-8").strip().splitlines()


class TestTheShapeDtypeDecidesTheRecordType:
    def test_an_mxfp4_checkpoint_still_tunes_its_bf16_shapes(self, tmp_path):
        """The measured production case: mxfp4 weights, bf16 dense GEMMs."""
        lines = _records(
            tmp_path,
            [_key(15842, 512, 4096, "torch.bfloat16")],
            precision="mxfp4",
        )
        assert lines == [f"{BF16},tn_512_15842_4096_ld_4096_4096_512"]

    def test_an_fp8_checkpoint_still_tunes_its_bf16_shapes(self, tmp_path):
        lines = _records(tmp_path, [_key(16, 1536, 7168, "torch.bfloat16")], precision="fp8")
        assert lines == [f"{BF16},tn_1536_16_7168_ld_7168_7168_1536"]

    def test_a_float16_shape_gets_the_half_record_type(self, tmp_path):
        lines = _records(tmp_path, [_key(16, 1536, 7168, "torch.float16")], precision="mxfp4")
        assert lines == [f"{FP16},tn_1536_16_7168_ld_7168_7168_1536"]

    def test_the_shape_dtype_wins_over_the_checkpoint_precision(self, tmp_path):
        # Not merely a fallback: where the two disagree, only the logged dtype
        # describes the GEMM that actually ran.
        lines = _records(tmp_path, [_key(16, 1536, 7168, "torch.bfloat16")], precision="fp16")
        assert lines == [f"{BF16},tn_1536_16_7168_ld_7168_7168_1536"]

    def test_otype_is_read_when_dtype_is_absent(self, tmp_path):
        lines = _records(
            tmp_path,
            [_key(16, 1536, 7168, otype="torch.bfloat16")],
            precision="mxfp4",
        )
        assert lines == [f"{BF16},tn_1536_16_7168_ld_7168_7168_1536"]

    def test_a_dtypeless_shape_falls_back_to_the_precision(self, tmp_path):
        # The pre-existing behaviour, kept: older demand files carry no dtype.
        lines = _records(tmp_path, [_key(16, 1536, 7168)], precision="bf16")
        assert lines == [f"{BF16},tn_1536_16_7168_ld_7168_7168_1536"]


class TestUnsupportedDtypesAreSkippedNotGuessed:
    def test_an_unrecognised_dtype_is_dropped_rather_than_coerced(self, tmp_path):
        # TunableOp keys on the record type. Writing an fp8 shape under a bf16
        # record type would tune something the runtime never looks up.
        lines = _records(
            tmp_path,
            [
                _key(16, 1536, 7168, "torch.bfloat16"),
                _key(32, 2048, 4096, "torch.float8_e4m3fn"),
            ],
            precision="mxfp4",
        )
        assert lines == [f"{BF16},tn_1536_16_7168_ld_7168_7168_1536"]

    def test_all_shapes_unsupported_yields_no_input(self, tmp_path):
        assert _records(tmp_path, [_key(16, 1536, 7168, "torch.float8_e4m3fn")], precision="mxfp4") is None


class TestTheStatusSaysWhatActuallyHappened:
    def test_all_unsupported_reports_skipped_not_input_missing(self, tmp_path):
        """The demand file arrived and parsed; there was simply nothing here to tune."""
        demand = _demand_file(tmp_path, [_key(16, 1536, 7168, "torch.float8_e4m3fn")])
        result = _tuner(tmp_path, demand, precision="mxfp4").run()

        assert result.status == "skipped"
        assert result.error_class == "unsupported_precision"
        assert "float8_e4m3fn" in (result.error or "")

    def test_a_genuinely_missing_demand_file_still_reports_input_missing(self, tmp_path):
        # The skip must not swallow the real failure it sits next to.
        result = _tuner(tmp_path, None, precision="mxfp4").run()

        assert result.status == "failed"
        assert result.error_class == "input_missing"
