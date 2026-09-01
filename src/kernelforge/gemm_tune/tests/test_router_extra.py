# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cover 1-stage log detection, quant-type resolution, and vllm dense paths."""

from __future__ import annotations

from kernelforge.gemm_tune.router import (
    _detect_1stage_from_log,
    _normalize_quant_type,
    _resolve_quant_type,
    select_tuners,
)
from kernelforge.gemm_tune.model_analyzer import ModelProfile


def _profile(is_moe=False, num_experts=0, **kw):
    defaults = {
        "model_path": "/m",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "moe_intermediate_size": 0,
        "num_experts_per_tok": 0,
    }
    defaults.update(kw)
    return ModelProfile(is_moe=is_moe, num_experts=num_experts, **defaults)


# ── _detect_1stage_from_log ──────────────────────────────────────────────────
def test_detect_1stage_none_path():
    assert _detect_1stage_from_log(None) is False


def test_detect_1stage_missing_file(tmp_path):
    assert _detect_1stage_from_log(str(tmp_path / "nope.log")) is False


def test_detect_1stage_positive(tmp_path):
    p = tmp_path / "s.log"
    p.write_text("boot\nusing 1stage default kernel\n")
    assert _detect_1stage_from_log(str(p)) is True


def test_detect_1stage_negative(tmp_path):
    p = tmp_path / "s.log"
    p.write_text("boot\nnormal 2stage\n")
    assert _detect_1stage_from_log(str(p)) is False


# ── _normalize_quant_type / _resolve_quant_type ──────────────────────────────
def test_normalize_quant_aliases():
    assert _normalize_quant_type("w8a8") == "per_token"
    assert _normalize_quant_type("per_1x128") == "blockscale"
    assert _normalize_quant_type("a4w4") == "fp4"
    assert _normalize_quant_type("custom") == "custom"


def test_resolve_quant_awq_gptq():
    assert _resolve_quant_type("fp8", "auto", _profile(quant_method="awq"), None) == "awq"
    assert _resolve_quant_type("fp8", "auto", _profile(quant_method="gptq"), None) == "gptq"


def test_resolve_quant_fp8_log_per_token(tmp_path):
    log = tmp_path / "k.log"
    log.write_text("QuantType.per_Token detected\n")
    assert _resolve_quant_type("fp8", "auto", _profile(), str(log)) == "per_token"


def test_resolve_quant_fp8_log_blockscale(tmp_path):
    log = tmp_path / "k.log"
    log.write_text("QuantType.per_1x128\n")
    assert _resolve_quant_type("fp8", "auto", _profile(), str(log)) == "blockscale"


def test_resolve_quant_fp8_log_bpreshuffle(tmp_path):
    log = tmp_path / "k.log"
    log.write_text("uses bpreshuffle kernels\n")
    assert _resolve_quant_type("fp8", "auto", _profile(), str(log)) == "bpreshuffle"


def test_resolve_quant_fp8_default_blockscale():
    assert _resolve_quant_type("fp8", "auto", _profile(), None) == "blockscale"


def test_resolve_quant_fp4_and_bf16():
    assert _resolve_quant_type("fp4", "auto", _profile(), None) == "fp4"
    assert _resolve_quant_type("mxfp4", "auto", _profile(), None) == "fp4"
    assert _resolve_quant_type("bf16", "auto", _profile(), None) == "none"
    assert _resolve_quant_type("int8", "auto", _profile(), None) == "none"


# ── sglang MoE branches ──────────────────────────────────────────────────────
def test_sglang_moe_per_token_1stage_log_skips(tmp_path):
    # per_token + non-fp8 precision + 1-stage log -> skip reason set.
    log = tmp_path / "s.log"
    log.write_text("using 1stage default\n")
    profile = _profile(is_moe=True, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768)
    specs = select_tuners(
        profile, framework="sglang", precision="bf16", quant_type="per_token", kernel_signature_log=str(log)
    )
    fmoe = [s for s in specs if s.name == "fmoe_ck"][0]
    assert not fmoe.should_run and "1-stage" in fmoe.skip_reason


def test_sglang_moe_per_token_no_1stage_runs():
    profile = _profile(is_moe=True, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768)
    specs = select_tuners(profile, framework="sglang", precision="bf16", quant_type="per_token")
    fmoe = [s for s in specs if s.name == "fmoe_ck"][0]
    assert fmoe.should_run


def test_sglang_moe_unsupported_combo_skips():
    profile = _profile(is_moe=True, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768)
    specs = select_tuners(profile, framework="sglang", precision="int8", quant_type="awq")
    fmoe = [s for s in specs if s.name == "fmoe_ck"][0]
    assert not fmoe.should_run and "Unsupported" in fmoe.skip_reason


# ── vllm dense paths ─────────────────────────────────────────────────────────
def test_vllm_dense_with_tunableop_input():
    profile = _profile(is_moe=False)
    specs = select_tuners(profile, framework="vllm", precision="bf16", has_tunableop_input=True)
    names = [s.name for s in specs if s.should_run]
    assert "vllm_dense_tunableop" in names


def test_vllm_dense_only_no_input_skips():
    profile = _profile(is_moe=False)
    specs = select_tuners(profile, framework="vllm", precision="bf16")
    dense = [s for s in specs if s.name == "vllm_dense_tunableop"][0]
    assert not dense.should_run and "TunableOp" in dense.skip_reason
