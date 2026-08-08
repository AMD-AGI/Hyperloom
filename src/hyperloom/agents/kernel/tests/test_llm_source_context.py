###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Guards on the runtime context handed to the source-resolution model tiers.

The context exists so a model can tell apart two same-named candidates by the
backend that is actually selected. That usefulness is only worth having if the
block cannot carry a credential out of the host, and cannot grow without bound.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from _llm_source_context import (  # noqa: E402
    _MAX_CONTEXT_CHARS,
    build_context_block,
    launcher_stack,
    model_config_summary,
    runtime_flags,
    server_arg_flags,
)


# --- secrets ------------------------------------------------------------------


def test_credentials_never_reach_the_context():
    """An allowlist decides what is sent; secrets are refused on top of that."""
    env = {
        "ANTHROPIC_API_KEY": "ak-secret",
        "OPENAI_API_KEY": "sk-secret",
        "HF_TOKEN": "hf-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "SGLANG_USE_AITER": "1",
    }
    block = build_context_block(server_args="--tp 8", framework_roots=("/repo",), env=env)
    for secret in ("ak-secret", "sk-secret", "hf-secret", "aws-secret"):
        assert secret not in block, secret
    assert "SGLANG_USE_AITER" in block


def test_a_secret_named_variable_is_dropped_even_if_allowlisted(monkeypatch):
    """The name pattern is a backstop against a careless allowlist edit."""
    monkeypatch.setattr(
        "_llm_source_context._ENV_ALLOWLIST",
        ("TP", "some_api_key", "Authorization"),
    )
    out = runtime_flags(
        None,
        {"TP": "8", "some_api_key": "leak-me", "Authorization": "leak-too"},
    )
    assert out["env"] == {"TP": "8"}


def test_unlisted_variables_are_not_forwarded():
    """Only path-selecting names travel, not whatever happens to be exported."""
    out = runtime_flags(None, {"MY_CUSTOM_VAR": "x", "SGLANG_USE_AITER": "1"})
    assert out["env"] == {"SGLANG_USE_AITER": "1"}


# --- serving command line -----------------------------------------------------


def test_the_raw_server_command_line_is_never_forwarded():
    """EXTRA_*_ARGS carries credentials and paths; only selectors may travel."""
    raw = "--api-key sk-supersecret --moe-runner-backend triton --model-path /data/private"
    block = build_context_block(server_args=raw, env={})
    assert "sk-supersecret" not in block
    assert "api-key" not in block
    assert "/data/private" not in block
    assert "triton" in block


def test_runtime_selection_includes_framework_and_precision():
    """Dispatch-critical scalar context uses the same selector validator."""
    block = build_context_block(
        framework="sglang",
        precision="fp8_e4m3",
        env={},
    )
    assert '"framework": "sglang"' in block
    assert '"precision": "fp8_e4m3"' in block


def test_a_denied_flags_value_is_consumed_with_it():
    """Skipping only the flag would leave its value as a stray bare token."""
    assert server_arg_flags("--api-key sk-leak --tp-size 8") == {"tp-size": "8"}


def test_a_credential_shaped_value_is_dropped_from_an_allowed_flag():
    """The name allowlist cannot see a secret passed as an innocuous value."""
    assert server_arg_flags("--quantization sk-abcdefghijklmnop") == {}
    assert server_arg_flags("--dtype " + "A" * 50) == {}


def test_structured_secret_values_are_dropped_from_allowed_flags():
    """URLs, authorization headers, JWTs, and controls are not selectors."""
    unsafe_args = (
        "--dtype https://user:pass@host/v1?api_key=secret",
        "--dtype 'authorization: Basic dXNlcjpwYXNz'",
        "--dtype 'basic dXNlcjpwYXNz'",
        "--dtype 'bearer lower-case-secret'",
        "--dtype eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature",
        "--dtype 'triton\ninjected'",
    )
    for server_args in unsafe_args:
        assert server_arg_flags(server_args) == {}


def test_an_unparseable_command_line_yields_nothing():
    """A partial parse of an unbalanced line risks emitting a fragment."""
    assert server_arg_flags('--moe-runner-backend "unclosed') == {}


def test_inline_and_bare_flag_forms():
    """``--flag=value`` and a valueless flag are both understood."""
    assert server_arg_flags("--attention-backend=aiter --api-key=sk-no") == {
        "attention-backend": "aiter"
    }
    assert server_arg_flags("--enable-torch-compile") == {"enable-torch-compile": True}


def test_common_selector_values_are_preserved():
    """Known backend, dtype, quantization, and numeric-list forms remain valid."""
    out = server_arg_flags(
        "--attention-backend aiter "
        "--decode-attention-backend triton "
        "--prefill-attention-backend flashinfer "
        "--dtype fp8_e4m3 "
        "--kv-cache-dtype torch.bfloat16 "
        "--quantization compressed-tensors"
    )
    assert out == {
        "attention-backend": "aiter",
        "decode-attention-backend": "triton",
        "prefill-attention-backend": "flashinfer",
        "dtype": "fp8_e4m3",
        "kv-cache-dtype": "torch.bfloat16",
        "quantization": "compressed-tensors",
    }
    assert runtime_flags(None, {"HIP_VISIBLE_DEVICES": "0,1"}) == {
        "env": {"HIP_VISIBLE_DEVICES": "0,1"}
    }


# --- model config ---------------------------------------------------------------


def test_config_summary_keeps_only_path_selecting_fields(tmp_path):
    cfg = {
        "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
        "model_type": "minimax_m3_vl",
        "quantization_config": {
            "quant_method": "mxfp8",
            "weight_block_size": [128, 128],
            "ignored": ["a", "b"],
        },
        "vocab_size": 200000,
        "bos_token_id": 1,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    got = model_config_summary(tmp_path)
    assert got["model_type"] == "minimax_m3_vl"
    assert got["quantization_config"] == {
        "quant_method": "mxfp8",
        "weight_block_size": [128, 128],
    }
    # Tokenizer-ish fields are noise for deciding which kernel file runs.
    assert "vocab_size" not in got
    assert "bos_token_id" not in got


def test_quantization_config_drops_unknown_and_secret_fields(tmp_path):
    """Only explicit selectors may cross the model-config egress boundary."""
    cfg = {
        "quantization_config": {
            "quant_method": "mxfp8",
            "api_key": "sk-must-not-leave",
            "vendor_metadata": "private",
            "weight_dtype": "sk-secret-in-an-allowed-field",
            "nested": {"hf_token": "hf_must_not_leave"},
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    got = model_config_summary(tmp_path)
    assert got["quantization_config"] == {"quant_method": "mxfp8"}
    rendered = json.dumps(got)
    for secret in ("sk-must-not-leave", "private", "sk-secret", "hf_must_not_leave"):
        assert secret not in rendered


def test_model_config_lists_are_bounded_and_redacted(tmp_path):
    """Allowlisted lists cannot crowd out later runtime context sections."""
    cfg = {
        "architectures": ["SafeArchitecture", "sk-secret"] + [f"Arch{i}" for i in range(30)],
        "quantization_config": {"weight_block_size": list(range(30))},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    got = model_config_summary(tmp_path)
    assert got["architectures"] == ["SafeArchitecture"] + [f"Arch{i}" for i in range(14)]
    assert got["quantization_config"]["weight_block_size"] == list(range(16))


def test_config_selectors_drop_unsafe_items_and_non_finite_numbers(tmp_path):
    """Config lists and quantization selectors apply the same scalar policy."""
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature"
    cfg = {
        "architectures": [
            "SafeArchitecture",
            "https://user:pass@host/v1?api_key=secret",
            "authorization: Basic dXNlcjpwYXNz",
            "bearer lower-case-secret",
            jwt,
            "triton\x00injected",
            "NaN",
            "Infinity",
        ],
        "torch_dtype": float("nan"),
        "num_experts": True,
        "num_attention_heads": 8,
        "head_dim": 64.5,
        "quantization_config": {
            "quant_method": [
                "mxfp8",
                "https://user:pass@host/v1?api_key=secret",
                "authorization: Basic dXNlcjpwYXNz",
                "bearer lower-case-secret",
                jwt,
                "nan",
                "infinity",
            ],
            "bits": float("inf"),
            "compute_dtype": float("-inf"),
            "weight_block_size": [128, float("nan"), float("inf"), 256],
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    got = model_config_summary(tmp_path)
    assert got == {
        "architectures": ["SafeArchitecture"],
        "num_experts": True,
        "num_attention_heads": 8,
        "head_dim": 64.5,
        "quantization_config": {
            "quant_method": ["mxfp8"],
            "weight_block_size": [128, 256],
        },
    }
    rendered = json.dumps(got)
    assert "NaN" not in rendered
    assert "Infinity" not in rendered


def test_missing_or_broken_config_degrades_quietly(tmp_path):
    assert model_config_summary(None) == {}
    assert model_config_summary(tmp_path) == {}
    (tmp_path / "config.json").write_text("{ not json", encoding="utf-8")
    assert model_config_summary(tmp_path) == {}


# --- launcher stack --------------------------------------------------------------


def test_launcher_stack_falls_back_to_the_single_resolved_frame():
    entry = {
        "source_file": "/repo/moe.py",
        "source_line": 247,
        "source_function": "_grouped_gemm",
    }
    assert launcher_stack(entry) == ["/repo/moe.py(247): _grouped_gemm"]


def test_launcher_stack_is_empty_without_a_call_site():
    assert launcher_stack({"source_file": "/repo/moe.py"}) == []


# --- size and degradation ----------------------------------------------------------


def test_context_is_capped():
    huge = "--flag " * 5000
    block = build_context_block(server_args=huge, framework_roots=("/repo",), env={})
    assert len(block) <= _MAX_CONTEXT_CHARS


def test_no_available_context_yields_an_empty_block():
    """Nothing to say means say nothing, rather than an empty scaffold."""
    assert build_context_block(env={}) == ""
