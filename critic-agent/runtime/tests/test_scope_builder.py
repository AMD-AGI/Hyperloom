# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Scope construction tests."""

from __future__ import annotations

import pytest

from runtime.errors import ScopeError
from runtime.scope_builder import (
    CRITICAL_SCOPE_KEYS,
    OPTIONAL_SCOPE_KEYS,
    ORG_DEFAULT,
    build_scope,
    derive_model_family,
    scope_cache_key,
)


def test_org_defaults_to_hyperloom():
    s = build_scope(
        {"framework": "sglang", "model": "deepseek-r1", "model_family": "deepseek",
         "workload": "decode", "precision": "fp8"}
    )
    assert s["org"] == ORG_DEFAULT


def test_explicit_context_wins_over_session():
    s = build_scope(
        {"framework": "sglang", "model": "deepseek-r1"},
        session_context={"framework": "vllm", "model_family": "deepseek",
                         "workload": "decode", "precision": "fp8"},
    )
    assert s["framework"] == "sglang"
    assert s["model_family"] == "deepseek"


def test_normalises_value_trim_and_lowercase():
    s = build_scope(
        {"framework": "  SGLang ", "model": "DeepSeek-R1",
         "model_family": "DeepSeek", "workload": "Decode", "precision": "FP8"}
    )
    assert s["framework"] == "sglang"
    assert s["model"] == "deepseek-r1"
    assert s["precision"] == "fp8"


def test_unknown_treated_as_missing_and_filled_from_session():
    s = build_scope(
        {"framework": "unknown", "model": "deepseek-r1"},
        session_context={"framework": "sglang"},
    )
    assert s["framework"] == "sglang"


def test_optional_keys_only_present_when_known():
    s_no_opt = build_scope(
        {"framework": "sglang", "model": "deepseek-r1", "model_family": "deepseek",
         "workload": "decode", "precision": "fp8"}
    )
    assert "scale" not in s_no_opt
    s_with_opt = build_scope(
        {"framework": "sglang", "model": "deepseek-r1", "model_family": "deepseek",
         "workload": "decode", "precision": "fp8", "scale": "8xMI300", "objective": "throughput"}
    )
    assert s_with_opt["scale"] == "8xmi300"
    assert s_with_opt["objective"] == "throughput"


def test_critical_keys_missing_raises_by_default():
    with pytest.raises(ScopeError, match="model"):
        build_scope({"framework": "sglang"})


def test_require_critical_false_returns_unknown_placeholder():
    s = build_scope({"framework": "sglang"}, require_critical=False)
    assert s["model"] == "unknown"
    assert s["framework"] == "sglang"


def test_model_family_derived_from_model_when_missing():
    s = build_scope(
        {"framework": "sglang", "model": "deepseek-r1-0528-fp8",
         "workload": "decode", "precision": "fp8"}
    )
    assert s["model_family"] == "deepseek"


def test_model_family_unknown_when_no_rule_matches():
    s = build_scope(
        {"framework": "sglang", "model": "phantom-2b",
         "workload": "decode", "precision": "fp8"},
        require_critical=False,
    )
    assert s["model_family"] == "unknown"


def test_scope_cache_key_is_stable_across_dict_order():
    a = scope_cache_key({"model": "x", "framework": "y"}, topic="t")
    b = scope_cache_key({"framework": "y", "model": "x"}, topic="t")
    assert a == b


def test_critical_keys_constants():
    assert "model" in CRITICAL_SCOPE_KEYS
    assert "framework" in CRITICAL_SCOPE_KEYS
    assert OPTIONAL_SCOPE_KEYS == ("scale", "objective")


def test_derive_model_family_known_and_unknown():
    assert derive_model_family("Qwen3-14B") == "qwen"
    assert derive_model_family("Llama-4-405B") == "llama"
    assert derive_model_family("phantom-2b") == ""
