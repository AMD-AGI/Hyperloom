# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import pytest

from hyperloom.common.llm_config import (
    LLMConfigError,
    apply_reasoning_effort,
    derive_openai_base_url,
    openai_client_kwargs,
    parse_custom_headers,
)


def test_apply_reasoning_effort_is_noop_without_env():
    params = {"model": "m", "messages": []}
    out = apply_reasoning_effort(params, env={})
    assert "reasoning_effort" not in out
    assert out is params  # mutates in place


def test_apply_reasoning_effort_injects_recognized_value():
    out = apply_reasoning_effort({"model": "m"}, env={"HYPERLOOM_REASONING_EFFORT": "MEDIUM"})
    assert out["reasoning_effort"] == "medium"
    out2 = apply_reasoning_effort({"model": "m"}, env={"OPENAI_REASONING_EFFORT": "high"})
    assert out2["reasoning_effort"] == "high"


def test_apply_reasoning_effort_ignores_unknown_value():
    out = apply_reasoning_effort({"model": "m"}, env={"HYPERLOOM_REASONING_EFFORT": "turbo"})
    assert "reasoning_effort" not in out


def test_parse_custom_headers_accepts_anthropic_env_format():
    headers = parse_custom_headers("Ocp-Apim-Subscription-Key: ak-test\nX-Team: hyperloom")
    assert headers == {
        "Ocp-Apim-Subscription-Key": "ak-test",
        "X-Team": "hyperloom",
    }


def test_derive_openai_base_url_from_amd_anthropic_endpoint():
    assert (
        derive_openai_base_url("https://llm-api.amd.com/anthropic")
        == "https://llm-api.amd.com/Unified/v1"
    )


def test_openai_kwargs_from_anthropic_only_gateway_env():
    kwargs = openai_client_kwargs(
        env={
            "ANTHROPIC_API_KEY": "ak-anthropic",
            "ANTHROPIC_BASE_URL": "https://llm-api.amd.com/anthropic",
            "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: ak-header",
        }
    )
    assert kwargs["api_key"] == "ak-anthropic"
    assert kwargs["base_url"] == "https://llm-api.amd.com/Unified/v1"
    assert kwargs["default_headers"] == {"Ocp-Apim-Subscription-Key": "ak-header"}


def test_openai_kwargs_auto_adds_amd_subscription_header_when_missing():
    kwargs = openai_client_kwargs(
        env={
            "ANTHROPIC_API_KEY": "ak-anthropic",
            "ANTHROPIC_BASE_URL": "https://llm-api.amd.com/anthropic",
        }
    )
    assert kwargs["default_headers"] == {"Ocp-Apim-Subscription-Key": "ak-anthropic"}


def test_openai_kwargs_preserves_explicit_openai_config():
    kwargs = openai_client_kwargs(
        env={
            "OPENAI_API_KEY": "sk-openai",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "ANTHROPIC_API_KEY": "ak-anthropic",
            "ANTHROPIC_BASE_URL": "https://llm-api.amd.com/anthropic",
        }
    )
    assert kwargs == {
        "api_key": "sk-openai",
        "base_url": "https://api.openai.com/v1",
    }


def test_openai_kwargs_requires_a_key():
    with pytest.raises(LLMConfigError):
        openai_client_kwargs(env={"ANTHROPIC_BASE_URL": "https://llm-api.amd.com/anthropic"})
