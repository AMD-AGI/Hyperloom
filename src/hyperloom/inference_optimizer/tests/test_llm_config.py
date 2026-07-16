# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import pytest

from hyperloom.common.llm_config import (
    LLMConfigError,
    apply_reasoning_effort,
    claude_sdk_env_options,
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


def test_parse_custom_headers_expands_env_references():
    headers = parse_custom_headers(
        "Ocp-Apim-Subscription-Key: ${ANTHROPIC_API_KEY}",
        env={"ANTHROPIC_API_KEY": "ak-from-env"},
    )
    assert headers == {"Ocp-Apim-Subscription-Key": "ak-from-env"}


def test_derive_openai_base_url_from_amd_anthropic_endpoint():
    assert (
        derive_openai_base_url("https://llm.example.invalid/anthropic")
        == "https://llm.example.invalid/Unified/v1"
    )


def test_openai_kwargs_reads_openai_custom_headers():
    """The OpenAI/Codex client applies OPENAI_CUSTOM_HEADERS verbatim."""
    kwargs = openai_client_kwargs(
        env={
            "_".join(("OPENAI", "API", "KEY")): "openai-token",
            "OPENAI_BASE_URL": "https://llm.example.invalid/Unified/v1",
            "OPENAI_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: ak-header",
        }
    )
    assert kwargs["default_headers"] == {"Ocp-Apim-Subscription-Key": "ak-header"}


def test_openai_kwargs_ignores_anthropic_custom_headers_and_host():
    """The OpenAI/Codex client reads only OPENAI_CUSTOM_HEADERS and does no
    host-based auto-injection. Base URL is still derived."""
    kwargs = openai_client_kwargs(
        env={
            "_".join(("ANTHROPIC", "API", "KEY")): "ak-anthropic",
            "ANTHROPIC_BASE_URL": "https://llm.example.invalid/anthropic",
            "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: ak-header",
        }
    )
    assert kwargs["api_key"] == "ak-anthropic"
    assert kwargs["base_url"] == "https://llm.example.invalid/Unified/v1"
    # Empty headers are omitted from kwargs entirely.
    assert "default_headers" not in kwargs


def test_openai_kwargs_preserves_explicit_openai_config():
    kwargs = openai_client_kwargs(
        env={
            "_".join(("OPENAI", "API", "KEY")): "openai-token",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "_".join(("ANTHROPIC", "API", "KEY")): "ak-anthropic",
            "ANTHROPIC_BASE_URL": "https://llm.example.invalid/anthropic",
        }
    )
    assert kwargs == {
        "api_key": "openai-token",
        "base_url": "https://api.openai.com/v1",
    }


def test_openai_kwargs_requires_a_key():
    with pytest.raises(LLMConfigError):
        openai_client_kwargs(env={"ANTHROPIC_BASE_URL": "https://llm.example.invalid/anthropic"})


def test_claude_sdk_env_options_from_deepseek_key_only():
    opts = claude_sdk_env_options(
        model="deepseek-chat",
        env={"_".join(("DEEPSEEK", "API", "KEY")): "deepseek-token"},
    )
    assert opts["setting_sources"] == []
    child_env = opts["env"]
    assert child_env["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert child_env["_".join(("ANTHROPIC", "API", "KEY"))] == "deepseek-token"
    assert child_env["_".join(("ANTHROPIC", "AUTH", "TOKEN"))] == "deepseek-token"
    assert child_env["ANTHROPIC_MODEL"] == "deepseek-chat"


def test_claude_sdk_env_options_keeps_explicit_deepseek_base_url():
    opts = claude_sdk_env_options(
        model="deepseek-chat",
        env={
            "_".join(("DEEPSEEK", "API", "KEY")): "deepseek-token",
            "DEEPSEEK_BASE_URL": "https://deepseek.example/anthropic",
        },
    )
    child_env = opts["env"]
    assert child_env["ANTHROPIC_BASE_URL"] == "https://deepseek.example/anthropic"


def test_claude_sdk_env_options_forwards_anthropic_custom_headers():
    """The Claude subprocess forwards ANTHROPIC_CUSTOM_HEADERS verbatim."""
    opts = claude_sdk_env_options(
        env={
            "_".join(("ANTHROPIC", "API", "KEY")): "ak-anthropic",
            "ANTHROPIC_BASE_URL": "https://llm.example.invalid/anthropic",
            "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: operator-key",
        }
    )
    headers = parse_custom_headers(opts["env"]["ANTHROPIC_CUSTOM_HEADERS"])
    assert headers["Ocp-Apim-Subscription-Key"] == "operator-key"


def test_claude_sdk_env_options_expands_anthropic_custom_header_reference():
    opts = claude_sdk_env_options(
        env={
            "_".join(("ANTHROPIC", "API", "KEY")): "ak-anthropic",
            "ANTHROPIC_BASE_URL": "https://llm-api.amd.com/anthropic",
            "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: ${ANTHROPIC_API_KEY}",
        }
    )
    headers = parse_custom_headers(opts["env"]["ANTHROPIC_CUSTOM_HEADERS"])
    assert headers["Ocp-Apim-Subscription-Key"] == "ak-anthropic"


def test_claude_sdk_env_options_no_header_auto_injection():
    """A gateway without an explicit ANTHROPIC_CUSTOM_HEADERS gets NO
    auto-injected subscription header."""
    opts = claude_sdk_env_options(
        env={
            "_".join(("ANTHROPIC", "API", "KEY")): "ak-anthropic",
            "ANTHROPIC_BASE_URL": "https://llm.example.invalid/anthropic",
        }
    )
    headers = parse_custom_headers(opts["env"].get("ANTHROPIC_CUSTOM_HEADERS"))
    assert "Ocp-Apim-Subscription-Key" not in headers


def test_claude_sdk_env_options_does_not_copy_openai_custom_headers():
    """OPENAI_CUSTOM_HEADERS is NOT copied onto the Claude (Anthropic) side; the
    claude path reads only ANTHROPIC_CUSTOM_HEADERS."""
    opts = claude_sdk_env_options(
        env={
            "_".join(("ANTHROPIC", "API", "KEY")): "ak-anthropic",
            "ANTHROPIC_BASE_URL": "https://llm.example.invalid/anthropic",
            "OPENAI_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: openai-key",
        }
    )
    assert "ANTHROPIC_CUSTOM_HEADERS" not in opts["env"]


def test_claude_sdk_env_options_disables_advisor_tool_by_default():
    """Claude Code's advisor-tool beta header (rejected by strict gateways) is
    disabled by default; an operator preset is preserved."""
    opts = claude_sdk_env_options(
        env={"_".join(("ANTHROPIC", "API", "KEY")): "ak-anthropic"}
    )
    assert opts["env"]["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] == "1"

    preset = claude_sdk_env_options(
        env={
            "_".join(("ANTHROPIC", "API", "KEY")): "ak-anthropic",
            "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "0",
        }
    )
    assert preset["env"]["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] == "0"
