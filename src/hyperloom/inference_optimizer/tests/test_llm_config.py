# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import hyperloom
from hyperloom.common import llm_config
from hyperloom.common.llm_config import (
    ANTHROPIC_CREDENTIAL_ENV_ORDER,
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_ANTHROPIC_VERSION,
    LLMConfigError,
    aanthropic_messages,
    achat_completion,
    chat_completion,
    anthropic_messages,
    apply_reasoning_effort,
    aresponse,
    build_http_timeout,
    claude_sdk_env_options,
    deepseek_compat_env,
    derive_openai_base_url,
    get_anthropic_client,
    has_anthropic_side,
    has_openai_side,
    is_anthropic_only,
    is_openai_only,
    get_async_anthropic_client,
    get_async_openai_client,
    get_openai_client,
    openai_client_kwargs,
    parse_custom_headers,
    provider_model_defaults,
    resolve_forge_llm_model,
)

_LEGACY_KEY = "_".join(("DEEPSEEK", "API", "KEY"))
_OPENAI_KEY = "_".join(("OPENAI", "API", "KEY"))


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
    assert derive_openai_base_url("https://llm.example.invalid/anthropic") == "https://llm.example.invalid/Unified/v1"


def test_derive_openai_base_url_is_case_insensitive():
    # AMD's default endpoint uses a capitalized "/Anthropic" segment (issue #929);
    # match case-insensitively so the OpenAI base URL is still derived.
    assert derive_openai_base_url("https://llm-api.amd.com/Anthropic") == "https://llm-api.amd.com/Unified/v1"
    # A lowercase "/unified" segment is normalized to the canonical "/Unified/v1".
    assert derive_openai_base_url("https://llm-api.amd.com/unified") == "https://llm-api.amd.com/Unified/v1"


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


def test_openai_kwargs_ignores_anthropic_custom_headers():
    """The OpenAI/Codex client never reads ANTHROPIC_CUSTOM_HEADERS, and explicit
    OpenAI-side config wins over the Anthropic host."""
    kwargs = openai_client_kwargs(
        env={
            "_".join(("OPENAI", "API", "KEY")): "openai-token",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "_".join(("ANTHROPIC", "API", "KEY")): "ak-anthropic",
            "ANTHROPIC_BASE_URL": "https://llm.example.invalid/anthropic",
            "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: ak-header",
        }
    )
    assert kwargs["api_key"] == "openai-token"
    assert kwargs["base_url"] == "https://api.openai.com/v1"
    # Empty headers are omitted from kwargs entirely.
    assert "default_headers" not in kwargs


# ---- the three deployment shapes ----
# Anthropic-only, codex-only (OpenAI-compatible only), and both configured.
_ANTHROPIC_ONLY_ENV = {
    "_".join(("ANTHROPIC", "AUTH", "TOKEN")): "gateway-token",
    "_".join(("ANTHROPIC", "API", "KEY")): "ak-anthropic",
    "ANTHROPIC_BASE_URL": "https://llm-api.amd.com/Anthropic",
    # AMD's gateway rejects a call without this, and the setup skill only ever
    # writes it here -- so the shape is not realistic without it.
    "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: ${ANTHROPIC_API_KEY}",
}
_CODEX_ONLY_ENV = {
    "_".join(("OPENAI", "API", "KEY")): "openai-token",
    "OPENAI_BASE_URL": "https://llm-api.amd.com/Unified/v1",
}


def test_anthropic_only_deployment_resolves_through_derivation():
    """No OpenAI-side config at all: derive the base URL, auth with the gateway token."""
    kwargs = openai_client_kwargs(env=dict(_ANTHROPIC_ONLY_ENV))
    assert kwargs["base_url"] == "https://llm-api.amd.com/Unified/v1"
    assert kwargs["api_key"] == "gateway-token"


def test_anthropic_only_deployment_falls_back_to_anthropic_api_key():
    """ANTHROPIC_AUTH_TOKEN is preferred, but ANTHROPIC_API_KEY alone also authenticates."""
    env = dict(_ANTHROPIC_ONLY_ENV)
    del env["_".join(("ANTHROPIC", "AUTH", "TOKEN"))]
    assert openai_client_kwargs(env=env)["api_key"] == "ak-anthropic"


def test_codex_only_deployment_is_unchanged():
    """No Anthropic side present: resolution is exactly the explicit OpenAI config."""
    assert openai_client_kwargs(env=dict(_CODEX_ONLY_ENV)) == {
        "api_key": "openai-token",
        "base_url": "https://llm-api.amd.com/Unified/v1",
    }


def test_both_configured_prefers_the_explicit_openai_side():
    """With both shapes present the Anthropic fallback must never be consulted."""
    kwargs = openai_client_kwargs(env={**_ANTHROPIC_ONLY_ENV, **_CODEX_ONLY_ENV})
    assert kwargs == {
        "api_key": "openai-token",
        "base_url": "https://llm-api.amd.com/Unified/v1",
    }


def test_explicit_openai_side_wins_key_and_url_independently():
    """A half-configured OpenAI side takes what it has and derives only the rest."""
    key_only = openai_client_kwargs(env={**_ANTHROPIC_ONLY_ENV, "_".join(("OPENAI", "API", "KEY")): "openai-token"})
    assert key_only["api_key"] == "openai-token"
    assert key_only["base_url"] == "https://llm-api.amd.com/Unified/v1"

    url_only = openai_client_kwargs(
        env={**_ANTHROPIC_ONLY_ENV, "OPENAI_BASE_URL": "https://explicit.example.invalid/v1"}
    )
    assert url_only["api_key"] == "gateway-token"
    assert url_only["base_url"] == "https://explicit.example.invalid/v1"


def test_llm_gateway_key_still_outranks_the_anthropic_fallback():
    env = {**_ANTHROPIC_ONLY_ENV, "LLM_GATEWAY_KEY": "gw-key"}
    assert openai_client_kwargs(env=env)["api_key"] == "gw-key"


_SUBSCRIPTION_HEADER = "Ocp-Apim-Subscription-Key"


# ---- credential-shape predicates ----
def test_shape_predicates_classify_all_three_deployments():
    """One shape test for backend selection, the TraceLens runner and the kernel_backend."""
    assert is_anthropic_only(_ANTHROPIC_ONLY_ENV)
    assert not is_openai_only(_ANTHROPIC_ONLY_ENV)

    assert is_openai_only(_CODEX_ONLY_ENV)
    assert not is_anthropic_only(_CODEX_ONLY_ENV)

    both = {**_ANTHROPIC_ONLY_ENV, **_CODEX_ONLY_ENV}
    assert not is_anthropic_only(both)
    assert not is_openai_only(both)


def test_shape_predicates_report_no_side_on_an_empty_env():
    """Neither shape holds when nothing is configured; both sides read false."""
    assert not has_anthropic_side({})
    assert not has_openai_side({})
    assert not is_anthropic_only({})
    assert not is_openai_only({})


@pytest.mark.parametrize("key", ["ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"])
def test_any_anthropic_variable_alone_marks_that_side_configured(key):
    assert has_anthropic_side({key: "value"})


@pytest.mark.parametrize("key", ["OPENAI_BASE_URL", "OPENAI_API_KEY"])
def test_any_openai_variable_alone_marks_that_side_configured(key):
    assert has_openai_side({key: "value"})


def test_shape_predicates_ignore_the_retired_deepseek_variables():
    """DeepSeek is migrated onto the standard pair, so it is not a third side.

    The forge kernel backend used to read these directly and would therefore disagree
    with backend selection about a legacy-only configuration.
    """
    legacy = {_LEGACY_KEY: "dk", "DEEPSEEK_BASE_URL": "https://api.deepseek.com"}
    assert not has_anthropic_side(legacy)
    assert not has_openai_side(legacy)
    assert not is_anthropic_only(legacy)
    # With DeepSeek ignored, an OpenAI side alongside it is still openai-only --
    # the kernel backend previously read these keys and answered False here.
    assert is_openai_only({**legacy, **_CODEX_ONLY_ENV})


def test_derived_base_url_carries_the_anthropic_gateway_headers():
    """An anthropic-only deployment must still send the gateway's subscription key.

    The derived URL is the same host as ANTHROPIC_BASE_URL, so without this the
    run resolves a client that the gateway rejects on every call -- worse than
    the clean configuration error it used to raise.
    """
    kwargs = openai_client_kwargs(env=dict(_ANTHROPIC_ONLY_ENV))
    assert kwargs["base_url"] == "https://llm-api.amd.com/Unified/v1"
    assert kwargs["default_headers"] == {_SUBSCRIPTION_HEADER: "ak-anthropic"}


def test_explicit_openai_base_url_does_not_borrow_anthropic_headers():
    """An explicit OpenAI URL may be a different host, whose headers we cannot guess."""
    kwargs = openai_client_kwargs(env={**_ANTHROPIC_ONLY_ENV, "OPENAI_BASE_URL": "https://other-host.example/v1"})
    assert kwargs["base_url"] == "https://other-host.example/v1"
    assert "default_headers" not in kwargs


def test_openai_custom_headers_win_over_the_derived_gateway_headers():
    """An operator who set the OpenAI side explicitly keeps that exact header set."""
    kwargs = openai_client_kwargs(
        env={**_ANTHROPIC_ONLY_ENV, "OPENAI_CUSTOM_HEADERS": f"{_SUBSCRIPTION_HEADER}: openai-sub"}
    )
    assert kwargs["default_headers"] == {_SUBSCRIPTION_HEADER: "openai-sub"}


def test_openai_kwargs_error_names_every_searched_key():
    """The message has to list what was actually searched, Anthropic side included."""
    with pytest.raises(LLMConfigError) as excinfo:
        openai_client_kwargs(env={"ANTHROPIC_BASE_URL": "https://llm.example.invalid/anthropic"})
    message = str(excinfo.value)
    for name in ("OPENAI_API_KEY", "LLM_GATEWAY_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
        assert name in message


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


def test_deepseek_compat_env_is_empty_without_legacy_vars():
    assert deepseek_compat_env({}) == {}
    assert deepseek_compat_env({"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}) == {}


def test_deepseek_compat_env_key_only_fills_both_protocol_sides():
    """One key, two endpoints: DeepSeek is a gateway, not a third provider."""
    updates = deepseek_compat_env({_LEGACY_KEY: "deepseek-token"})
    assert updates["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert updates["OPENAI_BASE_URL"] == "https://api.deepseek.com/v1"
    assert updates["_".join(("ANTHROPIC", "API", "KEY"))] == "deepseek-token"
    assert updates["_".join(("OPENAI", "API", "KEY"))] == "deepseek-token"
    assert updates["CLAUDE_MODEL"] == "deepseek-v4-pro"
    assert updates["CODEX_MODEL"] == "deepseek-v4-pro"
    assert updates["GEAK_CLAUDE_MODEL"] == "deepseek-v4-pro"


@pytest.mark.parametrize(
    ("legacy_url", "anthropic_url", "openai_url"),
    [
        ("https://gw.example/anthropic", "https://gw.example/anthropic", "https://gw.example/v1"),
        ("https://gw.example/v1", "https://gw.example/anthropic", "https://gw.example/v1"),
        ("https://gw.example/anthropic/", "https://gw.example/anthropic", "https://gw.example/v1"),
        ("https://gw.example", "https://gw.example", "https://gw.example/v1"),
    ],
)
def test_deepseek_compat_env_derives_the_sibling_endpoint(legacy_url, anthropic_url, openai_url):
    """The legacy URL named only one side; the other is derived, never guessed wrong."""
    updates = deepseek_compat_env({_LEGACY_KEY: "t", "DEEPSEEK_BASE_URL": legacy_url})
    assert updates["ANTHROPIC_BASE_URL"] == anthropic_url
    assert updates["OPENAI_BASE_URL"] == openai_url


def test_deepseek_compat_env_bare_known_host_gets_both_segments():
    """``https://api.deepseek.com`` is the documented OpenAI base, not a root."""
    updates = deepseek_compat_env({_LEGACY_KEY: "t", "DEEPSEEK_BASE_URL": "https://api.deepseek.com"})
    assert updates["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert updates["OPENAI_BASE_URL"] == "https://api.deepseek.com/v1"


@pytest.mark.parametrize(
    "configured",
    [
        {"_".join(("ANTHROPIC", "API", "KEY")): "sk-explicit"},
        {"_".join(("ANTHROPIC", "AUTH", "TOKEN")): "sk-explicit"},
        {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"},
    ],
)
def test_deepseek_compat_env_ignores_leftovers_once_anthropic_side_exists(configured):
    """A stale legacy key must never re-point an explicit Anthropic credential.

    Half-adopting the gateway would put ANTHROPIC_BASE_URL on DeepSeek's host
    while the operator's own key is what gets sent there, and would invent an
    OpenAI side they never asked for.
    """
    assert deepseek_compat_env({**configured, _LEGACY_KEY: "sk-legacy"}) == {}


def test_deepseek_compat_env_leaves_official_anthropic_endpoint_alone():
    """Regression: an explicit key alone still implies the OFFICIAL endpoint.

    ``_resolve_llm_endpoints`` fills ``ANTHROPIC_BASE_URL`` from an explicit
    Anthropic key; the shim must not get there first with DeepSeek's host.
    """
    env = {"_".join(("ANTHROPIC", "API", "KEY")): "sk-real-anthropic", _LEGACY_KEY: "sk-legacy"}
    updates = deepseek_compat_env(env)
    assert "ANTHROPIC_BASE_URL" not in updates
    assert "OPENAI_BASE_URL" not in updates
    assert "CLAUDE_MODEL" not in updates


def test_claude_sdk_env_options_does_not_mix_legacy_and_explicit_keys():
    """End-to-end guard for the same hazard through the Claude SDK options."""
    api_key_var = "_".join(("ANTHROPIC", "API", "KEY"))
    auth_token_var = "_".join(("ANTHROPIC", "AUTH", "TOKEN"))
    child_env = claude_sdk_env_options(
        env={
            api_key_var: "sk-explicit",
            "ANTHROPIC_BASE_URL": "https://llm.example.invalid/anthropic",
            _LEGACY_KEY: "sk-legacy",
        }
    )["env"]
    assert child_env[api_key_var] == "sk-explicit"
    assert child_env[auth_token_var] == "sk-explicit"


def test_provider_model_defaults_supplies_a_model_for_a_known_gateway():
    """A gateway serving only its own models must not get the AMD Claude id.

    This is the shape the docs now recommend: both sides pointed at DeepSeek
    with the standard variables and no ``DEEPSEEK_*`` anywhere.
    """
    defaults = provider_model_defaults(
        {
            "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
            "OPENAI_BASE_URL": "https://api.deepseek.com/v1",
            "_".join(("ANTHROPIC", "API", "KEY")): "sk-ds",
            "_".join(("OPENAI", "API", "KEY")): "sk-ds",
        }
    )
    assert defaults == {
        "CLAUDE_MODEL": "deepseek-v4-pro",
        "CODEX_MODEL": "deepseek-v4-pro",
        "GEAK_CLAUDE_MODEL": "deepseek-v4-pro",
    }


def test_provider_model_defaults_resolves_a_retired_config_too():
    """The legacy spelling must reach the same model, without preflight."""
    defaults = provider_model_defaults({_LEGACY_KEY: "sk-ds"})
    assert defaults["CLAUDE_MODEL"] == "deepseek-v4-pro"
    assert defaults["CODEX_MODEL"] == "deepseek-v4-pro"


def test_provider_model_defaults_never_overrides_explicit_models():
    defaults = provider_model_defaults(
        {
            "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
            "OPENAI_BASE_URL": "https://api.deepseek.com/v1",
            "CLAUDE_MODEL": "my-claude",
            "CODEX_MODEL": "my-codex",
        }
    )
    assert "CLAUDE_MODEL" not in defaults
    assert "CODEX_MODEL" not in defaults
    # GEAKv4 still follows the Anthropic-side model in effect.
    assert defaults["GEAK_CLAUDE_MODEL"] == "my-claude"


@pytest.mark.parametrize(
    "env",
    [
        {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"},
        {"OPENAI_BASE_URL": "https://api.openai.com/v1"},
        {
            "ANTHROPIC_BASE_URL": "https://llm.amd.example/Anthropic",
            "OPENAI_BASE_URL": "https://llm.amd.example/Unified/v1",
        },
        {},
    ],
)
def test_provider_model_defaults_is_silent_for_unknown_gateways(env):
    """Only hosts we know the catalog of get a model; everything else is left alone."""
    assert provider_model_defaults(env) == {}


def test_deepseek_compat_env_never_overrides_explicit_values():
    updates = deepseek_compat_env(
        {
            _LEGACY_KEY: "deepseek-token",
            "OPENAI_BASE_URL": "https://gateway.example/v1",
            "_".join(("OPENAI", "API", "KEY")): "gateway-token",
            "CLAUDE_MODEL": "claude-opus-4-8",
        }
    )
    assert "OPENAI_BASE_URL" not in updates
    assert "_".join(("OPENAI", "API", "KEY")) not in updates
    assert "CLAUDE_MODEL" not in updates
    assert updates["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"


def test_deepseek_compat_env_leaves_a_foreign_openai_side_alone():
    """Another gateway on the OpenAI side keeps its own key AND its own model.

    Replacing only ``CODEX_MODEL`` would leave that gateway being asked for
    ``deepseek-v4-pro``, which it does not serve.
    """
    updates = deepseek_compat_env(
        {
            _LEGACY_KEY: "sk-legacy",
            "OPENAI_BASE_URL": "https://gateway.example/v1",
            "_".join(("OPENAI", "API", "KEY")): "sk-gateway",
        }
    )
    assert "OPENAI_BASE_URL" not in updates
    assert "_".join(("OPENAI", "API", "KEY")) not in updates
    assert "CODEX_MODEL" not in updates
    # The Anthropic side is still free, so it is adopted.
    assert updates["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"


def test_deepseek_compat_env_geak_model_follows_explicit_claude_model():
    """GEAKv4 must not be pointed at a model the operator overrode."""
    updates = deepseek_compat_env({_LEGACY_KEY: "sk-legacy", "CLAUDE_MODEL": "claude-opus-5"})
    assert "CLAUDE_MODEL" not in updates
    assert updates["GEAK_CLAUDE_MODEL"] == "claude-opus-5"


def test_resolve_forge_llm_model_prefers_forge_env_over_orchestration():
    env = {
        "CLAUDE_MODEL": "claude-orchestration",
        "FORGE_CLAUDE_MODEL": "claude-forge-only",
        "CODEX_MODEL": "gpt-orchestration",
        "FORGE_CODEX_MODEL": "gpt-forge-only",
    }
    assert resolve_forge_llm_model("claude", env=env) == "claude-forge-only"
    assert resolve_forge_llm_model("codex", env=env) == "gpt-forge-only"


def test_resolve_forge_llm_model_falls_back_to_orchestration_and_default():
    assert resolve_forge_llm_model("claude", env={"CLAUDE_MODEL": "claude-orch"}) == "claude-orch"
    assert resolve_forge_llm_model("codex", env={}, default="gpt-default") == "gpt-default"
    assert (
        resolve_forge_llm_model(
            "claude",
            env={"FORGE_CLAUDE_MODEL": "claude-forge-only"},
            explicit="claude-payload",
        )
        == "claude-payload"
    )


def test_deepseek_compat_env_is_idempotent():
    env = {_LEGACY_KEY: "deepseek-token"}
    env.update(deepseek_compat_env(env))
    assert deepseek_compat_env(env) == {}


def test_claude_sdk_env_options_from_deepseek_key_only():
    opts = claude_sdk_env_options(
        model="deepseek-v4-pro",
        env={"_".join(("DEEPSEEK", "API", "KEY")): "deepseek-token"},
    )
    assert opts["setting_sources"] == []
    child_env = opts["env"]
    assert child_env["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert child_env["_".join(("ANTHROPIC", "API", "KEY"))] == "deepseek-token"
    assert child_env["_".join(("ANTHROPIC", "AUTH", "TOKEN"))] == "deepseek-token"
    assert child_env["ANTHROPIC_MODEL"] == "deepseek-v4-pro"


def test_claude_sdk_env_options_never_synthesizes_api_keys_from_oauth_token():
    """The subscription token must stay env-passthrough only.

    Either API-key var switches the Claude CLI out of subscription mode, so
    synthesizing one here would both re-bill the run and 401 it.
    """
    oauth_env = "_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN"))
    opts = claude_sdk_env_options(env={oauth_env: "sk-ant-oat01-fake"})
    child_env = opts["env"]
    assert child_env[oauth_env] == "sk-ant-oat01-fake"
    assert "_".join(("ANTHROPIC", "API", "KEY")) not in child_env
    assert "_".join(("ANTHROPIC", "AUTH", "TOKEN")) not in child_env


def test_claude_sdk_env_options_isolates_settings_for_oauth_only_env():
    """OAuth alone is a gateway signal, so the run still gets an isolated env."""
    opts = claude_sdk_env_options(env={"_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN")): "sk-ant-oat01-fake"})
    assert opts["setting_sources"] == []


def test_anthropic_credential_env_order_is_cli_precedence():
    """Highest precedence first; OAuth is live only when both key vars are unset."""
    assert ANTHROPIC_CREDENTIAL_ENV_ORDER == (
        "_".join(("ANTHROPIC", "API", "KEY")),
        "_".join(("ANTHROPIC", "AUTH", "TOKEN")),
        "_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN")),
    )


def test_claude_sdk_env_options_keeps_explicit_deepseek_base_url():
    opts = claude_sdk_env_options(
        model="deepseek-v4-pro",
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


# ---- client construction (get_openai_client / get_async_openai_client) ----
class _FakeSDKClient:
    """Stand-in for an ``openai`` SDK client class; records its kwargs."""

    kind = ""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeSyncSDKClient(_FakeSDKClient):
    kind = "OpenAI"


class _FakeAsyncSDKClient(_FakeSDKClient):
    kind = "AsyncOpenAI"


def _install_fake_openai_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the lazy ``import openai`` at recording stand-ins."""
    module = types.ModuleType("openai")
    module.OpenAI = _FakeSyncSDKClient
    module.AsyncOpenAI = _FakeAsyncSDKClient
    monkeypatch.setitem(sys.modules, "openai", module)


def test_get_openai_client_resolves_credentials_and_headers(monkeypatch):
    _install_fake_openai_sdk(monkeypatch)
    client = get_openai_client(
        env={
            _OPENAI_KEY: "openai-token",
            "OPENAI_BASE_URL": "https://llm.example.invalid/Unified/v1",
            "OPENAI_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: ak-header",
        }
    )
    assert client.kind == "OpenAI"
    assert client.kwargs == {
        "api_key": "openai-token",
        "base_url": "https://llm.example.invalid/Unified/v1",
        "default_headers": {"Ocp-Apim-Subscription-Key": "ak-header"},
    }


def test_get_async_openai_client_honours_custom_env_var_names(monkeypatch):
    _install_fake_openai_sdk(monkeypatch)
    client = get_async_openai_client(
        api_key_env="SCORER_KEY",
        base_url_env="SCORER_URL",
        env={"SCORER_KEY": "scorer-token", "SCORER_URL": "https://scorer.example.invalid/v1"},
    )
    assert client.kind == "AsyncOpenAI"
    assert client.kwargs["api_key"] == "scorer-token"
    assert client.kwargs["base_url"] == "https://scorer.example.invalid/v1"


def test_get_client_omits_timeout_when_unset(monkeypatch):
    """No ``timeout`` kwarg at all, so the SDK applies its own default."""
    _install_fake_openai_sdk(monkeypatch)
    assert "timeout" not in get_async_openai_client(env={_OPENAI_KEY: "tok"}).kwargs


def test_get_client_forwards_timeout_verbatim(monkeypatch):
    _install_fake_openai_sdk(monkeypatch)
    sentinel = object()
    client = get_async_openai_client(env={_OPENAI_KEY: "tok"}, timeout=sentinel)
    assert client.kwargs["timeout"] is sentinel


def test_get_client_is_never_cached(monkeypatch):
    """An AsyncOpenAI's httpx pool binds to one event loop, so sharing is unsafe."""
    _install_fake_openai_sdk(monkeypatch)
    env = {_OPENAI_KEY: "tok"}
    assert get_async_openai_client(env=env) is not get_async_openai_client(env=env)
    assert get_openai_client(env=env) is not get_openai_client(env=env)


def test_get_client_raises_clearly_without_the_openai_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(LLMConfigError, match="openai SDK not installed"):
        get_async_openai_client(env={_OPENAI_KEY: "tok"})
    with pytest.raises(LLMConfigError, match="openai SDK not installed"):
        get_openai_client(env={_OPENAI_KEY: "tok"})


def test_get_client_requires_some_gateway_key(monkeypatch):
    _install_fake_openai_sdk(monkeypatch)
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        get_async_openai_client(env={"OPENAI_BASE_URL": "https://api.openai.com/v1"})


# ---- Anthropic client construction ----
class _FakeHttpxClient:
    """Stand-in for an ``httpx`` client class; records its kwargs."""

    kind = ""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeSyncHttpxClient(_FakeHttpxClient):
    kind = "Client"


class _FakeAsyncHttpxClient(_FakeHttpxClient):
    kind = "AsyncClient"


def _install_fake_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the lazy ``import httpx`` at recording stand-ins."""
    module = types.ModuleType("httpx")
    module.Client = _FakeSyncHttpxClient
    module.AsyncClient = _FakeAsyncHttpxClient
    monkeypatch.setitem(sys.modules, "httpx", module)


_ANTHROPIC_KEY = "_".join(("ANTHROPIC", "API", "KEY"))
_ANTHROPIC_TOKEN = "_".join(("ANTHROPIC", "AUTH", "TOKEN"))


def test_get_anthropic_client_sets_auth_version_and_base_url(monkeypatch):
    _install_fake_httpx(monkeypatch)
    client = get_anthropic_client(
        env={_ANTHROPIC_KEY: "ak-anthropic", "ANTHROPIC_BASE_URL": "https://llm-api.amd.com/Anthropic/"}
    )
    assert client.kind == "Client"
    # The Anthropic base URL is used as-is; only the OpenAI side derives.
    assert client.kwargs["base_url"] == "https://llm-api.amd.com/Anthropic"
    assert client.kwargs["headers"]["x-api-key"] == "ak-anthropic"
    assert client.kwargs["headers"]["anthropic-version"] == DEFAULT_ANTHROPIC_VERSION
    assert client.kwargs["headers"]["Content-Type"] == "application/json"


def test_get_async_anthropic_client_defaults_the_public_base_url(monkeypatch):
    _install_fake_httpx(monkeypatch)
    client = get_async_anthropic_client(env={_ANTHROPIC_KEY: "ak-anthropic"})
    assert client.kind == "AsyncClient"
    assert client.kwargs["base_url"] == DEFAULT_ANTHROPIC_BASE_URL


def test_get_anthropic_client_falls_back_to_the_auth_token(monkeypatch):
    _install_fake_httpx(monkeypatch)
    client = get_anthropic_client(env={_ANTHROPIC_TOKEN: "gateway-token"})
    assert client.kwargs["headers"]["x-api-key"] == "gateway-token"


def test_get_anthropic_client_merges_custom_gateway_headers(monkeypatch):
    """AMD's gateway requires ANTHROPIC_CUSTOM_HEADERS; dropping them is a regression."""
    _install_fake_httpx(monkeypatch)
    client = get_anthropic_client(
        env={
            _ANTHROPIC_KEY: "ak-anthropic",
            "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: ${ANTHROPIC_API_KEY}\nX-Team: hyperloom",
        }
    )
    assert client.kwargs["headers"]["Ocp-Apim-Subscription-Key"] == "ak-anthropic"
    assert client.kwargs["headers"]["X-Team"] == "hyperloom"


def test_custom_headers_can_override_the_api_version(monkeypatch):
    """Merged last on purpose: a gateway pinned to another version stays reachable."""
    _install_fake_httpx(monkeypatch)
    client = get_anthropic_client(
        env={_ANTHROPIC_KEY: "ak-anthropic", "ANTHROPIC_CUSTOM_HEADERS": "anthropic-version: 2099-01-01"}
    )
    assert client.kwargs["headers"]["anthropic-version"] == "2099-01-01"


def test_get_anthropic_client_timeout_plumbing(monkeypatch):
    _install_fake_httpx(monkeypatch)
    assert "timeout" not in get_anthropic_client(env={_ANTHROPIC_KEY: "k"}).kwargs
    sentinel = object()
    assert get_async_anthropic_client(env={_ANTHROPIC_KEY: "k"}, timeout=sentinel).kwargs["timeout"] is sentinel


def test_get_anthropic_client_is_never_cached(monkeypatch):
    _install_fake_httpx(monkeypatch)
    env = {_ANTHROPIC_KEY: "k"}
    assert get_anthropic_client(env=env) is not get_anthropic_client(env=env)
    assert get_async_anthropic_client(env=env) is not get_async_anthropic_client(env=env)


def test_get_anthropic_client_requires_a_key(monkeypatch):
    _install_fake_httpx(monkeypatch)
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        get_anthropic_client(env={"ANTHROPIC_BASE_URL": "https://api.anthropic.com"})


def test_get_anthropic_client_raises_clearly_without_httpx(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(LLMConfigError, match="httpx not installed"):
        get_anthropic_client(env={_ANTHROPIC_KEY: "k"})
    with pytest.raises(LLMConfigError, match="httpx not installed"):
        get_async_anthropic_client(env={_ANTHROPIC_KEY: "k"})


# ---- anthropic_messages / aanthropic_messages ----
class _FakeAnthropicResponse:
    """One canned reply; ``error`` makes ``json()`` fail as a non-JSON body would."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        body: Any = None,
        text: str = "",
        error: BaseException | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = {} if body is None else body
        self._error = error
        self.text = text

    def json(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._body


class _FakeAnthropicTransport:
    """Records ``/v1/messages`` POSTs; the async twin awaits the same recorder."""

    def __init__(self, response: Any = None, *, error: BaseException | None = None) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def post(self, path: str, *, json: Any) -> Any:
        self.calls.append({"path": path, "json": json})
        if self._error is not None:
            raise self._error
        return self._response


class _FakeAsyncAnthropicTransport(_FakeAnthropicTransport):
    async def post(self, path: str, *, json: Any) -> Any:  # type: ignore[override]
        return _FakeAnthropicTransport.post(self, path, json=json)


_MESSAGE_BODY = {
    "content": [
        {"type": "text", "text": "hello "},
        {"type": "tool_use", "id": "x"},
        {"type": "text", "text": "world"},
    ],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 21, "output_tokens": 7},
}


def test_anthropic_messages_posts_to_the_messages_path_and_flattens():
    client = _FakeAnthropicTransport(_FakeAnthropicResponse(body=_MESSAGE_BODY))
    result = anthropic_messages(client, model="claude", messages=[{"role": "user", "content": "hi"}], max_tokens=8)
    assert result.text == "hello world"
    assert result.stop_reason == "end_turn"
    assert result.usage == {"input_tokens": 21, "output_tokens": 7}
    assert client.calls[0]["path"] == "/v1/messages"
    assert client.calls[0]["json"] == {
        "model": "claude",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
    }


@pytest.mark.asyncio
async def test_aanthropic_messages_flattens_the_same_way():
    client = _FakeAsyncAnthropicTransport(_FakeAnthropicResponse(body=_MESSAGE_BODY))
    result = await aanthropic_messages(client, model="claude", system="sys")
    assert (result.text, result.stop_reason) == ("hello world", "end_turn")
    assert client.calls[0]["json"] == {"model": "claude", "system": "sys"}


@pytest.mark.asyncio
async def test_aanthropic_messages_raises_on_non_2xx_with_status_and_body():
    client = _FakeAsyncAnthropicTransport(_FakeAnthropicResponse(status_code=401, body={}, text="unauthorized"))
    with pytest.raises(RuntimeError, match="status=401") as excinfo:
        await aanthropic_messages(client, model="claude")
    assert "unauthorized" in str(excinfo.value)


@pytest.mark.asyncio
async def test_aanthropic_messages_raises_on_non_json_body():
    client = _FakeAsyncAnthropicTransport(_FakeAnthropicResponse(error=ValueError("not json")))
    with pytest.raises(RuntimeError, match="non-JSON body"):
        await aanthropic_messages(client, model="claude")


@pytest.mark.asyncio
async def test_aanthropic_messages_propagates_transport_errors():
    client = _FakeAsyncAnthropicTransport(error=RuntimeError("gateway down"))
    with pytest.raises(RuntimeError, match="gateway down"):
        await aanthropic_messages(client, model="claude")


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, {"content": "nope"}])
async def test_aanthropic_messages_tolerates_unreadable_reply_shapes(body):
    """An unreadable reply (dict with missing/wrong-typed fields) yields empty text."""
    client = _FakeAsyncAnthropicTransport(_FakeAnthropicResponse(body=body))
    result = await aanthropic_messages(client, model="claude")
    assert result.text == ""
    assert result.stop_reason is None
    assert result.usage is None


@pytest.mark.asyncio
async def test_aanthropic_messages_raises_on_non_dict_body():
    """A JSON response whose top-level type is not dict raises RuntimeError."""
    client = _FakeAsyncAnthropicTransport(_FakeAnthropicResponse(body=["not", "a", "dict"]))
    with pytest.raises(RuntimeError, match="non-object"):
        await aanthropic_messages(client, model="claude")


def test_anthropic_version_is_defined_once_in_llm_config():
    """The header value must live here alone; four modules used to hardcode it."""
    source = Path(llm_config.__file__).read_text(encoding="utf-8")
    assert source.count('"2023-06-01"') == 1
    assert DEFAULT_ANTHROPIC_VERSION == "2023-06-01"


# ---- build_http_timeout ----
def test_build_http_timeout_defaults_write_and_pool_to_read():
    timeout = build_http_timeout(connect=3.0, read=7.0)
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (3.0, 7.0, 7.0, 7.0)


def test_build_http_timeout_takes_explicit_write_and_pool():
    timeout = build_http_timeout(connect=1.0, read=2.0, write=3.0, pool=4.0)
    assert (timeout.write, timeout.pool) == (3.0, 4.0)


def test_build_http_timeout_without_httpx_degrades_to_sdk_defaults(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "httpx", None)
    with caplog.at_level(logging.WARNING, logger="hyperloom.common.llm_config"):
        assert build_http_timeout(connect=1.0, read=2.0) is None
    assert "httpx unavailable" in caplog.text


# ---- achat_completion ----
class _StubCompletions:
    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome
        self.calls: list[dict[str, Any]] = []

    async def create(self, **params: Any) -> Any:
        self.calls.append(params)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class _StubClient:
    def __init__(self, outcome: Any) -> None:
        self.completions = _StubCompletions(outcome)
        self.chat = types.SimpleNamespace(completions=self.completions)


@pytest.mark.asyncio
async def test_achat_completion_flattens_choice_and_keeps_usage_verbatim():
    usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=7)
    choice = types.SimpleNamespace(message=types.SimpleNamespace(content="hello"), finish_reason="stop")
    client = _StubClient(types.SimpleNamespace(choices=[choice], usage=usage))
    result = await achat_completion(client, model="m", messages=[])
    assert (result.text, result.finish_reason) == ("hello", "stop")
    assert result.usage is usage
    assert client.completions.calls == [{"model": "m", "messages": []}]


@pytest.mark.asyncio
async def test_achat_completion_tolerates_missing_content_finish_and_usage():
    choice = types.SimpleNamespace(message=types.SimpleNamespace(content=None))
    client = _StubClient(types.SimpleNamespace(choices=[choice]))
    result = await achat_completion(client, model="m")
    assert result.text == ""
    assert result.finish_reason is None
    assert result.usage is None


@pytest.mark.asyncio
async def test_achat_completion_propagates_transport_errors():
    """A dead gateway must reach the caller that tags it with role context."""
    with pytest.raises(RuntimeError, match="gateway down"):
        await achat_completion(_StubClient(RuntimeError("gateway down")), model="m")


# ---- chat_completion ----
class _SyncStubCompletions:
    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def create(self, **params: Any) -> Any:
        self.calls.append(params)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class _SyncStubClient:
    def __init__(self, outcome: Any) -> None:
        self.completions = _SyncStubCompletions(outcome)
        self.chat = types.SimpleNamespace(completions=self.completions)


def test_chat_completion_flattens_choice_and_keeps_usage_verbatim():
    usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=7)
    choice = types.SimpleNamespace(message=types.SimpleNamespace(content="hello"), finish_reason="stop")
    client = _SyncStubClient(types.SimpleNamespace(choices=[choice], usage=usage))
    result = chat_completion(client, model="m", messages=[])
    assert (result.text, result.finish_reason) == ("hello", "stop")
    assert result.usage is usage
    assert client.completions.calls == [{"model": "m", "messages": []}]


def test_chat_completion_tolerates_missing_content_finish_and_usage():
    choice = types.SimpleNamespace(message=types.SimpleNamespace(content=None))
    client = _SyncStubClient(types.SimpleNamespace(choices=[choice]))
    result = chat_completion(client, model="m")
    assert result.text == ""
    assert result.finish_reason is None
    assert result.usage is None


def test_chat_completion_propagates_transport_errors():
    """A dead gateway must reach the caller that tags it with role context."""
    with pytest.raises(RuntimeError, match="gateway down"):
        chat_completion(_SyncStubClient(RuntimeError("gateway down")), model="m")


def test_chat_completion_does_not_request_a_stream():
    """The non-streaming entry point must not turn into a streamed request.

    Callers such as the breakdown reporter depend on the plain request shape,
    which some gateways treat differently from a streamed one.
    """
    choice = types.SimpleNamespace(message=types.SimpleNamespace(content="x"))
    client = _SyncStubClient(types.SimpleNamespace(choices=[choice]))
    chat_completion(client, model="m", messages=[])
    assert "stream" not in client.completions.calls[0]


# ---- aresponse ----
class _StubResponses:
    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome
        self.calls: list[dict[str, Any]] = []

    async def create(self, **params: Any) -> Any:
        self.calls.append(params)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class _StubResponsesClient:
    def __init__(self, outcome: Any) -> None:
        self.responses = _StubResponses(outcome)


def _message_item(text: str, *, citations: list[str] | None = None) -> dict[str, Any]:
    annotations = [{"type": "url_citation", "url": u} for u in (citations or [])]
    return {
        "type": "message",
        "content": [{"type": "output_text", "text": text, "annotations": annotations}],
    }


@pytest.mark.asyncio
async def test_aresponse_flattens_text_citations_status_and_tokens():
    resp = {
        "output": [
            {"type": "reasoning", "content": []},
            {"type": "web_search_call", "action": {"query": "x"}},
            _message_item("the answer", citations=["https://a.example", "https://b.example"]),
        ],
        "status": "completed",
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }
    client = _StubResponsesClient(resp)
    result = await aresponse(client, model="m", input="prompt")
    assert result.text == "the answer"
    assert result.citations == ["https://a.example", "https://b.example"]
    assert result.status == "completed"
    assert (result.input_tokens, result.output_tokens) == (11, 7)
    assert client.responses.calls == [{"model": "m", "input": "prompt"}]


@pytest.mark.asyncio
async def test_aresponse_joins_multiple_message_items_with_newlines():
    resp = {"output": [_message_item("first"), _message_item("second")]}
    result = await aresponse(_StubResponsesClient(resp), model="m")
    assert result.text == "first\nsecond"


@pytest.mark.asyncio
async def test_aresponse_skips_non_output_text_content_blocks():
    resp = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "refusal", "text": "ignored"},
                    {"type": "output_text", "text": "kept", "annotations": []},
                ],
            }
        ]
    }
    result = await aresponse(_StubResponsesClient(resp), model="m")
    assert (result.text, result.citations) == ("kept", [])


@pytest.mark.asyncio
@pytest.mark.parametrize("resp", [{"output": []}, {}])
async def test_aresponse_tolerates_empty_output_and_missing_usage(resp):
    result = await aresponse(_StubResponsesClient(resp), model="m")
    assert result.text == ""
    assert result.citations == []
    assert result.status is None
    assert (result.input_tokens, result.output_tokens) == (0, 0)


@pytest.mark.asyncio
async def test_aresponse_reads_attribute_carrying_sdk_objects():
    """The real SDK returns pydantic models, not dicts."""
    block = types.SimpleNamespace(
        type="output_text",
        text="from sdk",
        annotations=[types.SimpleNamespace(type="url_citation", url="https://sdk.example")],
    )
    resp = types.SimpleNamespace(
        output=[types.SimpleNamespace(type="message", content=[block])],
        status="incomplete",
        usage=types.SimpleNamespace(input_tokens=3, output_tokens=4),
    )
    result = await aresponse(_StubResponsesClient(resp), model="m")
    assert result.text == "from sdk"
    assert result.citations == ["https://sdk.example"]
    assert result.status == "incomplete"
    assert (result.input_tokens, result.output_tokens) == (3, 4)


@pytest.mark.asyncio
async def test_aresponse_treats_unusable_token_counts_as_zero():
    resp = {"output": [], "usage": {"input_tokens": "abc", "output_tokens": None}}
    result = await aresponse(_StubResponsesClient(resp), model="m")
    assert (result.input_tokens, result.output_tokens) == (0, 0)


@pytest.mark.asyncio
async def test_aresponse_propagates_transport_errors():
    with pytest.raises(RuntimeError, match="gateway down"):
        await aresponse(_StubResponsesClient(RuntimeError("gateway down")), model="m")


# ---- client-ownership guard ----
_MIGRATED_CALL_SITES = (
    "orchestrator/roles/codex.py",
    "orchestrator/roles/critic_agent.py",
    "orchestrator/scoring/proposal_scorer.py",
)

# ``a.b.create`` attribute chains that only llm_config may call directly.
_SANCTIONED_ONLY_ENDPOINTS = ("chat.completions.create", "responses.create")


def _call_site_trees() -> list[tuple[str, ast.Module]]:
    root = Path(hyperloom.__file__).resolve().parent
    return [(rel, ast.parse((root / rel).read_text(encoding="utf-8"))) for rel in _MIGRATED_CALL_SITES]


def _dotted_attribute_chain(node: ast.AST) -> str:
    """Render the trailing attribute chain of a call target, e.g. ``responses.create``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join(reversed(parts))


def test_migrated_call_sites_do_not_import_the_openai_sdk():
    """Client ownership is llm_config's; these call sites ask it for a client."""
    offenders: list[str] = []
    for rel, tree in _call_site_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [f"{rel}:{node.lineno}" for a in node.names if a.name.split(".")[0] == "openai"]
            elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "openai":
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, f"OpenAI clients must come from llm_config.get_*_openai_client(): {offenders}"


def test_migrated_call_sites_make_no_bare_llm_api_calls():
    """No call site may reach an LLM endpoint itself; llm_config owns every call."""
    offenders: list[str] = []
    for rel, tree in _call_site_trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            chain = _dotted_attribute_chain(node.func)
            if any(chain.endswith(endpoint) for endpoint in _SANCTIONED_ONLY_ENDPOINTS):
                offenders.append(f"{rel}:{node.lineno} calls .{chain}(")
    assert not offenders, f"LLM calls must go through llm_config's entry points: {offenders}"


# ---- anthropic_completion / aanthropic_completion: transport selection ----
_ANTHROPIC_KEY = "_".join(("ANTHROPIC", "API", "KEY"))
_ANTHROPIC_TOKEN = "_".join(("ANTHROPIC", "AUTH", "TOKEN"))
_OAUTH_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
_OAUTH_VALUE = "sk-ant-oat01-fake"


class _ClosingAnthropicTransport(_FakeAnthropicTransport):
    """Sync transport that records whether the caller closed it."""

    def __init__(self, response: Any = None, *, error: BaseException | None = None) -> None:
        super().__init__(response, error=error)
        self.closed = False

    def __enter__(self) -> "_ClosingAnthropicTransport":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        self.closed = True
        return False


class _ClosingAsyncAnthropicTransport(_ClosingAnthropicTransport):
    async def post(self, path: str, *, json: Any) -> Any:  # type: ignore[override]
        return _FakeAnthropicTransport.post(self, path, json=json)

    async def __aenter__(self) -> "_ClosingAsyncAnthropicTransport":
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        self.closed = True
        return False


class _FakeOneShotClient:
    """Stands in for the Claude CLI client; records the completion arguments."""

    instances: list["_FakeOneShotClient"] = []

    def __init__(
        self,
        timeout_s: float = 60.0,
        env: Any = None,
        component: str = "",
        operation: str = "",
    ) -> None:
        self.timeout_s = timeout_s
        self.env = env
        self.component = component
        self.operation = operation
        self.calls: list[dict[str, Any]] = []
        _FakeOneShotClient.instances.append(self)

    def messages(self, **kwargs: Any) -> llm_config.AnthropicMessageResult:
        self.calls.append(kwargs)
        return llm_config.AnthropicMessageResult(
            text="cli reply",
            stop_reason="end_turn",
            usage={"input_tokens": 3, "output_tokens": 2},
        )

    async def amessages(self, **kwargs: Any) -> llm_config.AnthropicMessageResult:
        return self.messages(**kwargs)


@pytest.fixture
def fake_one_shot(monkeypatch: pytest.MonkeyPatch) -> type[_FakeOneShotClient]:
    """Replace the Claude CLI client where ``_one_shot_client`` imports it."""
    from hyperloom.common import claude_oneshot

    _FakeOneShotClient.instances = []
    monkeypatch.setattr(claude_oneshot, "ClaudeOneShotClient", _FakeOneShotClient)
    return _FakeOneShotClient


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({_ANTHROPIC_KEY: "sk-ant-key"}, llm_config.ANTHROPIC_TRANSPORT_HTTP),
        ({_ANTHROPIC_TOKEN: "gateway-bearer"}, llm_config.ANTHROPIC_TRANSPORT_HTTP),
        ({_OAUTH_ENV: _OAUTH_VALUE}, llm_config.ANTHROPIC_TRANSPORT_SDK),
        # The CLI prefers a key over the subscription token, so the transport
        # that key authenticates must win here too.
        ({_ANTHROPIC_KEY: "sk-ant-key", _OAUTH_ENV: _OAUTH_VALUE}, llm_config.ANTHROPIC_TRANSPORT_HTTP),
        ({}, ""),
        ({_OAUTH_ENV: "   "}, ""),
    ],
)
def test_anthropic_transport_follows_the_credential_that_can_authenticate(env, expected):
    assert llm_config.anthropic_transport(env) == expected


def test_anthropic_transport_ready_is_false_without_a_credential():
    assert llm_config.anthropic_transport_ready({}) is False


def test_anthropic_transport_ready_skips_the_sdk_probe_on_the_http_transport(monkeypatch):
    """An API key needs no Claude CLI, so an absent SDK must not disable it."""
    from hyperloom.common import claude_oneshot

    def _missing() -> None:
        raise RuntimeError("claude_agent_sdk not installed")

    monkeypatch.setattr(claude_oneshot, "ensure_available", _missing)
    assert llm_config.anthropic_transport_ready({_ANTHROPIC_KEY: "sk-ant-key"}) is True


def test_ensure_available_rejects_a_missing_cli_binary(monkeypatch):
    """The SDK only spawns `claude`; an importable package with no reachable
    binary still cannot serve a call, and failing here beats failing at the
    first review."""
    from hyperloom.common import claude_oneshot

    monkeypatch.setattr(claude_oneshot, "_load_sdk", lambda: types.SimpleNamespace(__file__=None))
    monkeypatch.setattr(claude_oneshot.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="claude CLI is not available"):
        claude_oneshot.ensure_available()


def test_ensure_available_accepts_a_cli_on_path(monkeypatch):
    from hyperloom.common import claude_oneshot

    monkeypatch.setattr(claude_oneshot, "_load_sdk", lambda: types.SimpleNamespace(__file__=None))
    monkeypatch.setattr(claude_oneshot.shutil, "which", lambda _name: "/usr/bin/claude")

    claude_oneshot.ensure_available()


@pytest.mark.parametrize("binary_name", ["claude", "claude.exe"])
def test_ensure_available_accepts_either_bundled_binary_name(monkeypatch, tmp_path, binary_name):
    """The SDK ships the binary under a platform-dependent name. Matching one
    spelling exactly would fail a host whose SDK is perfectly usable -- and
    this probe is what refuses to build the critic at startup."""
    from hyperloom.common import claude_oneshot

    bundled = tmp_path / "_bundled"
    bundled.mkdir()
    (bundled / ".gitignore").write_text("*\n", encoding="utf-8")
    (bundled / binary_name).write_text("#!/bin/sh\n", encoding="utf-8")
    fake_sdk = types.SimpleNamespace(__file__=str(tmp_path / "__init__.py"))
    monkeypatch.setattr(claude_oneshot, "_load_sdk", lambda: fake_sdk)
    monkeypatch.setattr(claude_oneshot.shutil, "which", lambda _name: None)

    claude_oneshot.ensure_available()
    assert claude_oneshot._locate_cli(fake_sdk) == str(bundled / binary_name)


def test_ensure_available_ignores_non_binary_bundled_entries(monkeypatch, tmp_path):
    """A bundle carrying only its .gitignore is not a usable transport."""
    from hyperloom.common import claude_oneshot

    bundled = tmp_path / "_bundled"
    bundled.mkdir()
    (bundled / ".gitignore").write_text("*\n", encoding="utf-8")
    monkeypatch.setattr(
        claude_oneshot, "_load_sdk", lambda: types.SimpleNamespace(__file__=str(tmp_path / "__init__.py"))
    )
    monkeypatch.setattr(claude_oneshot.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="claude CLI is not available"):
        claude_oneshot.ensure_available()


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [(None, True), (RuntimeError("claude_agent_sdk not installed"), False)],
)
def test_anthropic_transport_ready_probes_the_sdk_for_a_subscription_token(monkeypatch, sdk_error, expected):
    from hyperloom.common import claude_oneshot

    def _probe() -> None:
        if sdk_error is not None:
            raise sdk_error

    monkeypatch.setattr(claude_oneshot, "ensure_available", _probe)
    assert llm_config.anthropic_transport_ready({_OAUTH_ENV: _OAUTH_VALUE}) is expected


def test_anthropic_completion_posts_to_the_messages_api_when_a_key_is_configured(monkeypatch):
    client = _ClosingAnthropicTransport(_FakeAnthropicResponse(body=_MESSAGE_BODY))
    monkeypatch.setattr(llm_config, "get_anthropic_client", lambda **_kw: client)

    result = llm_config.anthropic_completion(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
        system="sys",
        env={_ANTHROPIC_KEY: "sk-ant-key"},
    )

    assert result.text == "hello world"
    assert client.calls[0]["path"] == "/v1/messages"
    assert client.calls[0]["json"] == {
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
        "system": "sys",
    }
    assert client.closed, "the HTTP client must be closed once the completion returns"


def test_anthropic_completion_sends_temperature_on_the_http_path(monkeypatch):
    """The Messages API accepts it, and callers that ask for a low temperature
    to pin an output shape must keep getting one."""
    client = _ClosingAnthropicTransport(_FakeAnthropicResponse(body=_MESSAGE_BODY))
    monkeypatch.setattr(llm_config, "get_anthropic_client", lambda **_kw: client)

    llm_config.anthropic_completion(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        temperature=0.2,
        env={_ANTHROPIC_KEY: "sk-ant-key"},
    )

    assert client.calls[0]["json"]["temperature"] == 0.2


def test_anthropic_completion_omits_temperature_when_unset(monkeypatch):
    """Absent means absent: sending an explicit default would change the
    sampling behaviour of every caller that never asked for one."""
    client = _ClosingAnthropicTransport(_FakeAnthropicResponse(body=_MESSAGE_BODY))
    monkeypatch.setattr(llm_config, "get_anthropic_client", lambda **_kw: client)

    llm_config.anthropic_completion(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        env={_ANTHROPIC_KEY: "sk-ant-key"},
    )

    assert "temperature" not in client.calls[0]["json"]


def test_anthropic_completion_drops_temperature_on_the_cli_path(fake_one_shot):
    """The CLI has no temperature knob, so the argument is accepted and ignored
    rather than raising -- one entry point, two transports, same signature."""
    llm_config.anthropic_completion(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        temperature=0.2,
        env={_OAUTH_ENV: _OAUTH_VALUE},
    )

    assert "temperature" not in fake_one_shot.instances[0].calls[0]


def test_anthropic_completion_omits_an_absent_system_prompt(monkeypatch):
    client = _ClosingAnthropicTransport(_FakeAnthropicResponse(body=_MESSAGE_BODY))
    monkeypatch.setattr(llm_config, "get_anthropic_client", lambda **_kw: client)

    llm_config.anthropic_completion(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        env={_ANTHROPIC_KEY: "sk-ant-key"},
    )

    assert "system" not in client.calls[0]["json"]


def test_anthropic_completion_drives_the_claude_cli_for_a_subscription_token(monkeypatch, fake_one_shot):
    def _refuse(**_kw: Any) -> Any:
        raise AssertionError("a subscription token must not reach the Messages API")

    monkeypatch.setattr(llm_config, "get_anthropic_client", _refuse)

    result = llm_config.anthropic_completion(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
        system="sys",
        env={_OAUTH_ENV: _OAUTH_VALUE},
        timeout_s=12.0,
    )

    assert result.text == "cli reply"
    assert result.stop_reason == "end_turn"
    client = fake_one_shot.instances[0]
    assert client.timeout_s == 12.0
    assert client.calls[0] == {
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": "hi"}],
        "system": "sys",
        "max_tokens": 64,
    }


def test_anthropic_completion_hands_the_cli_the_caller_env(fake_one_shot):
    """The CLI resolves its own credential from the environment it is given, so
    an explicit mapping must reach it instead of being dropped for the ambient
    one — otherwise a caller that resolved credentials from provider-specific
    variables silently authenticates as something else."""
    caller_env = {_OAUTH_ENV: _OAUTH_VALUE, "ANTHROPIC_BASE_URL": "https://gw.example"}

    llm_config.anthropic_completion(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        env=caller_env,
    )

    assert fake_one_shot.instances[0].env == caller_env


@pytest.mark.asyncio
async def test_aanthropic_completion_hands_the_cli_the_caller_env(fake_one_shot):
    caller_env = {_OAUTH_ENV: _OAUTH_VALUE}

    await llm_config.aanthropic_completion(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        env=caller_env,
    )

    assert fake_one_shot.instances[0].env == caller_env


def test_anthropic_completion_keeps_the_cli_default_budget_when_unset(fake_one_shot):
    llm_config.anthropic_completion(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        env={_OAUTH_ENV: _OAUTH_VALUE},
    )
    assert fake_one_shot.instances[0].timeout_s == 60.0


def test_anthropic_completion_raises_when_no_credential_is_configured():
    with pytest.raises(LLMConfigError) as excinfo:
        llm_config.anthropic_completion(
            model="claude-opus-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            env={},
        )
    message = str(excinfo.value)
    for name in ANTHROPIC_CREDENTIAL_ENV_ORDER:
        assert name in message, f"the error must name every credential form: {name}"


@pytest.mark.asyncio
async def test_aanthropic_completion_posts_to_the_messages_api_and_closes(monkeypatch):
    client = _ClosingAsyncAnthropicTransport(_FakeAnthropicResponse(body=_MESSAGE_BODY))
    monkeypatch.setattr(llm_config, "get_async_anthropic_client", lambda **_kw: client)

    result = await llm_config.aanthropic_completion(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
        env={_ANTHROPIC_KEY: "sk-ant-key"},
    )

    assert result.text == "hello world"
    assert client.calls[0]["path"] == "/v1/messages"
    assert client.closed


@pytest.mark.asyncio
async def test_aanthropic_completion_drives_the_claude_cli_for_a_subscription_token(monkeypatch, fake_one_shot):
    def _refuse(**_kw: Any) -> Any:
        raise AssertionError("a subscription token must not reach the Messages API")

    monkeypatch.setattr(llm_config, "get_async_anthropic_client", _refuse)

    result = await llm_config.aanthropic_completion(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
        env={_OAUTH_ENV: _OAUTH_VALUE},
    )

    assert result.text == "cli reply"
    assert fake_one_shot.instances[0].calls[0]["max_tokens"] == 64


@pytest.mark.asyncio
async def test_aanthropic_completion_raises_when_no_credential_is_configured():
    with pytest.raises(LLMConfigError):
        await llm_config.aanthropic_completion(
            model="claude-opus-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            env={},
        )


def test_anthropic_completion_reads_the_ambient_environment_by_default(monkeypatch, fake_one_shot):
    for name in ANTHROPIC_CREDENTIAL_ENV_ORDER:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(_OAUTH_ENV, _OAUTH_VALUE)

    result = llm_config.anthropic_completion(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
    )

    assert result.text == "cli reply"


def test_claude_sdk_env_options_disables_advisor_tool_by_default():
    """Claude Code's advisor-tool beta header (rejected by strict gateways) is
    disabled by default; an operator preset is preserved."""
    opts = claude_sdk_env_options(env={"_".join(("ANTHROPIC", "API", "KEY")): "ak-anthropic"})
    assert opts["env"]["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] == "1"

    preset = claude_sdk_env_options(
        env={
            "_".join(("ANTHROPIC", "API", "KEY")): "ak-anthropic",
            "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "0",
        }
    )
    assert preset["env"]["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] == "0"
