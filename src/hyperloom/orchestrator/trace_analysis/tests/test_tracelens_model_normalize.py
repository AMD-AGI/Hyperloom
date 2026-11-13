# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TraceLens SDK model id normalization."""

from __future__ import annotations


import httpx
import pytest

from hyperloom.orchestrator.trace_analysis import _tracelens_checkout


_REAL_FETCH = _tracelens_checkout._fetch_gateway_model_ids


@pytest.fixture(scope="module")
def tl_module():
    from hyperloom.orchestrator.trace_analysis import tracelens_analysis

    return tracelens_analysis


@pytest.fixture(autouse=True)
def _clear_gateway_env(monkeypatch):
    for k in (
        "ANTHROPIC_MODEL",
        "CLAUDE_MODEL",
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "LLM_GATEWAY_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_CUSTOM_HEADERS",
    ):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(autouse=True)
def _no_gateway_model_list(monkeypatch):
    """No test reaches a real gateway: the model list is unavailable unless a test supplies one."""
    monkeypatch.setattr(_tracelens_checkout, "_fetch_gateway_model_ids", set)


def _serve_models(monkeypatch, *ids):
    monkeypatch.setattr(_tracelens_checkout, "_fetch_gateway_model_ids", lambda: {i.lower() for i in ids})


def _use_safe_gateway(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gw.example.invalid/api/v1/llm-proxy/v1")


def test_dot_form_opus_normalized_on_safe(tl_module, monkeypatch):
    _use_safe_gateway(monkeypatch)
    _serve_models(monkeypatch, "claude-opus-4-7")
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "claude-opus-4-7"


def test_unmarked_gateway_host_needs_an_explicit_marker(tl_module, monkeypatch):
    """A retired credential name no longer stands in for gateway detection."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://internal.example.invalid/api/v1")
    monkeypatch.setenv("LLM_GATEWAY_KEY", "ak-test")
    _serve_models(monkeypatch, "claude-opus-4-7")
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "Claude-Opus-4.7"

    monkeypatch.setenv("HYPERLOOM_STRICT_GATEWAY_MARKERS", "internal.example.invalid")
    assert tl_module._resolve_tracelens_model() == "claude-opus-4-7"


def test_already_dash_is_unchanged_on_safe(tl_module, monkeypatch):
    _use_safe_gateway(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-7")
    assert tl_module._resolve_tracelens_model() == "claude-opus-4-7"


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "deepseek-v4.1-pro"])
def test_non_claude_model_keeps_its_dots_on_safe(tl_module, monkeypatch, model):
    """The gateway serves non-Claude ids in dot form, so dashing them names a model it does not have."""
    _use_safe_gateway(monkeypatch)
    monkeypatch.setenv("CLAUDE_MODEL", model)
    assert tl_module._resolve_tracelens_model() == model


def test_non_safe_gateway_leaves_dot_form_untouched(tl_module, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "Claude-Opus-4.7"


def test_claude_model_backs_an_unset_anthropic_model(tl_module, monkeypatch):
    _use_safe_gateway(monkeypatch)
    _serve_models(monkeypatch, "claude-opus-4-7", "claude-sonnet-4-5")
    monkeypatch.setenv("CLAUDE_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "claude-opus-4-7"

    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    assert tl_module._resolve_tracelens_model() == "claude-sonnet-4-5"


def test_empty_env_yields_empty(tl_module, monkeypatch):
    _use_safe_gateway(monkeypatch)
    assert tl_module._resolve_tracelens_model() == ""


def test_whitespace_only_yields_empty(tl_module, monkeypatch):
    _use_safe_gateway(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_MODEL", "   ")
    assert tl_module._resolve_tracelens_model() == ""


def test_plain_openai_base_url_not_normalized(tl_module, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "Claude-Opus-4.7"


def test_no_gateway_env_leaves_dot_form_untouched(tl_module, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "Claude-Opus-4.7"


def test_self_hosted_litellm_base_url_is_treated_as_gateway(tl_module, monkeypatch):
    # Any 'litellm' host is treated as the strict gateway and normalized.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://litellm.internal.example.com/v1")
    _serve_models(monkeypatch, "claude-opus-4-7")
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "claude-opus-4-7"


def test_id_the_gateway_lists_is_sent_as_is(tl_module, monkeypatch):
    """The project1 gateway lists Opus 5.5 in dot form and rejects ``claude-opus-5-5``."""
    _use_safe_gateway(monkeypatch)
    _serve_models(monkeypatch, "claude-opus-5.5", "claude-opus-4-7")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-5.5")
    assert tl_module._resolve_tracelens_model() == "claude-opus-5.5"


def test_dash_id_the_gateway_lists_is_sent_as_is(tl_module, monkeypatch):
    _use_safe_gateway(monkeypatch)
    _serve_models(monkeypatch, "claude-opus-5.5", "claude-opus-4-7")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-7")
    assert tl_module._resolve_tracelens_model() == "claude-opus-4-7"


def test_listed_id_is_matched_without_regard_to_case(tl_module, monkeypatch):
    _use_safe_gateway(monkeypatch)
    _serve_models(monkeypatch, "claude-opus-5.5")
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-5.5")
    assert tl_module._resolve_tracelens_model() == "claude-opus-5.5"


def test_unlisted_dot_id_is_swapped_to_the_dash_id_the_gateway_lists(tl_module, monkeypatch):
    _use_safe_gateway(monkeypatch)
    _serve_models(monkeypatch, "claude-opus-4-7")
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "claude-opus-4-7"


def test_unlisted_dash_id_is_swapped_to_the_dot_id_the_gateway_lists(tl_module, monkeypatch):
    _use_safe_gateway(monkeypatch)
    _serve_models(monkeypatch, "claude-opus-5.5")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-5-5")
    assert tl_module._resolve_tracelens_model() == "claude-opus-5.5"


def test_id_in_neither_form_is_sent_unchanged(tl_module, monkeypatch):
    _use_safe_gateway(monkeypatch)
    _serve_models(monkeypatch, "claude-opus-5.5")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-9.9")
    assert tl_module._resolve_tracelens_model() == "claude-haiku-9.9"


def test_gateway_that_lists_no_models_gets_the_id_unchanged(tl_module, monkeypatch):
    _use_safe_gateway(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-5.5")
    assert tl_module._resolve_tracelens_model() == "claude-opus-5.5"


def _gateway_answers(monkeypatch, handler):
    """Route the gateway client's HTTP calls to ``handler`` instead of the network."""
    real_client = httpx.Client

    class _Client(real_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", _Client)


def test_model_list_request_carries_the_gateway_headers(monkeypatch):
    """AMD's gateway rejects a call without its subscription header, which only ``ANTHROPIC_CUSTOM_HEADERS`` holds."""
    monkeypatch.setattr(_tracelens_checkout, "_fetch_gateway_model_ids", _REAL_FETCH)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gw.example.invalid/api/v1/llm-proxy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: sub-test")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "Claude-Opus-5.5"}, {"id": "claude-opus-4-7"}]})

    _gateway_answers(monkeypatch, handler)

    assert _tracelens_checkout._fetch_gateway_model_ids() == {"claude-opus-5.5", "claude-opus-4-7"}
    (request,) = seen
    assert request.url.path == "/api/v1/llm-proxy/v1/models"
    assert request.headers["Ocp-Apim-Subscription-Key"] == "sub-test"
    assert request.headers["x-api-key"] == "sk-test"


def test_model_list_url_does_not_double_a_version_segment(monkeypatch):
    monkeypatch.setattr(_tracelens_checkout, "_fetch_gateway_model_ids", _REAL_FETCH)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gw.example.invalid/api/v1/llm-proxy/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"data": []})

    _gateway_answers(monkeypatch, handler)
    _tracelens_checkout._fetch_gateway_model_ids()
    assert paths == ["/api/v1/llm-proxy/v1/models"]


def test_a_gateway_that_refuses_the_model_list_is_logged_and_gives_no_ids(monkeypatch, caplog):
    monkeypatch.setattr(_tracelens_checkout, "_fetch_gateway_model_ids", _REAL_FETCH)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gw.example.invalid/api/v1/llm-proxy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _gateway_answers(monkeypatch, lambda request: httpx.Response(401, json={"error": "no subscription key"}))

    with caplog.at_level("WARNING"):
        assert _tracelens_checkout._fetch_gateway_model_ids() == set()
    assert "model list" in caplog.text
    assert "401" in caplog.text


def test_a_missing_credential_is_logged_and_gives_no_ids(monkeypatch, caplog):
    monkeypatch.setattr(_tracelens_checkout, "_fetch_gateway_model_ids", _REAL_FETCH)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gw.example.invalid/api/v1/llm-proxy")

    with caplog.at_level("WARNING"):
        assert _tracelens_checkout._fetch_gateway_model_ids() == set()
    assert "model list" in caplog.text
