# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the report-narrative LLM client adapters.

The adapters call through ``hyperloom.common.llm_config``, so these tests patch
its entry points by their real names: a rename on the contract side has to fail
here rather than silently install an unused stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.reporters.llm_client import (
    REPORT_HTTP_TIMEOUT_SEC,
    AnthropicClient,
    NullClient,
    OpenAIHttpClient,
    build_client_from_env,
)

_LLM_CONFIG = "hyperloom.common.llm_config"
_CLAUDE_ONESHOT = "hyperloom.common.claude_oneshot"

_REPORT_ENV_VARS = (
    "HYPERLOOM_REPORT_LLM_BACKEND",
    "HYPERLOOM_REPORT_MODEL",
    "HYPERLOOM_REPORT_MAX_TOKENS",
)


@dataclass
class _MessageReply:
    """Stand-in for ``llm_config.AnthropicMessageResult``."""

    text: str


@dataclass
class _CompletionReply:
    """Stand-in for ``llm_config.ChatCompletionResult``."""

    text: str


@pytest.fixture(autouse=True)
def _clean_report_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep an operator's own report-backend settings out of these tests."""
    for name in _REPORT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _install_factories(
    monkeypatch: pytest.MonkeyPatch,
    *,
    openai_error: Exception | None = None,
    anthropic_error: Exception | None = None,
    transport: str = "sdk",
) -> dict[str, list[dict[str, Any]]]:
    """Patch the OpenAI factory and the Anthropic transport probe.

    ``transport`` stands in for the credential shape llm_config would resolve,
    so a test can pick a branch without exporting provider variables.
    """
    built: dict[str, list[dict[str, Any]]] = {"openai": [], "anthropic": []}

    def _openai(**kwargs: Any) -> str:
        built["openai"].append(kwargs)
        if openai_error is not None:
            raise openai_error
        return "openai-client"

    def _ensure_available() -> None:
        built["anthropic"].append({})
        if anthropic_error is not None:
            raise anthropic_error

    monkeypatch.setattr(f"{_LLM_CONFIG}.get_openai_client", _openai)
    monkeypatch.setattr(f"{_LLM_CONFIG}.anthropic_transport", lambda *_a, **_kw: transport)
    monkeypatch.setattr(f"{_CLAUDE_ONESHOT}.ensure_available", _ensure_available)
    monkeypatch.setattr(f"{_LLM_CONFIG}.build_http_timeout", lambda **kwargs: kwargs)
    return built


def test_null_client_disables_the_narrative_pass() -> None:
    assert NullClient().complete(system="s", user="u") == ""


def test_openai_client_delegates_to_the_shared_chat_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _fake(client: Any, **params: Any) -> _CompletionReply:
        calls.append({"client": client, **params})
        return _CompletionReply("narrative")

    monkeypatch.setattr(f"{_LLM_CONFIG}.chat_completion", _fake)

    client = OpenAIHttpClient(client="openai-client", model="m", max_output_tokens=32)
    assert client.complete(system="sys", user="user") == "narrative"
    assert calls[0]["client"] == "openai-client"
    assert calls[0]["model"] == "m"
    assert calls[0]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    assert calls[0]["max_tokens"] == 32
    assert calls[0]["temperature"] == 0.2


def test_anthropic_client_delegates_to_the_single_shot_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _fake(**params: Any) -> _MessageReply:
        calls.append(params)
        return _MessageReply("narrative")

    monkeypatch.setattr(f"{_LLM_CONFIG}.anthropic_completion", _fake)

    client = AnthropicClient(model="m", max_output_tokens=8, timeout_s=12.0)
    assert client.complete(system="sys", user="user") == "narrative"
    assert calls[0]["model"] == "m"
    assert calls[0]["system"] == "sys"
    assert calls[0]["messages"] == [{"role": "user", "content": "user"}]
    assert calls[0]["max_tokens"] == 8
    assert calls[0]["timeout_s"] == 12.0


def test_provider_transport_errors_propagate_to_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(client: Any, **params: Any) -> _CompletionReply:
        raise RuntimeError("gateway 502")

    monkeypatch.setattr(f"{_LLM_CONFIG}.chat_completion", _boom)

    with pytest.raises(RuntimeError, match="gateway 502"):
        OpenAIHttpClient(client="openai-client").complete(system="s", user="u")


@pytest.mark.parametrize("backend", ["", "none", "off", "disabled", "NONE"])
def test_build_client_returns_none_when_backend_is_off(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    _install_factories(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_REPORT_LLM_BACKEND", backend)
    assert build_client_from_env() is None


def test_build_client_returns_none_when_backend_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_factories(monkeypatch)
    assert build_client_from_env() is None


def test_build_client_returns_none_for_an_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_factories(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_REPORT_LLM_BACKEND", "bedrock")
    assert build_client_from_env() is None


def test_build_client_wires_the_openai_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    built = _install_factories(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_REPORT_LLM_BACKEND", "openai")
    monkeypatch.setenv("HYPERLOOM_REPORT_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("HYPERLOOM_REPORT_MAX_TOKENS", "256")

    client = build_client_from_env()

    assert isinstance(client, OpenAIHttpClient)
    assert client.client == "openai-client"
    assert client.model == "gpt-5.6-sol"
    assert client.max_output_tokens == 256
    assert built["anthropic"] == []
    assert built["openai"][0]["timeout"] == {"connect": 60.0, "read": 60.0, "write": 60.0, "pool": 60.0}


def test_build_client_wires_the_anthropic_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    built = _install_factories(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_REPORT_LLM_BACKEND", "anthropic")

    client = build_client_from_env()

    assert isinstance(client, AnthropicClient)
    assert client.timeout_s == REPORT_HTTP_TIMEOUT_SEC
    assert client.model == "claude-opus-5"
    assert client.max_output_tokens == 1024
    assert built["openai"] == []


def test_build_client_skips_the_sdk_probe_on_the_http_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """An API key needs no Claude CLI, so its absence must not disable the pass."""
    built = _install_factories(
        monkeypatch,
        transport="http",
        anthropic_error=RuntimeError("claude_agent_sdk is not installed"),
    )
    monkeypatch.setenv("HYPERLOOM_REPORT_LLM_BACKEND", "anthropic")

    assert isinstance(build_client_from_env(), AnthropicClient)
    assert built["anthropic"] == [], "the SDK probe belongs to the CLI transport only"


def test_build_client_returns_none_without_an_anthropic_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_factories(monkeypatch, transport="")
    monkeypatch.setenv("HYPERLOOM_REPORT_LLM_BACKEND", "anthropic")

    assert build_client_from_env() is None


def test_build_client_falls_back_to_deterministic_when_the_claude_sdk_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_factories(monkeypatch, anthropic_error=RuntimeError("claude_agent_sdk is not installed"))
    monkeypatch.setenv("HYPERLOOM_REPORT_LLM_BACKEND", "anthropic")

    assert build_client_from_env() is None


def test_build_client_ignores_an_unparseable_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_factories(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_REPORT_LLM_BACKEND", "openai")
    monkeypatch.setenv("HYPERLOOM_REPORT_MAX_TOKENS", "not-a-number")

    client = build_client_from_env()

    assert isinstance(client, OpenAIHttpClient)
    assert client.max_output_tokens == 1024
