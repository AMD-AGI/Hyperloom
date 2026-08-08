# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the report-narrative LLM client adapters.

The adapters call through ``hyperloom.common.llm_config``. Those entry points
are patched with ``raising=False`` so this suite is independent of whether the
provider contract has landed in the module yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from hyperloom.common.llm_config import LLMConfigError
from hyperloom.inference_optimizer.breakdown.reporters.llm_client import (
    AnthropicHttpClient,
    NullClient,
    OpenAIHttpClient,
    build_client_from_env,
)

_LLM_CONFIG = "hyperloom.common.llm_config"

_REPORT_ENV_VARS = (
    "HYPERLOOM_REPORT_LLM_BACKEND",
    "HYPERLOOM_REPORT_MODEL",
    "HYPERLOOM_REPORT_MAX_TOKENS",
)


@dataclass
class _MessageReply:
    """Stand-in for ``llm_config.AnthropicMessageResult``."""

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
) -> dict[str, list[dict[str, Any]]]:
    """Patch both sync client factories; return per-provider construction kwargs."""
    built: dict[str, list[dict[str, Any]]] = {"openai": [], "anthropic": []}

    def _openai(**kwargs: Any) -> str:
        built["openai"].append(kwargs)
        if openai_error is not None:
            raise openai_error
        return "openai-client"

    def _anthropic(**kwargs: Any) -> str:
        built["anthropic"].append(kwargs)
        if anthropic_error is not None:
            raise anthropic_error
        return "anthropic-client"

    monkeypatch.setattr(f"{_LLM_CONFIG}.get_openai_client", _openai, raising=False)
    monkeypatch.setattr(f"{_LLM_CONFIG}.get_anthropic_client", _anthropic, raising=False)
    monkeypatch.setattr(f"{_LLM_CONFIG}.build_http_timeout", lambda **kwargs: kwargs, raising=False)
    return built


def test_null_client_disables_the_narrative_pass() -> None:
    assert NullClient().complete(system="s", user="u") == ""


def test_openai_client_delegates_to_the_shared_chat_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _fake(client: Any, **params: Any) -> tuple[str, object | None]:
        calls.append({"client": client, **params})
        return "narrative", None

    monkeypatch.setattr(f"{_LLM_CONFIG}.stream_chat_completion_text", _fake, raising=False)

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


def test_anthropic_client_delegates_to_the_shared_messages_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _fake(client: Any, **params: Any) -> _MessageReply:
        calls.append({"client": client, **params})
        return _MessageReply("narrative")

    monkeypatch.setattr(f"{_LLM_CONFIG}.anthropic_messages", _fake, raising=False)

    client = AnthropicHttpClient(client="anthropic-client", model="m", max_output_tokens=8)
    assert client.complete(system="sys", user="user") == "narrative"
    assert calls[0]["client"] == "anthropic-client"
    assert calls[0]["model"] == "m"
    assert calls[0]["system"] == "sys"
    assert calls[0]["messages"] == [{"role": "user", "content": "user"}]
    assert calls[0]["max_tokens"] == 8


def test_provider_transport_errors_propagate_to_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(client: Any, **params: Any) -> tuple[str, object | None]:
        raise RuntimeError("gateway 502")

    monkeypatch.setattr(f"{_LLM_CONFIG}.stream_chat_completion_text", _boom, raising=False)

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

    assert isinstance(client, AnthropicHttpClient)
    assert client.client == "anthropic-client"
    assert client.model == "claude-opus-5"
    assert client.max_output_tokens == 1024
    assert built["openai"] == []


def test_build_client_falls_back_to_deterministic_when_credentials_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_factories(monkeypatch, anthropic_error=LLMConfigError("ANTHROPIC_API_KEY not set in env"))
    monkeypatch.setenv("HYPERLOOM_REPORT_LLM_BACKEND", "anthropic")

    assert build_client_from_env() is None


def test_build_client_ignores_an_unparseable_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_factories(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_REPORT_LLM_BACKEND", "openai")
    monkeypatch.setenv("HYPERLOOM_REPORT_MAX_TOKENS", "not-a-number")

    client = build_client_from_env()

    assert isinstance(client, OpenAIHttpClient)
    assert client.max_output_tokens == 1024
