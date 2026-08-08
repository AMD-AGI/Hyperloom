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
from hyperloom.common.llm_config import (
    LLMConfigError,
    achat_completion,
    apply_reasoning_effort,
    build_http_timeout,
    claude_sdk_env_options,
    derive_openai_base_url,
    get_async_openai_client,
    get_openai_client,
    openai_client_kwargs,
    parse_custom_headers,
)

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


def test_openai_kwargs_ignores_anthropic_custom_headers_and_host():
    """The OpenAI/Codex client reads only the OpenAI side; Anthropic headers and
    host are ignored."""
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


def test_openai_kwargs_refuse_anthropic_only_env():
    """Anthropic-only credentials cannot auth an OpenAI-protocol client."""
    with pytest.raises(LLMConfigError):
        openai_client_kwargs(
            env={
                "_".join(("ANTHROPIC", "API", "KEY")): "ak-anthropic",
                "ANTHROPIC_BASE_URL": "https://llm.example.invalid/anthropic",
            }
        )


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
        model="deepseek-v4-pro",
        env={"_".join(("DEEPSEEK", "API", "KEY")): "deepseek-token"},
    )
    assert opts["setting_sources"] == []
    child_env = opts["env"]
    assert child_env["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert child_env["_".join(("ANTHROPIC", "API", "KEY"))] == "deepseek-token"
    assert child_env["_".join(("ANTHROPIC", "AUTH", "TOKEN"))] == "deepseek-token"
    assert child_env["ANTHROPIC_MODEL"] == "deepseek-v4-pro"


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


def test_get_client_requires_an_openai_side_key(monkeypatch):
    _install_fake_openai_sdk(monkeypatch)
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        get_async_openai_client(env={"_".join(("ANTHROPIC", "API", "KEY")): "ak-anthropic"})


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


# ---- client-ownership guard ----
_MIGRATED_CALL_SITES = (
    "orchestrator/roles/codex.py",
    "orchestrator/roles/critic_agent.py",
    "orchestrator/scoring/proposal_scorer.py",
)


def test_migrated_call_sites_do_not_import_the_openai_sdk():
    """Client ownership is llm_config's; these call sites ask it for a client."""
    root = Path(hyperloom.__file__).resolve().parent
    offenders: list[str] = []
    for rel in _MIGRATED_CALL_SITES:
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [f"{rel}:{node.lineno}" for a in node.names if a.name.split(".")[0] == "openai"]
            elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "openai":
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, f"OpenAI clients must come from llm_config.get_*_openai_client(): {offenders}"


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
