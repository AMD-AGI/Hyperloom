# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""RED contracts for native Codex OAuth and exact fallback disablement."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import kernelforge.agent_backends.registry as registry
from kernelforge.agent_backends.base import (
    AgentProviderUnavailableError,
    AgentRunSpec,
    AgentRuntimeConfig,
)
from kernelforge.agent_backends.codex import (
    CodexBackend,
    CodexUnavailableError,
)
from kernelforge.agent_backends.registry import (
    create_registered_backend,
    get_agent_provider,
    resolve_agent_runtime,
)
from kernelforge.cli import _agent_runtime_overrides
from kernelforge.config import Config


class _Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Client:
    created = []

    def __init__(self, config):
        self.config = config
        self.created.append(config)

    def close(self):
        return None


class _NativeSdk:
    CodexConfig = _Config
    Codex = _Client
    ApprovalMode = SimpleNamespace(deny_all="deny_all")
    Sandbox = SimpleNamespace(
        full_access="full_access",
        read_only="read_only",
        workspace_write="workspace_write",
    )


def native_runtime(home: Path, *, fallback_model: str = "") -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        provider="codex",
        model="gpt-5.6-sol-900k",
        fallback_model=fallback_model,
        sandbox_mode="read-only",
        options={
            "auth_mode": "native_oauth",
            "home": str(home),
        },
    )


def test_runtime_can_disable_registered_model_fallback_without_changing_defaults():
    claude = get_agent_provider("claude")
    codex = get_agent_provider("codex")
    claude_before = (
        claude.default_model,
        claude.fallback_model,
        claude.capabilities,
    )

    strict = resolve_agent_runtime(
        "codex",
        model="gpt-5.6-sol-900k",
        fallback_model="none",
        fallback_provider="none",
    )

    assert strict.fallback_model == ""
    assert strict.fallback_provider == ""
    assert resolve_agent_runtime("codex").fallback_model == "gpt-5.5"
    assert get_agent_provider("codex") is codex
    assert (
        claude.default_model,
        claude.fallback_model,
        claude.capabilities,
    ) == claude_before


def test_no_fallback_runtime_never_tries_registered_model_or_provider(monkeypatch):
    attempted = []

    def unavailable(registration, runtime, **kwargs):
        del registration, kwargs
        attempted.append((runtime.provider, runtime.model))
        raise AgentProviderUnavailableError("selected provider/model unavailable")

    monkeypatch.setattr(registry, "_prepare_backend", unavailable)
    runtime = resolve_agent_runtime(
        "codex",
        model="gpt-5.6-sol-900k",
        fallback_model="none",
        fallback_provider="none",
    )

    with pytest.raises(AgentProviderUnavailableError):
        create_registered_backend(runtime)

    assert attempted == [("codex", "gpt-5.6-sol-900k")]


def test_config_and_cli_expose_model_fallback_none():
    config = Config(
        agent_backend="codex",
        agent_model="gpt-5.6-sol-900k",
        agent_fallback_provider="none",
        agent_fallback_model="none",
        agent_precheck=False,
    )
    runtime = config.agent_runtime()
    assert config.agent_fallback_provider == ""
    assert config.agent_fallback_model == ""
    assert runtime.fallback_provider == ""
    assert runtime.fallback_model == ""

    overrides = _agent_runtime_overrides(
        model="gpt-5.6-sol-900k",
        agent_backend="codex",
        agent_cli=None,
        agent_timeout_sec=None,
        agent_reasoning_effort=None,
        agent_sandbox_mode=None,
        agent_fallback_provider="",
        agent_fallback_model="",
        agent_precheck=None,
        agent_options_json=None,
    )
    assert overrides == {
        "agent_model": "gpt-5.6-sol-900k",
        "agent_backend": "codex",
        "agent_fallback_provider": "",
        "agent_fallback_model": "",
    }


def test_native_oauth_uses_dedicated_home_and_strips_gateway_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    home = tmp_path / "codex-oauth"
    home.mkdir()
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong-gateway.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-native-codex")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "X-Secret: must-not-reach-native-codex")
    backend = CodexBackend(runtime=native_runtime(home))

    child = backend._child_environment()
    overrides = backend._config_overrides(None)

    assert child["CODEX_HOME"] == str(home.resolve())
    assert "OPENAI_BASE_URL" not in child
    assert "OPENAI_API_KEY" not in child
    assert "OPENAI_CUSTOM_HEADERS" not in child
    assert overrides == ("features.memories=false",)
    assert all("model_provider" not in item for item in overrides)
    assert all("base_url" not in item for item in overrides)
    assert all("env_key" not in item for item in overrides)


def test_native_oauth_does_not_force_forge_model_provider(tmp_path: Path):
    home = tmp_path / "codex-oauth"
    home.mkdir()
    backend = CodexBackend(runtime=native_runtime(home))
    spec = AgentRunSpec(
        system_prompt="Source only.",
        user_prompt="Return one candidate.",
        cwd=str(tmp_path),
        model="gpt-5.6-sol-900k",
        writable=False,
    )

    options = backend._thread_start_options(_NativeSdk, spec)

    assert options["model"] == "gpt-5.6-sol-900k"
    assert options["sandbox"] == "read_only"
    assert "model_provider" not in options


def test_gateway_mode_remains_unchanged(tmp_path: Path):
    runtime = AgentRuntimeConfig(
        provider="codex",
        model="gpt-gateway",
        options={"auth_mode": "gateway"},
    )
    backend = CodexBackend(
        runtime=runtime,
        gateway={
            "base_url": "https://gateway.example.invalid/v1",
            "key_env": "FAKE_CODEX_API_KEY",
            "headers": {"user": "test-user"},
        },
    )
    spec = AgentRunSpec("system", "user", str(tmp_path), model="gpt-gateway")

    overrides = backend._config_overrides(None)
    options = backend._thread_start_options(_NativeSdk, spec)

    assert any(item == 'model_provider="forge"' for item in overrides)
    assert any("base_url=" in item for item in overrides)
    assert any("env_key=" in item for item in overrides)
    assert options["model_provider"] == "forge"


def test_native_oauth_rejects_ambiguous_gateway_and_unsafe_home(tmp_path: Path):
    absolute = tmp_path / "codex-home"
    absolute.mkdir()
    gateway = {
        "base_url": "https://gateway.example.invalid/v1",
        "key_env": "FAKE_CODEX_API_KEY",
    }
    with pytest.raises(CodexUnavailableError, match="native_oauth.*gateway"):
        CodexBackend(runtime=native_runtime(absolute), gateway=gateway)

    relative = AgentRuntimeConfig(
        provider="codex",
        model="gpt-test",
        options={"auth_mode": "native_oauth", "home": "relative-home"},
    )
    with pytest.raises(CodexUnavailableError, match="absolute"):
        CodexBackend(runtime=relative)

    target = tmp_path / "real-home"
    target.mkdir()
    link = tmp_path / "linked-home"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(CodexUnavailableError, match="symlink"):
        CodexBackend(runtime=native_runtime(link))


def test_native_oauth_preflight_initializes_sdk_without_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    home = tmp_path / "codex-oauth"
    home.mkdir()
    _Client.created.clear()
    monkeypatch.setattr("kernelforge.agent_backends.codex._load_codex_sdk", lambda: _NativeSdk)
    monkeypatch.setattr(
        "kernelforge.agent_backends.codex._resolve_gateway",
        lambda: (_ for _ in ()).throw(AssertionError("gateway must not be resolved")),
    )
    backend = CodexBackend(runtime=native_runtime(home))

    backend.preflight()

    assert backend._preflight_done is True
    assert len(_Client.created) == 1
    config = _Client.created[0]
    assert config.env["CODEX_HOME"] == str(home.resolve())
    assert config.config_overrides == ("features.memories=false",)


def test_claude_provider_remains_registered_and_unmodified():
    claude = get_agent_provider("claude")
    assert claude.name == "claude"
    assert claude.default_model == "claude-opus-5"
    assert claude.fallback_model == "claude-opus-4-8"
    assert claude.capabilities.writable is True
    assert claude.capabilities.resumable is True
    assert claude.capabilities.stop_hooks is True
    assert claude.capabilities.native_subagents is True
    assert claude.capabilities.mcp is True
