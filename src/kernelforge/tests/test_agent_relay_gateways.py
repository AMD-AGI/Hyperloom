"""Tests for relay profiles reaching a gateway through the other configured line."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

import kernelforge.agent_backends.claude as claude_module
import kernelforge.agent_backends.registry as registry
from kernelforge.agent_backends.base import AgentRuntimeConfig
from kernelforge.loop.bootstrap.relay import config_for_profile, probe_agent_profile
from kernelforge.config import Config
from kernelforge.agent_backends.gateway import anthropic_environment_from_openai, openai_gateway_from_anthropic
from kernelforge.loop.agent_relay import AgentProfile, AgentRelayConfig

_LINE_ENV = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_CUSTOM_HEADERS",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_CUSTOM_HEADERS",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)

_CLAUDE = AgentProfile("claude", "claude", "claude-opus-5")
_CODEX = AgentProfile("codex", "codex", "gpt-6-astra")


@pytest.fixture
def clean_env(monkeypatch):
    for name in _LINE_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _anthropic_line(env) -> None:
    env.setenv("ANTHROPIC_BASE_URL", "https://gw.example/llm-proxy")
    env.setenv("ANTHROPIC_AUTH_TOKEN", "anthropic-secret")
    env.setenv("ANTHROPIC_CUSTOM_HEADERS", "user: jiaqliu\nAuthorization: Bearer other")


def _openai_line(env) -> None:
    env.setenv("OPENAI_BASE_URL", "https://gw.example/llm-proxy/v1")
    env.setenv("OPENAI_API_KEY", "openai-secret")
    env.setenv("OPENAI_CUSTOM_HEADERS", "user: jiaqliu\nx-api-key: other")


def test_openai_gateway_borrows_the_anthropic_line_without_its_auth_headers(clean_env):
    _anthropic_line(clean_env)

    gateway = openai_gateway_from_anthropic()

    assert gateway.base_url == "https://gw.example/llm-proxy/v1"
    assert gateway.key_env == "ANTHROPIC_AUTH_TOKEN"
    assert gateway.headers == {"user": "jiaqliu"}


def test_anthropic_environment_borrows_the_openai_line(clean_env):
    _openai_line(clean_env)

    assert anthropic_environment_from_openai() == {
        "ANTHROPIC_BASE_URL": "https://gw.example/llm-proxy",
        "ANTHROPIC_AUTH_TOKEN": "openai-secret",
        "ANTHROPIC_CUSTOM_HEADERS": "user: jiaqliu",
    }


@pytest.mark.parametrize(
    ("anthropic_base_url", "openai_base_url"),
    [
        ("https://gw.example/llm-proxy", "https://gw.example/llm-proxy/v1"),
        ("https://gw.example/llm-proxy/", "https://gw.example/llm-proxy/v1"),
        ("https://gw.example/llm-proxy/v1", "https://gw.example/llm-proxy/v1"),
    ],
)
def test_the_borrowed_openai_route_is_one_v1_segment_from_the_anthropic_route(
    clean_env, anthropic_base_url, openai_base_url
):
    _anthropic_line(clean_env)
    clean_env.setenv("ANTHROPIC_BASE_URL", anthropic_base_url)

    assert openai_gateway_from_anthropic().base_url == openai_base_url


def test_a_failed_borrowed_line_names_its_address_and_the_providers_own_variables(clean_env, monkeypatch, tmp_path):
    _anthropic_line(clean_env)
    monkeypatch.setattr(
        registry, "get_agent_provider", lambda name: SimpleNamespace(name=name, availability=lambda: True)
    )

    def create(runtime, **_kwargs):
        if runtime.provider == "codex":
            raise RuntimeError("404 Not Found")
        return SimpleNamespace(runtime=runtime)

    monkeypatch.setattr(registry, "create_registered_backend", create)
    relay = AgentRelayConfig(primary_profile_id="claude", profiles=(_CLAUDE, _CODEX))

    status = probe_agent_profile(
        _CODEX,
        relay_config=relay,
        config=Config(workspace=str(tmp_path), agent_backend="claude"),
        workspace=str(tmp_path),
    )

    assert status.available is False
    assert "at https://gw.example/llm-proxy/v1" in status.reason
    assert "set OPENAI_BASE_URL and OPENAI_API_KEY" in status.reason
    assert "404 Not Found" in status.reason
    assert "anthropic-secret" not in status.reason


def test_no_line_configured_borrows_nothing(clean_env):
    assert not openai_gateway_from_anthropic().is_complete()
    assert anthropic_environment_from_openai() == {}


def test_codex_relay_profile_borrows_the_anthropic_line_only_when_openai_is_unset(clean_env, tmp_path):
    _anthropic_line(clean_env)
    config = Config(workspace=str(tmp_path), agent_backend="claude", agent_options={"claude_only": True})

    borrowed = config_for_profile(config, _CODEX, primary_provider="claude").agent_options

    assert borrowed == {
        "gateway": {
            "base_url": "https://gw.example/llm-proxy/v1",
            "key_env": "ANTHROPIC_AUTH_TOKEN",
            "headers": {"user": "jiaqliu"},
        }
    }
    _openai_line(clean_env)
    assert config_for_profile(config, _CODEX, primary_provider="claude").agent_options == {}


def test_claude_relay_profile_borrows_the_openai_line_only_when_anthropic_is_unset(clean_env, tmp_path):
    _openai_line(clean_env)
    config = Config(workspace=str(tmp_path), agent_backend="codex")

    borrowed = config_for_profile(config, _CLAUDE, primary_provider="codex").agent_options

    assert borrowed["env"]["ANTHROPIC_BASE_URL"] == "https://gw.example/llm-proxy"
    assert borrowed["env"]["ANTHROPIC_AUTH_TOKEN"] == "openai-secret"
    _anthropic_line(clean_env)
    assert config_for_profile(config, _CLAUDE, primary_provider="codex").agent_options == {}


def test_primary_keeps_its_own_options_and_never_borrows(clean_env, tmp_path):
    _openai_line(clean_env)
    config = Config(workspace=str(tmp_path), agent_backend="claude", agent_options={"claude_only": True})

    assert config_for_profile(config, _CLAUDE, primary_provider="claude").agent_options == {"claude_only": True}


def test_a_borrowed_credential_is_not_part_of_persisted_relay_config(clean_env, tmp_path):
    _openai_line(clean_env)
    relay = AgentRelayConfig(primary_profile_id="codex", profiles=(_CODEX, _CLAUDE))
    config_for_profile(Config(workspace=str(tmp_path), agent_backend="codex"), _CLAUDE, primary_provider="codex")

    assert "openai-secret" not in json.dumps(asdict(relay))


def test_claude_probe_reaches_the_endpoint_its_runtime_names(clean_env, monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["env"] = kwargs.get("env")
        return SimpleNamespace(returncode=0, stdout=json.dumps({"result": "OK"}), stderr="")

    monkeypatch.setattr(claude_module.subprocess, "run", fake_run)
    backend = claude_module.ClaudeBackend(
        AgentRuntimeConfig(
            provider="claude",
            model="claude-opus-5",
            options={"env": {"ANTHROPIC_BASE_URL": "https://gw.example/llm-proxy"}},
        )
    )

    backend.probe(cwd=".")

    assert captured["env"]["ANTHROPIC_BASE_URL"] == "https://gw.example/llm-proxy"


def test_claude_probe_inherits_the_process_environment_without_runtime_env(clean_env, monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["env"] = kwargs.get("env", "unset")
        return SimpleNamespace(returncode=0, stdout=json.dumps({"result": "OK"}), stderr="")

    monkeypatch.setattr(claude_module.subprocess, "run", fake_run)
    claude_module.ClaudeBackend(AgentRuntimeConfig(provider="claude", model="claude-opus-5")).probe(cwd=".")

    assert captured["env"] is None
