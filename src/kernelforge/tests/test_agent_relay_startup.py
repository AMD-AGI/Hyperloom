"""Tests for startup-only Agent relay availability checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import kernelforge.agent_backends.registry as registry
from kernelforge.agent_backends.registry import get_agent_provider
from kernelforge.cli.relay import (
    build_search_actor_pool,
    probe_agent_profiles,
    resolve_default_relay_config,
)
from kernelforge.config import Config
from kernelforge.loop.agent_relay import AgentProfile, AgentRelayConfig
from kernelforge.loop.search_actors import SearchActorBundle


def _relay_config(primary: str = "claude") -> AgentRelayConfig:
    return AgentRelayConfig(
        primary_profile_id=primary,
        profiles=(
            AgentProfile("claude", "claude", "claude-opus-5", "max"),
            AgentProfile("codex", "codex", "gpt-6-astra", "max"),
        ),
    )


def _provider(name: str):
    return SimpleNamespace(
        name=name,
        availability=lambda: True,
    )


def test_profiles_read_each_providers_model_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5")
    monkeypatch.setenv("CODEX_MODEL", "gpt-6-astra")

    relay_config = resolve_default_relay_config(
        Config(
            workspace=str(tmp_path),
            agent_backend="claude",
            agent_model="claude-opus-5",
        )
    )

    assert relay_config.primary_profile_id == "claude"
    assert [(profile.profile_id, profile.provider, profile.model) for profile in relay_config.profiles] == [
        ("claude", "claude", "claude-opus-5"),
        ("codex", "codex", "gpt-6-astra"),
    ]
    assert all(profile.reasoning_effort for profile in relay_config.profiles)


def test_primary_keeps_an_explicit_model_over_its_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5")
    monkeypatch.setenv("CODEX_MODEL", "gpt-6-astra")

    relay_config = resolve_default_relay_config(
        Config(
            workspace=str(tmp_path),
            agent_backend="claude",
            agent_model="claude-opus-4-8",
        )
    )

    assert relay_config.primary_profile.model == "claude-opus-4-8"
    assert [profile.provider for profile in relay_config.profiles] == ["claude", "codex"]


def test_unset_model_variables_fall_back_to_registry_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    monkeypatch.delenv("CODEX_MODEL", raising=False)

    relay_config = resolve_default_relay_config(
        Config(
            workspace=str(tmp_path),
            agent_backend="claude",
            agent_model="",
        )
    )

    assert [(profile.profile_id, profile.model) for profile in relay_config.profiles] == [
        ("claude", get_agent_provider("claude").default_model),
        ("codex", get_agent_provider("codex").default_model),
    ]


def test_executable_override_stays_with_the_primary_provider(tmp_path):
    from kernelforge.cli.relay import config_for_profile

    config = Config(workspace=str(tmp_path), agent_backend="claude", agent_cli="/opt/claude")
    codex = AgentProfile("codex", "codex", "gpt-6-astra")
    claude = AgentProfile("claude", "claude", "claude-opus-5")

    assert config_for_profile(config, claude, primary_provider="claude").agent_cli == "/opt/claude"
    assert config_for_profile(config, codex, primary_provider="claude").agent_cli == ""


def test_startup_probe_checks_every_exact_profile(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(registry, "get_agent_provider", _provider)

    def create(runtime, **kwargs):
        seen.append((runtime.provider, runtime.model, runtime.fallback_provider, kwargs))
        return SimpleNamespace(runtime=runtime)

    monkeypatch.setattr(registry, "create_registered_backend", create)
    snapshot = probe_agent_profiles(
        _relay_config(),
        config=Config(
            workspace=str(tmp_path),
            agent_backend="claude",
            agent_model="claude-opus-5",
            agent_reasoning_effort="max",
        ),
        workspace=str(tmp_path),
    )

    assert [(provider, model) for provider, model, _fallback, _kwargs in seen] == [
        ("claude", "claude-opus-5"),
        ("codex", "gpt-6-astra"),
    ]
    assert all(fallback == "" for _provider, _model, fallback, _kwargs in seen)
    assert all(kwargs["probe_cwd"] == str(tmp_path) for *_prefix, kwargs in seen)
    assert snapshot.profiles["claude"].available is True
    assert snapshot.profiles["codex"].available is True


def test_optional_profile_probe_failure_is_saved_and_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "get_agent_provider", _provider)

    def create(runtime, **_kwargs):
        if runtime.provider == "codex":
            raise RuntimeError("model is unavailable")
        return SimpleNamespace(runtime=runtime)

    monkeypatch.setattr(registry, "create_registered_backend", create)
    relay_config = _relay_config()
    snapshot = probe_agent_profiles(
        relay_config,
        config=Config(
            workspace=str(tmp_path),
            agent_backend="claude",
            agent_model="claude-opus-5",
            agent_reasoning_effort="max",
        ),
        workspace=str(tmp_path),
    )
    built = []
    pool = build_search_actor_pool(
        relay_config,
        availability=snapshot,
        bundle_factory=lambda profile: (
            built.append(profile.profile_id) or SearchActorBundle(profile=profile, agent_fn=lambda: None)
        ),
    )

    assert built == ["claude"]
    assert pool.availability.profiles["codex"].available is False
    assert "model is unavailable" in pool.availability.profiles["codex"].reason


def test_primary_profile_probe_failure_refuses_startup(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "get_agent_provider", _provider)
    monkeypatch.setattr(
        registry,
        "create_registered_backend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("primary unavailable")),
    )

    with pytest.raises(ValueError, match="primary Agent profile is unavailable"):
        probe_agent_profiles(
            _relay_config(),
            config=Config(
                workspace=str(tmp_path),
                agent_backend="claude",
                agent_model="claude-opus-5",
                agent_reasoning_effort="max",
            ),
            workspace=str(tmp_path),
        )


def test_switching_the_implementer_provider_drops_the_other_providers_cli_and_options(monkeypatch, tmp_path):
    from kernelforge.orchestrator import agent as agent_module

    captured = {}

    def create(runtime, **_kwargs):
        captured["runtime"] = runtime
        return SimpleNamespace(name=runtime.provider, runtime=runtime)

    monkeypatch.setattr(agent_module, "create_registered_backend", create)
    config = Config(
        workspace=str(tmp_path),
        agent_backend="codex",
        agent_cli="/opt/bin/codex",
        agent_options={"codex_only": True},
    )

    agent_module._resolve_implementer_backend(config, "claude", None)

    assert captured["runtime"].provider == "claude"
    assert captured["runtime"].executable == ""
    assert captured["runtime"].options == {}


def test_relay_summary_reports_whether_a_switch_is_possible():
    from kernelforge.cli.relay import agent_relay_summary, relay_switch_possible
    from kernelforge.loop.agent_relay import AgentAvailabilitySnapshot, ProfileAvailability
    from kernelforge.loop.run_state import RunState

    state = RunState()
    state.agent_relay.active_profile_id = "claude"
    state.agent_availability_snapshot = AgentAvailabilitySnapshot(
        profiles={
            "claude": ProfileAvailability(provider="claude", model="claude-opus-5", available=True),
            "codex": ProfileAvailability(provider="codex", model="gpt-6-astra", available=False, reason="no key"),
        }
    )

    summary = agent_relay_summary(state)

    assert relay_switch_possible(state.agent_availability_snapshot) is False
    assert summary["switch_possible"] is False
    assert summary["active_profile_id"] == "claude"
    assert summary["profiles"]["codex"] == {
        "provider": "codex",
        "model": "gpt-6-astra",
        "available": False,
        "reason": "no key",
    }
