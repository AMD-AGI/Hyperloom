"""Tests for startup-only Agent relay availability checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import kernelforge.agent_backends.registry as registry
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
            AgentProfile("claude", "claude", "claude-opus-5.5", "max"),
            AgentProfile("codex", "codex", "gpt-6", "max"),
        ),
    )


def _provider(name: str):
    return SimpleNamespace(
        name=name,
        availability=lambda: True,
    )


def test_default_profiles_target_opus_5_5_and_gpt_6(tmp_path):
    relay_config = resolve_default_relay_config(
        Config(
            workspace=str(tmp_path),
            agent_backend="claude",
            agent_model="",
        )
    )

    models = {profile.provider: profile.model for profile in relay_config.profiles}
    assert models == {
        "claude": "claude-opus-5.5",
        "codex": "gpt-6",
    }
    assert relay_config.primary_profile_id == "claude_opus_5_5"


def test_custom_primary_still_probes_both_target_profiles(tmp_path):
    relay_config = resolve_default_relay_config(
        Config(
            workspace=str(tmp_path),
            agent_backend="claude",
            agent_model="claude-custom",
        )
    )

    assert relay_config.primary_profile_id == "primary"
    assert {(profile.profile_id, profile.model) for profile in relay_config.profiles} == {
        ("primary", "claude-custom"),
        ("claude_opus_5_5", "claude-opus-5.5"),
        ("codex_gpt_6", "gpt-6"),
    }


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
            agent_model="claude-opus-5.5",
            agent_reasoning_effort="max",
        ),
        workspace=str(tmp_path),
    )

    assert [(provider, model) for provider, model, _fallback, _kwargs in seen] == [
        ("claude", "claude-opus-5.5"),
        ("codex", "gpt-6"),
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
            agent_model="claude-opus-5.5",
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
                agent_model="claude-opus-5.5",
                agent_reasoning_effort="max",
            ),
            workspace=str(tmp_path),
        )
