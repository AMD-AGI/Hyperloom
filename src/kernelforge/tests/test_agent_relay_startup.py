"""Tests for Agent relay resolution, startup availability checks and search actor construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

import kernelforge.agent_backends.registry as registry
from kernelforge.workflows.forge_loop.bootstrap.relay import (
    SearchActorBuildContext,
    build_search_actor_pool,
    probe_agent_profiles,
    resolve_relay_policy,
    resolve_session_relay,
)
from kernelforge.config import Config
from kernelforge.workflows.forge_loop.search.agent_relay import (
    AgentAvailabilitySnapshot,
    AgentProfile,
    AgentProfileSpec,
    AgentRelayConfig,
    AgentRelayPolicy,
    ProfileAvailability,
    StartPreference,
    select_start_profile,
)
from kernelforge.workflows.forge_loop.search.search_actors import SearchActorBundle


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


def _session_relay(config: Config) -> AgentRelayConfig:
    return resolve_session_relay(resolve_relay_policy(config), config)


def test_policy_records_one_member_per_provider_without_models(tmp_path):
    policy = resolve_relay_policy(Config(workspace=str(tmp_path), agent_backend="codex", agent_model="gpt-6-astra"))

    assert policy == AgentRelayPolicy(
        primary_profile_id="codex",
        profiles=(AgentProfileSpec("codex", "codex"), AgentProfileSpec("claude", "claude")),
        blockage_threshold=3,
    )
    assert (
        AgentRelayPolicy.from_dict(
            {
                "primary_profile_id": "codex",
                "profiles": [
                    {"profile_id": "codex", "provider": "codex"},
                    {"profile_id": "claude", "provider": "claude"},
                ],
                "blockage_threshold": 3,
            }
        )
        == policy
    )


def test_a_persisted_profile_model_is_refused():
    with pytest.raises(ValueError, match="unknown fields: model"):
        AgentRelayPolicy.from_dict(
            {
                "primary_profile_id": "claude",
                "profiles": [{"profile_id": "claude", "provider": "claude", "model": "claude-opus-5"}],
                "blockage_threshold": 3,
            }
        )


def test_profiles_read_each_providers_model_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5")
    monkeypatch.setenv("CODEX_MODEL", "gpt-6-astra")

    relay_config = _session_relay(
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

    relay_config = _session_relay(
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

    relay_config = _session_relay(
        Config(
            workspace=str(tmp_path),
            agent_backend="claude",
            agent_model="",
        )
    )

    assert [(profile.profile_id, profile.model) for profile in relay_config.profiles] == [
        ("claude", registry.get_agent_provider("claude").default_model),
        ("codex", registry.get_agent_provider("codex").default_model),
    ]


def test_a_resumed_session_runs_the_model_and_effort_it_was_given(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_MODEL", "gpt-6-astra")
    policy = resolve_relay_policy(Config(workspace=str(tmp_path), agent_backend="claude", agent_model="claude-opus-5"))

    relay_config = resolve_session_relay(
        policy,
        Config(
            workspace=str(tmp_path),
            agent_backend="claude",
            agent_model="claude-opus-4-8",
            agent_reasoning_effort="low",
        ),
    )

    assert [(profile.profile_id, profile.model, profile.reasoning_effort) for profile in relay_config.profiles] == [
        ("claude", "claude-opus-4-8", "low"),
        ("codex", "gpt-6-astra", "low"),
    ]


def test_a_resumed_session_without_a_backend_starts_on_the_campaigns_primary(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_MODEL", "gpt-6-astra")
    policy = resolve_relay_policy(Config(workspace=str(tmp_path), agent_backend="codex", agent_model="gpt-6-astra"))

    relay_config = resolve_session_relay(policy, Config(workspace=str(tmp_path), agent_backend="auto", agent_model=""))

    assert relay_config.primary_profile == AgentProfile(
        "codex", "codex", "gpt-6-astra", relay_config.primary_profile.reasoning_effort
    )


def test_a_resumed_session_cannot_change_the_primary_provider(tmp_path):
    policy = resolve_relay_policy(Config(workspace=str(tmp_path), agent_backend="codex", agent_model="gpt-6-astra"))

    with pytest.raises(ValueError, match="model relay starts on codex; the agent backend cannot change on resume"):
        resolve_session_relay(policy, Config(workspace=str(tmp_path), agent_backend="claude"))


def test_a_campaign_without_relay_has_no_session_relay(tmp_path):
    assert resolve_session_relay(None, Config(workspace=str(tmp_path), agent_backend="claude")) is None


@dataclass(frozen=True)
class _Session:
    agent_backend: str = ""


@dataclass(frozen=True)
class _Setup:
    session: _Session = field(default_factory=_Session)


def _capture_supervisor(monkeypatch) -> dict:
    from kernelforge.workflows.forge_loop.bootstrap import lanes
    from kernelforge.workflows.forge_loop.roles import agent, orchestration, supervisor

    captured: dict = {}

    def make_agent_fn(config, setup, **_kwargs):
        return SimpleNamespace(backend_name=config.agent_backend, backend_model=config.agent_model)

    def make_supervisor_fn(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(agent, "make_agent_fn", make_agent_fn)
    monkeypatch.setattr(orchestration, "make_orchestration_service", lambda **_kwargs: None)
    monkeypatch.setattr(supervisor, "make_supervisor_fn", make_supervisor_fn)
    monkeypatch.setattr(lanes, "_make_lane_agent_factory", lambda **_kwargs: None)
    monkeypatch.setattr(lanes, "_require_lane_provider_capabilities", lambda *_args: None)
    return captured


@pytest.mark.parametrize(
    ("relay", "supervisor_backend", "provider", "model"),
    [
        (True, "", "claude", "claude-opus-5"),
        (True, "codex", "codex", "gpt-6-astra"),
        (False, "codex", "codex", "claude-opus-5"),
    ],
    ids=["relay_follows_the_actor", "relay_honours_the_flag", "no_relay_honours_the_flag"],
)
def test_supervisor_backend_flag_is_honoured(monkeypatch, tmp_path, relay, supervisor_backend, provider, model):
    captured = _capture_supervisor(monkeypatch)
    config = Config(workspace=str(tmp_path), agent_backend="claude", agent_model="claude-opus-5")
    builder = SearchActorBuildContext(
        config=config,
        setup=_Setup(),
        lane_binding=None,
        lanes=1,
        program_md="",
        specialist_definitions={},
        critic_enabled=False,
        supervisor_backend=supervisor_backend,
        relay=_relay_config() if relay else None,
    )

    builder.build(AgentProfile("claude", "claude", "claude-opus-5", "max"))

    assert captured["backend"] == provider
    assert captured["config"].agent_model == model


def test_executable_override_stays_with_the_primary_provider(tmp_path):
    from kernelforge.workflows.forge_loop.bootstrap.relay import config_for_profile

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


def _availability(**available: bool) -> AgentAvailabilitySnapshot:
    def status(provider: str, model: str) -> ProfileAvailability:
        up = available[provider]
        return ProfileAvailability(
            provider=provider,
            model=model,
            provider_available=True,
            model_available=up,
            available=up,
            reason="" if up else f"{provider} down",
        )

    return AgentAvailabilitySnapshot(
        profiles={"claude": status("claude", "claude-opus-5"), "codex": status("codex", "gpt-6-astra")}
    )


@pytest.mark.parametrize(
    ("primary", "available", "preference", "expected"),
    [
        ("codex", {"claude": True, "codex": True}, StartPreference("claude"), "codex"),
        ("codex", {"claude": True, "codex": False}, StartPreference("claude"), "claude"),
        ("claude", {"claude": False, "codex": True}, StartPreference("claude"), "codex"),
        ("claude", {"claude": True, "codex": True}, StartPreference("claude", "codex"), "codex"),
        ("claude", {"claude": False, "codex": True}, StartPreference("", "codex"), "codex"),
        ("claude", {"claude": True, "codex": False}, StartPreference("claude", "codex"), "claude"),
    ],
    ids=[
        "available_primary",
        "fallback_provider",
        "fallback_naming_the_primary_takes_any_member",
        "resume_continues_on_its_profile",
        "resume_continues_with_fallback_disabled",
        "resume_profile_down_starts_on_the_primary",
    ],
)
def test_start_profile(primary, available, preference, expected):
    assert select_start_profile(_relay_config(primary), _availability(**available), preference) == expected


def test_the_fallback_provider_is_preferred_among_the_other_members():
    config = AgentRelayConfig(
        primary_profile_id="claude",
        profiles=(
            AgentProfile("claude", "claude", "claude-opus-5"),
            AgentProfile("codex", "codex", "gpt-6-astra"),
            AgentProfile("kimi", "kimi", "kimi-k3"),
        ),
    )
    availability = _availability(claude=False, codex=True)
    availability.profiles["kimi"] = ProfileAvailability(provider="kimi", model="kimi-k3", available=True)

    assert select_start_profile(config, availability, StartPreference("kimi")) == "kimi"
    assert select_start_profile(config, availability, StartPreference("claude")) == "codex"


@pytest.mark.parametrize(
    ("available", "preference", "message"),
    [
        (
            {"claude": False, "codex": True},
            StartPreference(""),
            "Agent profile claude is unavailable: claude down; provider fallback is disabled",
        ),
        (
            {"claude": False, "codex": True},
            StartPreference("", "claude"),
            "Agent profile claude is unavailable: claude down; provider fallback is disabled",
        ),
        (
            {"claude": False, "codex": False},
            StartPreference("claude"),
            "no Agent profile can start this session: Agent profile claude is unavailable: claude down; "
            "Agent profile codex is unavailable: codex down",
        ),
    ],
    ids=["fallback_disabled", "resume_profile_down_and_fallback_disabled", "no_member_available"],
)
def test_no_runnable_start_profile_refuses_startup(available, preference, message):
    with pytest.raises(ValueError) as error:
        select_start_profile(_relay_config("claude"), _availability(**available), preference)

    assert str(error.value).startswith(message)


def test_search_actor_pool_starts_on_the_fallback_when_the_primary_cannot_run():
    relay_config = _relay_config("codex")
    pool = build_search_actor_pool(
        relay_config,
        availability=_availability(claude=True, codex=False),
        start=StartPreference("claude"),
        bundle_factory=lambda profile: SearchActorBundle(profile=profile, agent_fn=lambda: None),
    )

    assert pool.start_profile_id == "claude"
    assert not pool.has_bundle("codex")


def test_switching_the_implementer_provider_drops_the_other_providers_cli_and_options(monkeypatch, tmp_path):
    from kernelforge.workflows.forge_loop.roles import agent as agent_module

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
    from kernelforge.workflows.forge_loop.bootstrap.relay import agent_relay_summary, relay_switch_possible
    from kernelforge.workflows.forge_loop.search.agent_relay import AgentAvailabilitySnapshot, ProfileAvailability
    from kernelforge.workflows.forge_loop.campaign.run_state import RunState

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


@pytest.mark.parametrize(
    ("claude_up", "line"),
    [
        (True, "Agent relay: resuming on profile codex, which the campaign was running"),
        (False, "WARNING: primary Agent profile claude is unavailable; starting on profile codex"),
    ],
    ids=["resumed_profile", "primary_down"],
)
def test_startup_report_says_why_the_session_is_not_on_the_primary(capsys, claude_up, line):
    from kernelforge.workflows.forge_loop.bootstrap.relay import _report_search_actors

    pool = build_search_actor_pool(
        _relay_config("claude"),
        availability=_availability(claude=claude_up, codex=True),
        start=StartPreference("claude", "codex"),
        bundle_factory=lambda profile: SearchActorBundle(profile=profile, agent_fn=lambda: None),
    )

    _report_search_actors(pool, relay_enabled=True)

    assert line in capsys.readouterr().out


def test_startup_report_explains_a_session_without_relay_and_how_to_enable_it(capsys):
    from kernelforge.workflows.forge_loop.bootstrap.relay import _report_search_actors

    pool = build_search_actor_pool(
        _relay_config("claude"),
        availability=_availability(claude=True, codex=False),
        start=StartPreference("claude"),
        bundle_factory=lambda profile: SearchActorBundle(profile=profile, agent_fn=lambda: None),
    )

    _report_search_actors(pool, relay_enabled=True)

    out = capsys.readouterr().out
    assert "Agent profile: codex / gpt-6-astra: unavailable (codex down)" in out
    assert "Model relay: off for this session, only claude passed its startup check" in out
    assert "OPENAI_* for codex" in out
    assert "WARNING" not in out
