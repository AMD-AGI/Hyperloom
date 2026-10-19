# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Startup construction of model-relay profiles and search actors."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from kernelforge.cli.lanes import LaneBinding
from kernelforge.config import Config, resolve_agent_model
from kernelforge.llm.gateway import (
    anthropic_environment_from_openai,
    openai_gateway_from_anthropic,
    resolve_openai_gateway,
)
from kernelforge.loop.agent_relay import (
    AgentAvailabilitySnapshot,
    AgentProfile,
    AgentRelayConfig,
    ProfileAvailability,
    ProviderAvailability,
    select_start_profile,
)
from kernelforge.loop.search_actors import SearchActorBundle, SearchActorPool
from kernelforge.orchestrator.agent import ImplementerSetup


@dataclass
class SearchActorBuildContext:
    """Shared campaign inputs used to construct one strict search actor per Agent profile."""

    config: Config
    setup: ImplementerSetup
    lane_binding: LaneBinding
    lanes: int
    program_md: str
    specialist_definitions: dict
    critic_enabled: bool
    relay_enabled: bool
    primary_provider: str
    usage: Any = None
    supervisor_backend: str = ""

    def build(self, profile: AgentProfile) -> SearchActorBundle:
        """Build and probe all profile-bound planning and implementation services."""
        from kernelforge.cli.lanes import (
            _make_lane_agent_factory,
            _require_lane_provider_capabilities,
        )
        from kernelforge.orchestrator.agent import make_agent_fn
        from kernelforge.orchestrator.orchestration import make_orchestration_service
        from kernelforge.orchestrator.supervisor import make_supervisor_fn

        actor_config = (
            config_for_profile(self.config, profile, primary_provider=self.primary_provider)
            if self.relay_enabled
            else self.config
        )
        setup = replace(self.setup, session=replace(self.setup.session, agent_backend=profile.provider))
        actor_fn = make_agent_fn(actor_config, setup, usage=self.usage)
        effective_profile = self._resolved_profile(profile, actor_fn)
        _require_lane_provider_capabilities(effective_profile.provider, self.lanes)
        lane_factory = _make_lane_agent_factory(
            make_agent=make_agent_fn,
            config=actor_config,
            binding=self.lane_binding,
            setup=setup,
            usage=self.usage,
        )
        orchestration = make_orchestration_service(
            config=actor_config,
            usage=self.usage,
            definitions=self.specialist_definitions,
            enable_plan_critic=self.critic_enabled,
        )
        supervisor = make_supervisor_fn(
            program_md=self.program_md,
            gpu_target=actor_config.gpu_target,
            backend=profile.provider if self.relay_enabled else self.supervisor_backend or profile.provider,
            usage=self.usage,
            config=actor_config,
        )
        return SearchActorBundle(effective_profile, actor_fn, orchestration, lane_factory, supervisor)

    def _resolved_profile(self, profile: AgentProfile, actor_fn: Callable) -> AgentProfile:
        effective = (
            getattr(actor_fn, "backend_name", profile.provider),
            getattr(actor_fn, "backend_model", profile.model),
        )
        if self.relay_enabled and effective != (profile.provider, profile.model):
            raise ValueError(f"relay profile {profile.profile_id} resolved to {effective[0]}/{effective[1]}")
        if effective == (profile.provider, profile.model):
            return profile
        # Without relay a provider fallback is allowed; the identity stays, the record names what actually runs.
        return AgentProfile(profile.profile_id, effective[0], effective[1], profile.reasoning_effort)


def resolve_default_relay_config(
    config: Config,
    *,
    blockage_threshold: int = 3,
) -> AgentRelayConfig:
    """Build one profile per registered provider, the selected one first as primary.

    Every model follows the rule the primary already uses: ``--model`` for the primary, else the provider's model
    variable, else the registry's ``default_model``. A model the gateway does not serve fails the startup probe and
    leaves that profile unavailable rather than running something else.
    """
    from kernelforge.agent_backends.registry import list_agent_providers

    strict_config = replace(config, agent_fallback_provider="")
    primary = strict_config.agent_runtime()
    profiles = [AgentProfile(primary.provider, primary.provider, primary.model, primary.reasoning_effort)]
    for provider in list_agent_providers():
        if provider == primary.provider:
            continue
        runtime = strict_config.agent_runtime_for(provider, resolve_agent_model(provider))
        profiles.append(AgentProfile(runtime.provider, runtime.provider, runtime.model, runtime.reasoning_effort))
    return AgentRelayConfig(
        primary_profile_id=primary.provider,
        profiles=tuple(profiles),
        blockage_threshold=blockage_threshold,
    )


def config_for_profile(
    config: Config,
    profile: AgentProfile,
    *,
    primary_provider: str,
    precheck: bool = False,
) -> Config:
    """Create a strict Config view for one profile without provider fallback."""
    is_primary = profile.provider == primary_provider
    return replace(
        config,
        agent_backend=profile.provider,
        agent_model=profile.model,
        # ``--agent-cli``/``FORGE_AGENT_CLI`` and the provider options belong to the primary provider.
        agent_cli=config.agent_cli if is_primary else "",
        agent_options=dict(config.agent_options) if is_primary else _borrowed_line_options(profile.provider),
        agent_fallback_provider="",
        agent_precheck=precheck,
    )


def _borrowed_line_options(provider: str) -> dict:
    """Point a relay provider that has no gateway of its own at the other configured line.

    An explicitly configured line always wins. The borrowed line is only a candidate:
    the startup probe decides whether the gateway serves this provider's protocol.
    """
    from kernelforge.agent_backends.registry import get_agent_provider

    if provider == "codex" and not resolve_openai_gateway().is_complete():
        gateway = openai_gateway_from_anthropic()
        if gateway.is_complete():
            return {"gateway": {"base_url": gateway.base_url, "key_env": gateway.key_env, "headers": gateway.headers}}
    if provider == "claude" and not get_agent_provider("claude").credentialed(os.environ):
        environment = anthropic_environment_from_openai()
        if environment:
            return {"env": environment}
    return {}


def probe_agent_profiles(
    relay_config: AgentRelayConfig,
    *,
    config: Config,
    workspace: str,
) -> AgentAvailabilitySnapshot:
    """Probe every exact provider/model pair before optimization setup begins."""
    from kernelforge.agent_backends.registry import (
        create_registered_backend,
        get_agent_provider,
    )

    provider_status: dict[str, ProviderAvailability] = {}
    profile_status: dict[str, ProfileAvailability] = {}
    for profile in relay_config.profiles:
        registration = get_agent_provider(profile.provider)
        provider_available, failure = _provider_is_available(registration)
        model_available = False
        if provider_available:
            borrowed = False
            try:
                strict_config = config_for_profile(
                    config,
                    profile,
                    primary_provider=relay_config.primary_profile.provider,
                    precheck=True,
                )
                borrowed = profile.provider != relay_config.primary_profile.provider and bool(
                    strict_config.agent_options
                )
                if borrowed:
                    print(
                        f"  Agent profile: {profile.profile_id} has no gateway of its own; "
                        "probing the other configured gateway on its protocol",
                        file=sys.stderr,
                    )
                create_registered_backend(
                    strict_config.agent_runtime(),
                    preflight=True,
                    probe_cwd=workspace,
                )
                model_available = True
                failure = ""
            except Exception as error:  # noqa: BLE001 - isolate optional profiles
                failure = f"{type(error).__name__}: {error}"
                if borrowed:
                    failure = f"the other configured gateway does not serve this protocol: {failure}"
        provider_status[profile.provider] = ProviderAvailability(
            available=provider_available,
            reason="" if provider_available else failure,
        )
        profile_status[profile.profile_id] = ProfileAvailability(
            provider=profile.provider,
            model=profile.model,
            provider_available=provider_available,
            model_available=model_available,
            available=provider_available and model_available,
            reason="" if model_available else failure,
        )
    return AgentAvailabilitySnapshot(
        checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        providers=provider_status,
        profiles=profile_status,
    )


def build_search_actor_pool(
    relay_config: AgentRelayConfig,
    *,
    bundle_factory: Callable[[AgentProfile], SearchActorBundle],
    availability: AgentAvailabilitySnapshot | None = None,
    fallback_provider: str = "",
) -> SearchActorPool:
    """Construct every available profile's search actor and choose the profile this session starts on.

    Raises ``ValueError`` when neither the primary nor the fallback provider's profile can run.
    """
    from kernelforge.agent_backends.registry import get_agent_provider

    provider_status: dict[str, ProviderAvailability] = {}
    profile_status: dict[str, ProfileAvailability] = {}
    bundles: dict[str, SearchActorBundle] = {}
    for profile in relay_config.profiles:
        saved = availability.profiles.get(profile.profile_id) if availability is not None else None
        registration = get_agent_provider(profile.provider)
        provider_available, provider_reason = _provider_is_available(registration)
        if saved is not None:
            provider_available = saved.provider_available
            provider_reason = saved.reason
        bundle = None
        failure = provider_reason
        if provider_available and (saved is None or saved.available):
            try:
                bundle = bundle_factory(profile)
            except Exception as error:  # noqa: BLE001 - availability must isolate one optional profile
                failure = f"{type(error).__name__}: {error}"
        available = bundle is not None
        provider_status[profile.provider] = ProviderAvailability(
            available=provider_available,
            reason="" if provider_available else provider_reason,
        )
        profile_status[profile.profile_id] = ProfileAvailability(
            provider=profile.provider,
            model=profile.model,
            provider_available=provider_available,
            model_available=available and (saved is None or saved.model_available),
            available=available,
            reason="" if available else failure,
        )
        if bundle is not None:
            bundles[profile.profile_id] = bundle

    snapshot = AgentAvailabilitySnapshot(
        checked_at=(
            availability.checked_at if availability is not None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ),
        session_index=(availability.session_index if availability is not None else 0),
        providers=provider_status,
        profiles=profile_status,
    )
    return SearchActorPool(
        config=relay_config,
        availability=snapshot,
        bundles=bundles,
        start_profile_id=select_start_profile(relay_config, snapshot, fallback_provider=fallback_provider),
    )


def relay_switch_possible(availability: AgentAvailabilitySnapshot) -> bool:
    """Whether at least two profiles passed the startup check, so a blockage has somewhere to switch to."""
    return sum(1 for status in availability.profiles.values() if status.available) >= 2


def agent_relay_summary(run_state) -> dict:
    """What the campaign result reports about relay: per-profile availability and the active profile."""
    snapshot = run_state.agent_availability_snapshot
    relay = run_state.agent_relay
    return {
        "switch_possible": relay_switch_possible(snapshot),
        "active_profile_id": relay.active_profile_id,
        "blockage_epoch": relay.blockage_epoch,
        "profiles": {
            profile_id: {
                "provider": status.provider,
                "model": status.model,
                "available": status.available,
                "reason": status.reason,
            }
            for profile_id, status in snapshot.profiles.items()
        },
    }


def _provider_is_available(registration) -> tuple[bool, str]:
    try:
        if not registration.availability():
            return False, "provider SDK or CLI is not installed"
    except Exception as error:  # noqa: BLE001 - third-party provider predicate
        return False, f"{type(error).__name__}: {error}"
    return True, ""


__all__ = [
    "SearchActorBuildContext",
    "agent_relay_summary",
    "build_search_actor_pool",
    "config_for_profile",
    "probe_agent_profiles",
    "relay_switch_possible",
    "resolve_default_relay_config",
]
