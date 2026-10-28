# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Startup construction of model-relay profiles and search actors."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

from kernelforge.cli.lanes import LaneBinding
from kernelforge.config import Config, resolve_agent_model
from kernelforge.agent_backends.runtime import agent_runtime, agent_runtime_for
from kernelforge.agent_backends.gateway import (
    anthropic_environment_from_openai,
    openai_gateway_from_anthropic,
    resolve_openai_gateway,
)
from kernelforge.loop.agent_relay import (
    AgentAvailabilitySnapshot,
    AgentProfile,
    AgentProfileSpec,
    AgentRelayConfig,
    AgentRelayPolicy,
    ProfileAvailability,
    ProviderAvailability,
    StartPreference,
    select_start_profile,
)
from kernelforge.loop.search_actors import SearchActorBundle, SearchActorPool
from kernelforge.orchestrator.agent import ImplementerSetup

log = logging.getLogger(__name__)


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
    usage: Any = None
    supervisor_backend: str = ""
    # This session's relay; ``None`` for a campaign without relay, whose one actor may fall back to another provider.
    relay: AgentRelayConfig | None = None

    def build(self, profile: AgentProfile) -> SearchActorBundle:
        """Build and probe all profile-bound planning and implementation services."""
        from kernelforge.cli.lanes import (
            _make_lane_agent_factory,
            _require_lane_provider_capabilities,
        )
        from kernelforge.orchestrator.agent import make_agent_fn
        from kernelforge.orchestrator.orchestration import make_orchestration_service
        from kernelforge.orchestrator.supervisor import make_supervisor_fn

        actor_config = self._profile_config(profile) if self.relay is not None else self.config
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
        supervisor_provider = self.supervisor_backend or profile.provider
        supervisor = make_supervisor_fn(
            program_md=self.program_md,
            gpu_target=actor_config.gpu_target,
            backend=supervisor_provider,
            usage=self.usage,
            config=self._supervisor_config(supervisor_provider, profile, actor_config),
        )
        return SearchActorBundle(effective_profile, actor_fn, orchestration, lane_factory, supervisor)

    def _profile_config(self, profile: AgentProfile) -> Config:
        return config_for_profile(self.config, profile, primary_provider=self.relay.primary_profile.provider)

    def _supervisor_config(self, provider: str, profile: AgentProfile, actor_config: Config) -> Config:
        """The Supervisor shares the actor's line unless ``--supervisor-backend`` names another provider."""
        if provider == profile.provider:
            return actor_config
        members = self.relay.profiles if self.relay is not None else ()
        member = next((candidate for candidate in members if candidate.provider == provider), None)
        return self._profile_config(member) if member is not None else self.config

    def _resolved_profile(self, profile: AgentProfile, actor_fn: Callable) -> AgentProfile:
        effective = (
            getattr(actor_fn, "backend_name", profile.provider),
            getattr(actor_fn, "backend_model", profile.model),
        )
        if self.relay is not None and effective != (profile.provider, profile.model):
            raise ValueError(f"relay profile {profile.profile_id} resolved to {effective[0]}/{effective[1]}")
        if effective == (profile.provider, profile.model):
            return profile
        # Without relay a provider fallback is allowed; the identity stays, the record names what actually runs.
        return AgentProfile(profile.profile_id, effective[0], effective[1], profile.reasoning_effort)


def resolve_relay_policy(
    config: Config,
    *,
    blockage_threshold: int = 3,
) -> AgentRelayPolicy:
    """One relay member per auto-selectable provider, the provider this campaign starts on first as primary.

    A provider only an explicit name reaches joins the relay only as the primary that name chose.
    """
    from kernelforge.agent_backends.registry import get_agent_provider, list_agent_providers

    primary = agent_runtime(replace(config, agent_fallback_provider="")).provider
    members = (name for name in list_agent_providers() if name != primary and get_agent_provider(name).auto_selectable)
    providers = [primary, *members]
    return AgentRelayPolicy(
        primary_profile_id=primary,
        profiles=tuple(AgentProfileSpec(provider, provider) for provider in providers),
        blockage_threshold=blockage_threshold,
    )


def resolve_session_relay(policy: AgentRelayPolicy | None, config: Config) -> AgentRelayConfig | None:
    """Resolve every relay member's model and reasoning effort from this session's options.

    The primary takes ``--model``; every member without one takes its provider's model variable, else the registry's
    ``default_model``. A model the gateway does not serve fails the startup probe and leaves that profile unavailable
    rather than running something else. Raises ``ValueError`` when the session names a primary provider other than
    the one the campaign was created with.
    """
    if policy is None:
        return None
    primary = policy.primary_profile
    if config.agent_backend not in ("auto", primary.provider):
        raise ValueError(
            f"this campaign's model relay starts on {primary.provider}; the agent backend cannot change on resume "
            f"(got {config.agent_backend} from --agent-backend or FORGE_AGENT_BACKEND)"
        )
    strict_config = replace(config, agent_fallback_provider="")
    profiles = []
    for member in policy.profiles:
        model = config.agent_model if member.profile_id == primary.profile_id else ""
        runtime = agent_runtime_for(strict_config, member.provider, model or resolve_agent_model(member.provider))
        profiles.append(AgentProfile(member.profile_id, member.provider, runtime.model, runtime.reasoning_effort))
    return AgentRelayConfig(
        primary_profile_id=primary.profile_id,
        profiles=tuple(profiles),
        blockage_threshold=policy.blockage_threshold,
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
    relay_config: AgentRelayConfig | None,
    *,
    config: Config,
    workspace: str,
) -> AgentAvailabilitySnapshot | None:
    """Probe every exact provider/model pair before optimization setup begins; ``None`` for a campaign without relay."""
    if relay_config is None:
        return None
    profile_status = {
        profile.profile_id: probe_agent_profile(profile, relay_config=relay_config, config=config, workspace=workspace)
        for profile in relay_config.profiles
    }
    provider_status = {
        status.provider: ProviderAvailability(
            available=status.provider_available,
            reason="" if status.provider_available else status.reason,
        )
        for status in profile_status.values()
    }
    return AgentAvailabilitySnapshot(
        checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        providers=provider_status,
        profiles=profile_status,
    )


def probe_agent_profile(
    profile: AgentProfile,
    *,
    relay_config: AgentRelayConfig,
    config: Config,
    workspace: str,
) -> ProfileAvailability:
    """Run one profile's strict provider/model check: at startup, and again after its API retries ran out."""
    from kernelforge.agent_backends.registry import (
        create_registered_backend,
        get_agent_provider,
    )

    provider_available, failure = _provider_is_available(get_agent_provider(profile.provider))
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
            borrowed = profile.provider != relay_config.primary_profile.provider and bool(strict_config.agent_options)
            if borrowed:
                print(
                    f"  Agent profile: {profile.profile_id} has no gateway of its own; "
                    "probing the other configured gateway on its protocol",
                    file=sys.stderr,
                )
            create_registered_backend(
                agent_runtime(strict_config),
                preflight=True,
                probe_cwd=workspace,
            )
            model_available = True
            failure = ""
        except Exception as error:  # isolation boundary: one agent profile's strict preflight
            log.warning("Agent profile preflight failed for %s", profile.profile_id, exc_info=True)
            failure = f"{type(error).__name__}: {error}"
            if borrowed:
                failure = _borrowed_line_failure(profile.provider, strict_config.agent_options, failure)
    return ProfileAvailability(
        provider=profile.provider,
        model=profile.model,
        provider_available=provider_available,
        model_available=model_available,
        available=provider_available and model_available,
        reason="" if model_available else failure,
    )


def build_search_actor_pool(
    relay_config: AgentRelayConfig,
    *,
    bundle_factory: Callable[[AgentProfile], SearchActorBundle],
    availability: AgentAvailabilitySnapshot | None = None,
    start: StartPreference = StartPreference(),
    recheck: Callable[[AgentProfile], ProfileAvailability] | None = None,
) -> SearchActorPool:
    """Construct every available profile's search actor and choose the profile this session starts on.

    Raises ``ValueError`` when no profile ``start`` allows can run.
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
            except Exception as error:  # isolation boundary: one profile's search-actor build
                log.warning("search actor build failed for %s", profile.profile_id, exc_info=True)
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
        start_profile_id=select_start_profile(relay_config, snapshot, start),
        recheck=recheck,
    )


@dataclass(frozen=True)
class SearchActorStartup:
    """The session's search actors, and whether the campaign relays between them."""

    pool: SearchActorPool
    relay_enabled: bool

    @property
    def start_bundle(self) -> SearchActorBundle:
        return self.pool.bundle(self.pool.start_profile_id)

    @property
    def loop_search_actors(self) -> SearchActorPool | None:
        """What the loop switches between; ``None`` keeps a campaign without relay on its single actor."""
        return self.pool if self.relay_enabled else None

    def analysis_config(self, config: Config) -> Config:
        """The shared Analysis Agent runs on the profile the session starts on."""
        if not self.relay_enabled:
            return config
        primary = self.pool.config.primary_profile.provider
        return config_for_profile(config, self.start_bundle.profile, primary_provider=primary)


def start_search_actors(
    session_relay: AgentRelayConfig | None,
    builder: SearchActorBuildContext,
    *,
    availability: AgentAvailabilitySnapshot | None,
    workspace: str,
    resume_profile_id: str = "",
) -> SearchActorStartup:
    """Build every runnable profile's search actor, choose the start profile, and report both.

    A campaign without relay runs the configured provider as its only profile, with provider fallback allowed.
    ``resume_profile_id`` is the profile a resumed campaign was running. Raises ``ValueError`` when no profile the
    session may start on can run.
    """
    relay_enabled = session_relay is not None
    relay_config = session_relay or _single_profile_relay(builder.config)
    pool = build_search_actor_pool(
        relay_config,
        bundle_factory=replace(builder, relay=session_relay).build,
        availability=availability,
        start=StartPreference(builder.config.agent_fallback_provider, resume_profile_id),
        recheck=(
            partial(probe_agent_profile, relay_config=relay_config, config=builder.config, workspace=workspace)
            if relay_enabled
            else None
        ),
    )
    _report_search_actors(pool, relay_enabled=relay_enabled)
    return SearchActorStartup(pool, relay_enabled)


def _single_profile_relay(config: Config) -> AgentRelayConfig:
    runtime = agent_runtime(config)
    profile = AgentProfile(runtime.provider, runtime.provider, runtime.model, runtime.reasoning_effort)
    return AgentRelayConfig(primary_profile_id=profile.profile_id, profiles=(profile,))


def _report_search_actors(pool: SearchActorPool, *, relay_enabled: bool) -> None:
    for profile_id, status in pool.availability.profiles.items():
        state = "available" if status.available else f"unavailable ({status.reason})"
        print(f"  Agent profile: {profile_id} / {status.model}: {state}")
    start, primary = pool.start_profile_id, pool.config.primary_profile_id
    primary_status = pool.availability.profiles.get(primary)
    if start != primary and primary_status is not None and primary_status.available:
        print(f"  Agent relay: resuming on profile {start}, which the campaign was running")
    elif start != primary:
        print(f"  WARNING: primary Agent profile {primary} is unavailable; starting on profile {start}")
    if relay_enabled and not relay_switch_possible(pool.availability):
        print(
            f"  Model relay: off for this session, only {start} passed its startup check (reasons above). "
            "Give another provider a line of its own (ANTHROPIC_* for claude, OPENAI_* for codex) or use a gateway "
            "serving both protocols to turn it on; until then a search blockage runs the Supervisor"
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


def _borrowed_line_failure(provider: str, options: dict, failure: str) -> str:
    """Name the borrowed address that failed and the variables that give the provider a line of its own."""
    if provider == "codex":
        address, own_line = options["gateway"]["base_url"], "OPENAI_BASE_URL and OPENAI_API_KEY"
    else:
        address, own_line = options["env"]["ANTHROPIC_BASE_URL"], "ANTHROPIC_BASE_URL and an Anthropic key"
    return (
        f"the other configured gateway did not answer on this protocol at {address}; "
        f"set {own_line} if it routes this protocol elsewhere: {failure}"
    )


def _provider_is_available(registration) -> tuple[bool, str]:
    try:
        if not registration.availability():
            return False, "provider SDK or CLI is not installed"
    except Exception as error:  # isolation boundary: one provider's third-party availability predicate
        log.warning("Agent provider availability check failed: %s", error, exc_info=True)
        return False, f"{type(error).__name__}: {error}"
    return True, ""


__all__ = [
    "SearchActorBuildContext",
    "SearchActorStartup",
    "agent_relay_summary",
    "build_search_actor_pool",
    "config_for_profile",
    "probe_agent_profile",
    "probe_agent_profiles",
    "relay_switch_possible",
    "resolve_relay_policy",
    "resolve_session_relay",
    "start_search_actors",
]
