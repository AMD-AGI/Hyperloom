# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Startup construction of model-relay profiles and search actors."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from kernelforge.config import Config, resolve_agent_model
from kernelforge.loop.agent_relay import (
    AgentAvailabilitySnapshot,
    AgentProfile,
    AgentRelayConfig,
    ProfileAvailability,
    ProviderAvailability,
)
from kernelforge.loop.search_actors import SearchActorBundle, SearchActorPool


@dataclass
class SearchActorBuildContext:
    """Shared campaign inputs used to construct one strict search actor."""

    config: Config
    program_md: str
    pr_task_context: str
    pr_kb_repo: str
    kernel_backend: str
    driver: str
    snr_threshold: float
    max_blocks: int
    session_timeout_sec: int
    validation_timeout_sec: int
    bench_timeout_sec: int
    bench_repeat: int
    permission_mode: str | None
    task_type: str
    source_files: list[str]
    target_functions: list[str]
    profiling_enabled: bool
    commit_new_paths: list[str]
    usage: Any
    lanes: int
    workspace_dir: str
    specialist_definitions: dict
    critic_enabled: bool
    relay_enabled: bool
    primary_provider: str
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
        actor_fn = make_agent_fn(
            config=actor_config,
            program_md=self.program_md,
            pre_task_context=self.pr_task_context,
            pr_kb_repo=self.pr_kb_repo,
            kernel_backend_name=self.kernel_backend,
            insession_gate=True,
            driver_script=self.driver,
            snr_threshold=self.snr_threshold,
            max_blocks=self.max_blocks,
            session_timeout_sec=self.session_timeout_sec,
            validation_timeout_sec=self.validation_timeout_sec,
            bench_timeout_sec=self.bench_timeout_sec,
            bench_repeat=self.bench_repeat,
            permission_mode=self.permission_mode,
            task_type=self.task_type,
            source_files=self.source_files,
            target_functions=self.target_functions,
            profiling_enabled=self.profiling_enabled,
            agent_backend=profile.provider,
            commit_new_paths=self.commit_new_paths,
            usage=self.usage,
        )
        effective_profile = self._resolved_profile(profile, actor_fn)
        _require_lane_provider_capabilities(effective_profile.provider, self.lanes)
        lane_factory = _make_lane_agent_factory(
            make_agent=make_agent_fn,
            config=actor_config,
            workspace_dir=self.workspace_dir,
            driver=self.driver,
            source_files=self.source_files,
            session_kwargs=self._lane_session_kwargs(profile),
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
        return AgentProfile(
            effective[0],
            effective[0],
            effective[1],
            profile.reasoning_effort,
        )

    def _lane_session_kwargs(self, profile: AgentProfile) -> dict:
        return {
            "program_md": self.program_md,
            "pre_task_context": self.pr_task_context,
            "pr_kb_repo": self.pr_kb_repo,
            "kernel_backend_name": self.kernel_backend,
            "snr_threshold": self.snr_threshold,
            "max_blocks": self.max_blocks,
            "session_timeout_sec": self.session_timeout_sec,
            "validation_timeout_sec": self.validation_timeout_sec,
            "bench_timeout_sec": self.bench_timeout_sec,
            "bench_repeat": self.bench_repeat,
            "permission_mode": self.permission_mode,
            "task_type": self.task_type,
            "commit_new_paths": self.commit_new_paths,
            "target_functions": self.target_functions,
            "agent_backend": profile.provider,
            "usage": self.usage,
        }


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
    return replace(
        config,
        agent_backend=profile.provider,
        agent_model=profile.model,
        # ``--agent-cli``/``FORGE_AGENT_CLI`` names the primary provider's executable.
        agent_cli=config.agent_cli if profile.provider == primary_provider else "",
        agent_fallback_provider="",
        agent_precheck=precheck,
    )


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
            try:
                strict_config = config_for_profile(
                    config,
                    profile,
                    primary_provider=relay_config.primary_profile.provider,
                    precheck=True,
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
    snapshot = AgentAvailabilitySnapshot(
        checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        providers=provider_status,
        profiles=profile_status,
    )
    primary = snapshot.profiles[relay_config.primary_profile_id]
    if not primary.available:
        raise ValueError(f"primary Agent profile is unavailable: {primary.reason}")
    return snapshot


def build_search_actor_pool(
    relay_config: AgentRelayConfig,
    *,
    bundle_factory: Callable[[AgentProfile], SearchActorBundle],
    availability: AgentAvailabilitySnapshot | None = None,
) -> SearchActorPool:
    """Probe every profile by constructing its complete search actor at startup."""
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
    )


def _provider_is_available(registration) -> tuple[bool, str]:
    try:
        if not registration.availability():
            return False, "provider SDK or CLI is not installed"
    except Exception as error:  # noqa: BLE001 - third-party provider predicate
        return False, f"{type(error).__name__}: {error}"
    return True, ""


__all__ = [
    "SearchActorBuildContext",
    "build_search_actor_pool",
    "config_for_profile",
    "probe_agent_profiles",
    "resolve_default_relay_config",
]
