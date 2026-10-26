# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runtime bundles for switching the complete KernelForge search actor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from kernelforge.loop.agent_relay import (
    AgentAvailabilitySnapshot,
    AgentProfile,
    AgentRelayConfig,
    ProfileAvailability,
)


@dataclass(frozen=True)
class SearchActorBundle:
    """All profile-bound services used to plan and implement an iteration."""

    profile: AgentProfile
    agent_fn: Callable | None
    orchestration_service: Any = None
    agent_factory: Callable | None = None
    supervisor_fn: Callable | None = None
    # Builds a further planning service for this profile, so planning units of one round run side by side without
    # sharing one service's per-run state.
    orchestration_factory: Callable[[], Any] | None = None


class SearchActorPool:
    """Validated runtime bundles and their process-local availability snapshot."""

    def __init__(
        self,
        *,
        config: AgentRelayConfig,
        availability: AgentAvailabilitySnapshot,
        bundles: Mapping[str, SearchActorBundle],
        start_profile_id: str = "",
        recheck: Callable[[AgentProfile], ProfileAvailability] | None = None,
    ) -> None:
        self.config = config
        self.availability = availability
        # Re-runs one profile's startup check; ``None`` when this session cannot check a profile again.
        self.recheck = recheck
        self._bundles = dict(bundles)
        self.start_profile_id = start_profile_id or config.primary_profile_id
        configured = {profile.profile_id for profile in config.profiles}
        unknown = set(self._bundles) - configured
        if unknown:
            raise ValueError("search actor bundles contain unknown profiles: " + ", ".join(sorted(unknown)))
        for profile_id, bundle in self._bundles.items():
            if bundle.profile.profile_id != profile_id:
                raise ValueError(f"search actor bundle key does not match profile: {profile_id}")
        if self.start_profile_id not in self._bundles:
            raise ValueError(f"start search actor bundle is unavailable: {self.start_profile_id}")

    @property
    def profiles(self) -> tuple[AgentProfile, ...]:
        return self.config.profiles

    @property
    def blockage_threshold(self) -> int:
        return self.config.blockage_threshold

    def bundle(self, profile_id: str) -> SearchActorBundle:
        """Return one available bundle without probing or falling back."""
        try:
            return self._bundles[profile_id]
        except KeyError as error:
            raise ValueError(f"search actor profile is unavailable: {profile_id}") from error

    def has_bundle(self, profile_id: str) -> bool:
        return profile_id in self._bundles


__all__ = ["SearchActorBundle", "SearchActorPool"]
