# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Durable state and pure policy helpers for KernelForge Agent relay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TypeVar


INTERVENTION_CROSS_HARNESS = "cross_harness"
INTERVENTION_SAME_HARNESS = "same_harness"
INTERVENTION_SUPERVISOR = "supervisor"
INTERVENTION_KINDS = frozenset(
    {
        "",
        INTERVENTION_CROSS_HARNESS,
        INTERVENTION_SAME_HARNESS,
        INTERVENTION_SUPERVISOR,
    }
)

_ProfileT = TypeVar("_ProfileT", "AgentProfileSpec", "AgentProfile")

_NON_SEARCH_DECISIONS = frozenset(
    {
        "AGENT_ERROR",
        "API_ERROR",
        "ORCHESTRATION_ERROR",
        "REVERT_CONTENDED",
        "REVERT_INTEGRITY",
    }
)


@dataclass(frozen=True)
class AgentProfileSpec:
    """One relay member as the campaign records it: its identity and harness, but not its model."""

    profile_id: str
    provider: str

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("agent profile id is required")
        if not self.provider.strip():
            raise ValueError("agent profile provider is required")


@dataclass(frozen=True)
class AgentRelayPolicy:
    """Campaign-immutable relay membership and blockage policy.

    Models and reasoning effort are session options: each session resolves them into an ``AgentRelayConfig``.
    """

    primary_profile_id: str
    profiles: tuple[AgentProfileSpec, ...]
    blockage_threshold: int = 3

    def __post_init__(self) -> None:
        _validate_relay_membership(self.primary_profile_id, self.profiles, self.blockage_threshold)

    @property
    def primary_profile(self) -> AgentProfileSpec:
        return _profile_mapping(self.profiles)[self.primary_profile_id]

    @classmethod
    def from_dict(cls, payload: object) -> AgentRelayPolicy:
        """Strictly rebuild the persisted relay policy."""
        data = _strict_mapping(payload, cls, "agent relay policy")
        raw_profiles = data["profiles"]
        if not isinstance(raw_profiles, (list, tuple)):
            raise ValueError("agent relay policy profiles must be a list")
        profiles = tuple(
            AgentProfileSpec(**_strict_mapping(item, AgentProfileSpec, "agent relay profile")) for item in raw_profiles
        )
        return cls(
            primary_profile_id=str(data["primary_profile_id"]),
            profiles=profiles,
            blockage_threshold=int(data["blockage_threshold"]),
        )


@dataclass(frozen=True)
class AgentProfile:
    """One relay member resolved for a session: its harness and the exact model it runs."""

    profile_id: str
    provider: str
    model: str
    reasoning_effort: str = ""

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("agent profile id is required")
        if not self.provider.strip():
            raise ValueError("agent profile provider is required")
        if not self.model.strip():
            raise ValueError("agent profile model is required")


@dataclass(frozen=True)
class AgentRelayConfig:
    """The relay policy with every member's model resolved for one session."""

    primary_profile_id: str
    profiles: tuple[AgentProfile, ...]
    blockage_threshold: int = 3

    def __post_init__(self) -> None:
        _validate_relay_membership(self.primary_profile_id, self.profiles, self.blockage_threshold)

    @property
    def primary_profile(self) -> AgentProfile:
        return _profile_mapping(self.profiles)[self.primary_profile_id]


@dataclass
class ProviderAvailability:
    """Availability of one harness/provider, from the session's startup check."""

    available: bool = False
    reason: str = ""


@dataclass
class ProfileAvailability:
    """Availability of one exact provider/model profile, from its last check this session."""

    provider: str = ""
    model: str = ""
    provider_available: bool = False
    model_available: bool = False
    available: bool = False
    reason: str = ""


@dataclass
class AgentAvailabilitySnapshot:
    """Availability facts established before the first optimization iteration."""

    checked_at: str = ""
    session_index: int = 0
    providers: dict[str, ProviderAvailability] = field(default_factory=dict)
    profiles: dict[str, ProfileAvailability] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: object) -> AgentAvailabilitySnapshot:
        """Strictly rebuild a persisted startup snapshot."""
        data = _strict_mapping(payload, cls, "agent availability snapshot")
        providers = _availability_entries(
            data["providers"],
            ProviderAvailability,
            "agent availability providers",
        )
        profiles = _availability_entries(
            data["profiles"],
            ProfileAvailability,
            "agent availability profiles",
        )
        snapshot = cls(
            checked_at=str(data["checked_at"]),
            session_index=int(data["session_index"]),
            providers=providers,
            profiles=profiles,
        )
        if snapshot.session_index < 0:
            raise ValueError("agent availability session_index must be non-negative")
        return snapshot


@dataclass
class AgentRelayState:
    """Durable relay cursor and blockage state."""

    active_profile_id: str = ""
    active_no_improvement_streak: int = 0
    blockage_epoch: int = 0
    intervention_kind: str = ""
    previous_profile_id: str = ""
    switch_iteration: int = 0
    switch_reason: str = ""

    def validate(self) -> None:
        """Reject state that cannot drive a deterministic relay decision."""
        if self.active_no_improvement_streak < 0:
            raise ValueError("agent relay streak must be non-negative")
        if self.blockage_epoch < 0:
            raise ValueError("agent relay blockage_epoch must be non-negative")
        if self.switch_iteration < 0:
            raise ValueError("agent relay switch_iteration must be non-negative")
        if self.intervention_kind not in INTERVENTION_KINDS:
            raise ValueError(f"unsupported relay intervention kind: {self.intervention_kind!r}")


@dataclass(frozen=True)
class RelayIntervention:
    """One policy decision at an iteration boundary."""

    kind: str
    target_profile_id: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.kind not in INTERVENTION_KINDS - {""}:
            raise ValueError(f"unsupported relay intervention kind: {self.kind!r}")
        if self.kind == INTERVENTION_SUPERVISOR and self.target_profile_id:
            raise ValueError("supervisor intervention cannot target another profile")
        if self.kind != INTERVENTION_SUPERVISOR and not self.target_profile_id:
            raise ValueError("profile switch intervention requires a target profile")


def decision_counts_toward_blockage(decision: str) -> bool:
    """Whether an attributable optimization outcome advances the relay streak."""
    normalized = str(decision or "").strip().upper()
    if not normalized or normalized == "KEEP":
        return False
    return normalized not in _NON_SEARCH_DECISIONS


def apply_relay_keep(state: AgentRelayState) -> AgentRelayState:
    """A KEEP ends the blockage epoch: the streak starts over and the switch record is cleared."""
    state.active_no_improvement_streak = 0
    state.intervention_kind = ""
    state.previous_profile_id = ""
    state.switch_iteration = 0
    state.switch_reason = ""
    return state


def apply_relay_search_outcome(
    state: AgentRelayState,
    *,
    decision: str,
    producer_profile_id: str,
) -> AgentRelayState:
    """Advance the active profile's streak by one non-KEEP outcome it produced and that counts as search.

    An outcome no profile produced, such as a stacked merge of two archived candidates, is not the active profile's.
    """
    if producer_profile_id != state.active_profile_id:
        return state
    if decision_counts_toward_blockage(decision):
        state.active_no_improvement_streak += 1
    return state


def relay_blockage_reached(state: AgentRelayState, *, threshold: int) -> bool:
    """Whether the active profile has reached the configured blockage threshold."""
    _require_positive_threshold(threshold)
    return state.active_no_improvement_streak >= threshold


def select_relay_intervention(
    state: AgentRelayState,
    *,
    profiles: Sequence[AgentProfile],
    availability: AgentAvailabilitySnapshot,
) -> RelayIntervention:
    """The first intervention this environment can run, checked in a fixed order at every blockage.

    A different harness and model, else the same harness with a different model, else the Supervisor on the active
    profile. Availability is the session's startup snapshot.
    """
    by_id = _profile_mapping(profiles)
    active = by_id.get(state.active_profile_id)
    if active is None:
        return RelayIntervention(
            kind=INTERVENTION_SUPERVISOR,
            reason="active profile is not present in campaign configuration",
        )
    candidates = [
        profile
        for profile in profiles
        if profile.profile_id != active.profile_id and _profile_is_available(profile, availability)
    ]
    for profile in candidates:
        if profile.provider != active.provider:
            return RelayIntervention(
                kind=INTERVENTION_CROSS_HARNESS,
                target_profile_id=profile.profile_id,
                reason="active profile reached the blockage threshold",
            )
    for profile in candidates:
        if profile.model != active.model:
            return RelayIntervention(
                kind=INTERVENTION_SAME_HARNESS,
                target_profile_id=profile.profile_id,
                reason="no cross-harness profile is available",
            )
    return RelayIntervention(
        kind=INTERVENTION_SUPERVISOR,
        reason="no other relay profile is available",
    )


def select_replacement_for_unavailable(
    state: AgentRelayState,
    *,
    profiles: Sequence[AgentProfile],
    availability: AgentAvailabilitySnapshot,
) -> RelayIntervention | None:
    """Choose an available profile to take over from an active profile this session cannot run."""
    by_id = _profile_mapping(profiles)
    active = by_id.get(state.active_profile_id)
    candidates = [
        profile
        for profile in profiles
        if profile.profile_id != state.active_profile_id and _profile_is_available(profile, availability)
    ]
    if not candidates:
        return None
    cross = [profile for profile in candidates if active is None or profile.provider != active.provider]
    target = (cross or candidates)[0]
    return RelayIntervention(
        kind=INTERVENTION_CROSS_HARNESS if cross else INTERVENTION_SAME_HARNESS,
        target_profile_id=target.profile_id,
        reason="active profile is unavailable in this session",
    )


@dataclass(frozen=True)
class StartPreference:
    """Where a session may start when the primary profile cannot run."""

    #: Preferred member when the primary cannot run; ``""`` disables starting a session on any other member.
    fallback_provider: str = ""
    #: The profile a resumed campaign was running; it continues there whenever that profile can run.
    resume_profile_id: str = ""


def select_start_profile(
    config: AgentRelayConfig,
    availability: AgentAvailabilitySnapshot,
    preference: StartPreference,
) -> str:
    """The profile a session starts on.

    A resumed campaign continues on the profile it was running. Otherwise the primary starts; if it cannot run, the
    fallback provider's member, then any other available member, unless fallback is disabled.
    """
    by_id = _profile_mapping(config.profiles)
    resumed = by_id.get(preference.resume_profile_id)
    if resumed is not None and _profile_is_available(resumed, availability):
        return resumed.profile_id
    primary = config.primary_profile
    if _profile_is_available(primary, availability):
        return primary.profile_id
    if not preference.fallback_provider:
        raise ValueError(
            f"{_unavailable(primary, availability)}; provider fallback is disabled, so the session does not start "
            "on another relay member"
        )
    others = [
        profile
        for profile in config.profiles
        if profile.profile_id != primary.profile_id and _profile_is_available(profile, availability)
    ]
    preferred = [profile for profile in others if profile.provider == preference.fallback_provider]
    if not others:
        raise ValueError(
            "no Agent profile can start this session: "
            + "; ".join(_unavailable(profile, availability) for profile in config.profiles)
        )
    return (preferred or others)[0].profile_id


def _unavailable(profile: AgentProfile, availability: AgentAvailabilitySnapshot) -> str:
    status = availability.profiles.get(profile.profile_id)
    return f"Agent profile {profile.profile_id} is unavailable: {status.reason if status else 'not probed'}"


def apply_profile_switch(
    state: AgentRelayState,
    *,
    intervention: RelayIntervention,
    iteration: int,
) -> AgentRelayState:
    """Persist a profile switch before the new profile starts an iteration."""
    if intervention.kind == INTERVENTION_SUPERVISOR:
        raise ValueError("supervisor intervention is not a profile switch")
    if iteration < 1:
        raise ValueError("relay switch iteration must be positive")
    previous = state.active_profile_id
    state.previous_profile_id = previous
    state.active_profile_id = intervention.target_profile_id
    state.active_no_improvement_streak = 0
    state.blockage_epoch += 1
    state.intervention_kind = intervention.kind
    state.switch_iteration = iteration
    state.switch_reason = intervention.reason
    return state


def apply_supervisor_fallback(
    state: AgentRelayState,
    *,
    intervention: RelayIntervention,
    iteration: int,
) -> AgentRelayState:
    """Open a fresh supervised search window without changing profiles."""
    if intervention.kind != INTERVENTION_SUPERVISOR:
        raise ValueError("supervisor fallback requires a supervisor intervention")
    if iteration < 1:
        raise ValueError("supervisor iteration must be positive")
    state.active_no_improvement_streak = 0
    state.blockage_epoch += 1
    state.intervention_kind = intervention.kind
    state.previous_profile_id = ""
    state.switch_iteration = iteration
    state.switch_reason = intervention.reason
    return state


def apply_profile_recheck(
    snapshot: AgentAvailabilitySnapshot,
    profile_id: str,
    status: ProfileAvailability,
) -> AgentAvailabilitySnapshot:
    """Replace one profile's availability with the result of re-running its startup check."""
    if profile_id not in snapshot.profiles:
        raise ValueError(f"unknown Agent profile: {profile_id}")
    snapshot.profiles[profile_id] = status
    snapshot.providers[status.provider] = ProviderAvailability(
        available=status.provider_available,
        reason="" if status.provider_available else status.reason,
    )
    return snapshot


def migrate_run_state_relay_fields(payload: object) -> dict[str, Any]:
    """Add relay defaults only to checkpoints written before relay existed."""
    if not isinstance(payload, dict):
        raise ValueError("run state must be a JSON object")
    migrated = dict(payload)
    keys = {"agent_relay", "agent_availability_snapshot"}
    present = keys & set(migrated)
    if not present:
        migrated["agent_relay"] = {
            "active_profile_id": "",
            "active_no_improvement_streak": 0,
            "blockage_epoch": 0,
            "intervention_kind": "",
            "previous_profile_id": "",
            "switch_iteration": 0,
            "switch_reason": "",
        }
        migrated["agent_availability_snapshot"] = {
            "checked_at": "",
            "session_index": 0,
            "providers": {},
            "profiles": {},
        }
    return migrated


def _validate_relay_membership(primary_profile_id: str, profiles: Sequence, blockage_threshold: int) -> None:
    if primary_profile_id not in _profile_mapping(profiles):
        raise ValueError("primary Agent profile is not configured")
    _require_positive_threshold(blockage_threshold)


def _require_positive_threshold(threshold: int) -> None:
    if threshold < 1:
        raise ValueError("relay blockage threshold must be positive")


def _strict_mapping(payload: object, model: type, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    expected = set(model.__dataclass_fields__)
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing:
        raise ValueError(f"{label} missing fields: " + ", ".join(sorted(missing)))
    if unknown:
        raise ValueError(f"{label} has unknown fields: " + ", ".join(sorted(unknown)))
    return dict(payload)


def _availability_entries(payload: object, model: type, label: str) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    entries = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{label} keys must be non-empty strings")
        entries[key] = model(**_strict_mapping(value, model, f"{label}.{key}"))
    return entries


def _profile_mapping(profiles: Sequence[_ProfileT]) -> Mapping[str, _ProfileT]:
    by_id = {profile.profile_id: profile for profile in profiles}
    if len(by_id) != len(profiles):
        raise ValueError("Agent profile ids must be unique")
    return by_id


def _profile_is_available(
    profile: AgentProfile,
    snapshot: AgentAvailabilitySnapshot,
) -> bool:
    entry = snapshot.profiles.get(profile.profile_id)
    return bool(entry and entry.available and entry.provider == profile.provider and entry.model == profile.model)


__all__ = [
    "AgentAvailabilitySnapshot",
    "AgentProfile",
    "AgentProfileSpec",
    "AgentRelayConfig",
    "AgentRelayPolicy",
    "AgentRelayState",
    "INTERVENTION_CROSS_HARNESS",
    "INTERVENTION_SAME_HARNESS",
    "INTERVENTION_SUPERVISOR",
    "ProfileAvailability",
    "ProviderAvailability",
    "RelayIntervention",
    "StartPreference",
    "apply_profile_recheck",
    "apply_profile_switch",
    "apply_relay_keep",
    "apply_relay_search_outcome",
    "apply_supervisor_fallback",
    "decision_counts_toward_blockage",
    "migrate_run_state_relay_fields",
    "relay_blockage_reached",
    "select_relay_intervention",
    "select_replacement_for_unavailable",
    "select_start_profile",
]
