"""Tests for durable Agent relay state and intervention policy."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from kernelforge.loop.agent_relay import (
    AgentAvailabilitySnapshot,
    AgentProfile,
    AgentRelayConfig,
    AgentRelayState,
    INTERVENTION_CROSS_HARNESS,
    INTERVENTION_SAME_HARNESS,
    INTERVENTION_SUPERVISOR,
    ProfileAvailability,
    ProviderAvailability,
    apply_profile_switch,
    apply_relay_keep,
    apply_relay_search_outcome,
    apply_supervisor_fallback,
    invalidate_profile,
    migrate_run_state_relay_fields,
    relay_blockage_reached,
    select_relay_intervention,
    select_replacement_for_unavailable,
)
from kernelforge.loop.run_state import RunState
from kernelforge.loop.runner import IterationLoop
from kernelforge.loop.search_actors import SearchActorBundle, SearchActorPool


def _profiles() -> tuple[AgentProfile, ...]:
    return (
        AgentProfile("claude_opus", "claude", "claude-opus-5"),
        AgentProfile("codex_gpt", "codex", "gpt-6-astra"),
        AgentProfile("claude_sonnet", "claude", "claude-sonnet-5.5"),
    )


def _snapshot(*, codex: bool = True, sonnet: bool = True) -> AgentAvailabilitySnapshot:
    return AgentAvailabilitySnapshot(
        checked_at="2026-10-16T08:00:00Z",
        session_index=1,
        providers={
            "claude": ProviderAvailability(available=True),
            "codex": ProviderAvailability(available=codex),
        },
        profiles={
            "claude_opus": ProfileAvailability(
                provider="claude",
                model="claude-opus-5",
                provider_available=True,
                model_available=True,
                available=True,
            ),
            "codex_gpt": ProfileAvailability(
                provider="codex",
                model="gpt-6-astra",
                provider_available=codex,
                model_available=codex,
                available=codex,
            ),
            "claude_sonnet": ProfileAvailability(
                provider="claude",
                model="claude-sonnet-5.5",
                provider_available=True,
                model_available=sonnet,
                available=sonnet,
            ),
        },
    )


@pytest.mark.parametrize(
    "decision",
    [
        "REVERT_PERF",
        "NO_CHANGES",
        "BUILD_FAILED",
        "REVERT_VALIDATION",
        "CRASH",
    ],
)
def test_search_failures_advance_the_active_profile_streak(decision):
    state = AgentRelayState(active_profile_id="claude_opus")

    apply_relay_search_outcome(
        state,
        decision=decision,
        producer_profile_id="claude_opus",
    )

    assert state.active_no_improvement_streak == 1


@pytest.mark.parametrize(
    "decision",
    [
        "AGENT_ERROR",
        "API_ERROR",
        "ORCHESTRATION_ERROR",
        "REVERT_CONTENDED",
        "REVERT_INTEGRITY",
    ],
)
def test_infrastructure_failures_do_not_advance_the_streak(decision):
    state = AgentRelayState(
        active_profile_id="claude_opus",
        active_no_improvement_streak=2,
    )

    apply_relay_search_outcome(
        state,
        decision=decision,
        producer_profile_id="claude_opus",
    )

    assert state.active_no_improvement_streak == 2


def test_outcome_from_another_profile_does_not_change_the_active_streak():
    state = AgentRelayState(active_profile_id="claude_opus")

    apply_relay_search_outcome(
        state,
        decision="REVERT_PERF",
        producer_profile_id="codex_gpt",
    )

    assert state.active_no_improvement_streak == 0


def test_three_search_failures_reach_blockage_and_keep_resets_the_epoch():
    state = AgentRelayState(
        active_profile_id="claude_opus",
        profiles_tried_since_keep=["codex_gpt"],
        intervention_kind=INTERVENTION_CROSS_HARNESS,
    )
    for _ in range(3):
        apply_relay_search_outcome(
            state,
            decision="REVERT_PERF",
            producer_profile_id="claude_opus",
        )

    assert relay_blockage_reached(state, threshold=3)

    apply_relay_keep(state)

    assert state.active_no_improvement_streak == 0
    assert state.profiles_tried_since_keep == []
    assert state.intervention_kind == ""


def test_cross_harness_profile_has_priority():
    state = AgentRelayState(active_profile_id="claude_opus")

    intervention = select_relay_intervention(
        state,
        profiles=_profiles(),
        availability=_snapshot(),
    )

    assert intervention.kind == INTERVENTION_CROSS_HARNESS
    assert intervention.target_profile_id == "codex_gpt"


def test_same_harness_profile_is_used_when_cross_harness_is_unavailable():
    state = AgentRelayState(active_profile_id="claude_opus")

    intervention = select_relay_intervention(
        state,
        profiles=_profiles(),
        availability=_snapshot(codex=False),
    )

    assert intervention.kind == INTERVENTION_SAME_HARNESS
    assert intervention.target_profile_id == "claude_sonnet"


def test_supervisor_is_used_after_all_available_profiles_were_tried():
    state = AgentRelayState(
        active_profile_id="claude_opus",
        profiles_tried_since_keep=["codex_gpt", "claude_sonnet"],
    )

    intervention = select_relay_intervention(
        state,
        profiles=_profiles(),
        availability=_snapshot(),
    )

    assert intervention.kind == INTERVENTION_SUPERVISOR
    assert intervention.target_profile_id == ""


def test_profile_switch_is_persistable_before_the_target_runs():
    state = AgentRelayState(
        active_profile_id="claude_opus",
        active_no_improvement_streak=3,
    )
    intervention = select_relay_intervention(
        state,
        profiles=_profiles(),
        availability=_snapshot(),
    )

    apply_profile_switch(state, intervention=intervention, iteration=4)

    assert state.active_profile_id == "codex_gpt"
    assert state.previous_profile_id == "claude_opus"
    assert state.profiles_tried_since_keep == ["claude_opus"]
    assert state.active_no_improvement_streak == 0
    assert state.blockage_epoch == 1
    assert state.switch_iteration == 4


def test_supervisor_opens_a_new_window_without_switching_profiles():
    state = AgentRelayState(
        active_profile_id="claude_opus",
        active_no_improvement_streak=3,
    )
    intervention = select_relay_intervention(
        state,
        profiles=(_profiles()[0],),
        availability=_snapshot(),
    )

    apply_supervisor_fallback(state, intervention=intervention, iteration=4)

    assert state.active_profile_id == "claude_opus"
    assert state.active_no_improvement_streak == 0
    assert state.blockage_epoch == 1
    assert state.intervention_kind == INTERVENTION_SUPERVISOR


def test_runtime_invalidation_updates_only_the_saved_snapshot():
    snapshot = _snapshot()

    invalidate_profile(snapshot, "codex_gpt", "credential expired")

    assert snapshot.profiles["codex_gpt"].available is False
    assert snapshot.profiles["codex_gpt"].reason == "credential expired"
    assert snapshot.providers["codex"].available is True


def test_legacy_migration_adds_both_relay_fields_atomically():
    migrated = migrate_run_state_relay_fields({"campaign_id": "campaign"})

    assert migrated["agent_relay"]["active_profile_id"] == ""
    assert migrated["agent_availability_snapshot"]["profiles"] == {}


def test_partial_relay_checkpoint_is_not_migrated():
    payload = {
        "agent_relay": {
            "active_profile_id": "claude_opus",
        }
    }

    migrated = migrate_run_state_relay_fields(payload)

    assert "agent_availability_snapshot" not in migrated


class _StateStore:
    def __init__(self):
        self.events = []
        self.saved = []

    def append_event(self, event):
        self.events.append(event)

    def save(self, state):
        self.saved.append(state.to_dict())


def _loop_for_boundary(tmp_path, *, streak: int) -> IterationLoop:
    loop = object.__new__(IterationLoop)
    loop.ic = SimpleNamespace(workspace_dir=str(tmp_path))
    loop.run_state = RunState()
    loop.run_state.agent_relay = AgentRelayState(
        active_profile_id="claude_opus",
        active_no_improvement_streak=streak,
    )
    loop.run_state.agent_availability_snapshot = _snapshot()
    loop.state_store = _StateStore()
    loop._lane_queue = [object()]
    loop._last_lane_plans = ["old plan"]
    loop._working_tree_diff = lambda: ""
    loop.monitor = SimpleNamespace(no_improve_streak=streak)
    return loop


def _search_actor_pool() -> SearchActorPool:
    profiles = _profiles()
    return SearchActorPool(
        config=AgentRelayConfig(
            primary_profile_id="claude_opus",
            profiles=profiles,
            blockage_threshold=3,
        ),
        availability=_snapshot(),
        bundles={
            profile.profile_id: SearchActorBundle(
                profile=profile,
                agent_fn=lambda: None,
            )
            for profile in profiles
        },
    )


def test_iteration_boundary_switches_before_the_next_actor_runs(tmp_path):
    loop = _loop_for_boundary(tmp_path, streak=3)

    pool = _search_actor_pool()
    loop._search_actors = pool
    bundle = loop._prepare_search_actor(4, pool)
    force_supervisor, _reason = loop._supervisor_due(4)

    assert bundle.profile.profile_id == "codex_gpt"
    assert force_supervisor is False
    assert loop.run_state.agent_relay.active_profile_id == "codex_gpt"
    assert loop.run_state.search_mode == "EXPLOIT"
    assert loop._lane_queue == []
    assert loop._last_lane_plans == []
    assert loop.monitor.no_improve_streak == 0
    assert loop.state_store.saved
    assert loop.state_store.events[-1]["type"] == "relay_profile_switched"


def test_iteration_boundary_uses_supervisor_when_no_alternate_is_available(tmp_path):
    loop = _loop_for_boundary(tmp_path, streak=3)
    only_primary = _profiles()[0]
    pool = SearchActorPool(
        config=AgentRelayConfig(
            primary_profile_id=only_primary.profile_id,
            profiles=(only_primary,),
        ),
        availability=_snapshot(),
        bundles={
            only_primary.profile_id: SearchActorBundle(
                profile=only_primary,
                agent_fn=lambda: None,
            )
        },
    )

    loop._search_actors = pool
    bundle = loop._prepare_search_actor(4, pool)
    force_supervisor, _reason = loop._supervisor_due(4)

    assert bundle.profile.profile_id == "claude_opus"
    assert force_supervisor is True
    assert loop.run_state.agent_relay.intervention_kind == INTERVENTION_SUPERVISOR


def test_keep_after_a_switch_keeps_the_intervention_in_its_provenance(tmp_path):
    loop = _loop_for_boundary(tmp_path, streak=3)
    loop._prepare_search_actor(2, _search_actor_pool())
    session_sink: dict = {}
    loop._bind_session_producer(session_sink, loop.run_state.agent_relay.active_profile_id)

    apply_relay_keep(loop.run_state.agent_relay)
    result = SimpleNamespace(agent_profile_id="", blockage_epoch=0, agent_provider="", agent_model="")
    loop._stamp_result_profile(result, session_sink)
    provenance = loop._handoff_agent_provenance(session_sink)

    assert loop.run_state.agent_relay.intervention_kind == ""
    assert provenance == {
        "agent_profile_id": "codex_gpt",
        "agent_provider": "codex",
        "agent_model": "gpt-6-astra",
        "blockage_epoch": 1,
        "intervention_kind": INTERVENTION_CROSS_HARNESS,
    }
    assert (result.agent_profile_id, result.agent_model, result.blockage_epoch) == ("codex_gpt", "gpt-6-astra", 1)


def test_an_unavailable_active_profile_is_replaced_even_if_already_tried():
    state = AgentRelayState(active_profile_id="codex_gpt", profiles_tried_since_keep=["claude_opus"])

    replacement = select_replacement_for_unavailable(
        state,
        profiles=_profiles(),
        availability=_snapshot(codex=False),
    )

    assert replacement.kind == INTERVENTION_CROSS_HARNESS
    assert replacement.target_profile_id == "claude_opus"
    assert replacement.reason == "active profile is unavailable in this session"


def test_no_available_replacement_is_reported_as_none():
    snapshot = _snapshot(codex=False, sonnet=False)
    snapshot.profiles["claude_opus"].available = False

    assert (
        select_replacement_for_unavailable(
            AgentRelayState(active_profile_id="claude_opus"),
            profiles=_profiles(),
            availability=snapshot,
        )
        is None
    )


def test_a_resumed_session_moves_off_an_active_profile_it_cannot_run(tmp_path):
    loop = _loop_for_boundary(tmp_path, streak=0)
    loop.run_state.agent_relay.active_profile_id = "codex_gpt"
    loop.run_state.agent_availability_snapshot = _snapshot(codex=False)
    profiles = _profiles()
    pool = SearchActorPool(
        config=AgentRelayConfig(primary_profile_id="codex_gpt", profiles=profiles),
        availability=_snapshot(codex=False),
        bundles={
            profile.profile_id: SearchActorBundle(profile=profile, agent_fn=lambda: None)
            for profile in profiles
            if profile.profile_id != "codex_gpt"
        },
        start_profile_id="claude_opus",
    )

    loop._search_actors = pool
    bundle = loop._prepare_search_actor(3, pool)
    force_supervisor, _reason = loop._supervisor_due(3)

    assert bundle.profile.profile_id == "claude_opus"
    assert force_supervisor is False
    assert [event["type"] for event in loop.state_store.events] == [
        "relay_profile_unavailable",
        "relay_profile_switched",
    ]
    assert loop.state_store.events[-1]["reason"] == "active profile is unavailable in this session"


def test_a_runnable_profile_below_the_threshold_needs_no_intervention(tmp_path):
    loop = _loop_for_boundary(tmp_path, streak=2)

    pool = _search_actor_pool()
    loop._search_actors = pool
    bundle = loop._prepare_search_actor(3, pool)
    force_supervisor, _reason = loop._supervisor_due(3)

    assert bundle.profile.profile_id == "claude_opus"
    assert force_supervisor is False
    assert loop.state_store.events == []
