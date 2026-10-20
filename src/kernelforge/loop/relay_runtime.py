# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent relay coordination for the iteration loop: which search actor runs, and who produced what."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict

from kernelforge.loop.agent_relay import (
    INTERVENTION_SUPERVISOR,
    RelayIntervention,
    apply_profile_recheck,
    apply_profile_switch,
    apply_relay_keep,
    apply_relay_search_outcome,
    apply_supervisor_fallback,
    relay_blockage_reached,
    select_relay_intervention,
    select_replacement_for_unavailable,
)
from kernelforge.loop.run_state import CriticRuling, make_event
from kernelforge.loop.search_actors import SearchActorBundle, SearchActorPool
from kernelforge.loop.search_policy import (
    OBJECTIVE_DISCOVER_NEW_MECHANISM,
    SEARCH_MODE_DIVERSIFY,
    SearchPolicyDecision,
)


class RelayRuntimeMixin:
    """Own the active Agent profile: start-up, boundary switches, Supervisor fallback, and producer provenance."""

    #: Every relay profile's search actor for this session; ``None`` when the campaign has no relay.
    _search_actors: SearchActorPool | None = None
    #: Profiles whose API retries ran out this session, checked again at the next iteration boundary.
    _profiles_to_recheck: frozenset[str] = frozenset()

    def _supervisor_banner(self) -> str:
        """How the Supervisor is scheduled this session, in the operator's terms."""
        if self._search_actors is not None:
            return "  Supervisor: enabled at a relay blockage when no other Agent profile is available"
        return (
            f"  Supervisor: enabled (after {self.ic.supervise_after} stalls, "
            f"cooldown {self.ic.supervise_cooldown}, no intervention cap)"
        )

    def _session_start_bundle(self) -> SearchActorBundle | None:
        """The actor recovery initializes with: the persisted active profile when it can run, else the start one."""
        actors = self._search_actors
        if actors is None:
            return None
        active = self.run_state.agent_relay.active_profile_id or actors.start_profile_id
        return actors.bundle(active if actors.has_bundle(active) else actors.start_profile_id)

    def _record_agent_availability(self) -> None:
        """Persist the process-start snapshot after the session index advances."""
        actors = self._search_actors
        if actors is None:
            return
        snapshot = copy.deepcopy(actors.availability)
        snapshot.session_index = self.run_state.session_index
        self.run_state.agent_availability_snapshot = snapshot
        if not self.run_state.agent_relay.active_profile_id:
            self.run_state.agent_relay.active_profile_id = actors.start_profile_id
        self.state_store.append_event(
            make_event(
                "agent_availability_snapshot",
                self.run_state.iteration,
                snapshot=asdict(snapshot),
                start_profile_id=actors.start_profile_id,
            )
        )

    def _iteration_bundle(self, iteration: int) -> SearchActorBundle | None:
        """The actor this iteration runs on, switched first if the boundary calls for it."""
        if self._search_actors is None:
            return None
        return self._prepare_search_actor(iteration, self._search_actors)

    def _supervisor_due(self, iteration: int) -> tuple[bool, str]:
        """Whether this iteration consults the Supervisor, and why."""
        if self._search_actors is None:
            return self.monitor.should_intervene(iteration)
        relay = self.run_state.agent_relay
        # Under relay the Supervisor is the last intervention of a blockage epoch, chosen at the boundary.
        if relay.intervention_kind == INTERVENTION_SUPERVISOR and relay.switch_iteration == iteration:
            return True, relay.switch_reason
        return False, ""

    def _relay_search_policy(self, iteration: int, decision: SearchPolicyDecision) -> SearchPolicyDecision:
        """The search policy for this iteration: a new profile starts by diversifying away from the blocked line."""
        relay = self.run_state.agent_relay
        if relay.switch_iteration != iteration or relay.intervention_kind in {"", INTERVENTION_SUPERVISOR}:
            return decision
        return SearchPolicyDecision(
            mode=SEARCH_MODE_DIVERSIFY,
            reason_codes=("AGENT_RELAY_BLOCKAGE",),
            objective_kind=OBJECTIVE_DISCOVER_NEW_MECHANISM,
            residence_iterations_remaining=0,
        )

    def _lane_plans_match_active_profile(self, planned_iteration: int) -> bool:
        """Whether a recoverable lane plan was written by the profile now active, so it may still be dispatched."""
        try:
            manifest = json.loads(self._lane_plan_manifest_path(planned_iteration).read_text(encoding="utf-8"))
            planned_profile_id = str(manifest.get("agent_profile_id") or "")
        except (OSError, ValueError, TypeError):
            return False
        active_profile_id = self.run_state.agent_relay.active_profile_id
        if active_profile_id and planned_profile_id != active_profile_id:
            print(
                f"  [lanes] iteration {planned_iteration} belongs to "
                f"{planned_profile_id or 'a legacy profile'}, not active "
                f"profile {active_profile_id}; planning this round afresh"
            )
            return False
        return True

    def _select_boundary_intervention(
        self,
        search_actors: SearchActorPool,
    ) -> tuple[RelayIntervention | None, str]:
        """The intervention this boundary needs, if any, and the event type that records why."""
        relay = self.run_state.agent_relay
        availability = self.run_state.agent_availability_snapshot
        status = availability.profiles.get(relay.active_profile_id)
        runnable = search_actors.has_bundle(relay.active_profile_id) and bool(status and status.available)
        if not runnable:
            replacement = select_replacement_for_unavailable(
                relay,
                profiles=search_actors.profiles,
                availability=availability,
            )
            if replacement is not None:
                return replacement, "relay_profile_unavailable"
            if not search_actors.has_bundle(relay.active_profile_id):
                raise RuntimeError(f"no Agent profile can run: {relay.active_profile_id} has no search actor")
        if not relay_blockage_reached(relay, threshold=search_actors.blockage_threshold):
            return None, ""
        intervention = select_relay_intervention(relay, profiles=search_actors.profiles, availability=availability)
        return intervention, "relay_blockage_detected"

    def _prepare_search_actor(self, iteration: int, search_actors: SearchActorPool) -> SearchActorBundle:
        """Select and durably switch the complete search actor at a clean boundary."""
        relay = self.run_state.agent_relay
        if not relay.active_profile_id:
            relay.active_profile_id = search_actors.start_profile_id
            self.state_store.save(self.run_state)
        self._recheck_profiles(iteration, search_actors)
        intervention, event_type = self._select_boundary_intervention(search_actors)
        if intervention is None:
            return search_actors.bundle(relay.active_profile_id)
        self.state_store.append_event(
            make_event(
                event_type,
                iteration,
                active_profile_id=relay.active_profile_id,
                streak=relay.active_no_improvement_streak,
                selected_intervention=intervention.kind,
                target_profile_id=intervention.target_profile_id or None,
                reason=intervention.reason,
            )
        )
        if intervention.kind == INTERVENTION_SUPERVISOR:
            apply_supervisor_fallback(relay, intervention=intervention, iteration=iteration)
            self.state_store.save(self.run_state)
            return search_actors.bundle(relay.active_profile_id)
        self._switch_search_actor(iteration, intervention)
        return search_actors.bundle(relay.active_profile_id)

    def _recheck_profiles(self, iteration: int, search_actors: SearchActorPool) -> None:
        """Re-run the startup check of every profile whose API retries ran out, before anything is selected."""
        pending, self._profiles_to_recheck = self._profiles_to_recheck, frozenset()
        if search_actors.recheck is None:
            return
        by_id = {profile.profile_id: profile for profile in search_actors.profiles}
        snapshot = self.run_state.agent_availability_snapshot
        for profile_id in sorted(pending & set(by_id)):
            status = search_actors.recheck(by_id[profile_id])
            apply_profile_recheck(snapshot, profile_id, status)
            self.state_store.append_event(
                make_event(
                    "relay_profile_rechecked",
                    iteration,
                    agent_profile_id=profile_id,
                    available=status.available,
                    reason=status.reason,
                )
            )
        if pending:
            self.state_store.save(self.run_state)

    def _switch_search_actor(self, iteration: int, intervention: RelayIntervention) -> None:
        """Persist the switch before the new profile runs, dropping what the previous profile planned."""
        if self._pending_keep_path.exists():
            raise RuntimeError("cannot switch Agent profile while pending KEEP exists")
        if self._working_tree_diff().strip():
            raise RuntimeError("cannot switch Agent profile with a tracked candidate diff")
        relay = self.run_state.agent_relay
        apply_profile_switch(relay, intervention=intervention, iteration=iteration)
        self.run_state.last_critic = CriticRuling()
        self._lane_queue = []
        self._persist_lane_queue()
        self._last_lane_plans = []
        self.state_store.save(self.run_state)
        self.state_store.append_event(
            make_event(
                "relay_profile_switched",
                iteration,
                previous_profile_id=relay.previous_profile_id,
                active_profile_id=relay.active_profile_id,
                intervention_kind=relay.intervention_kind,
                blockage_epoch=relay.blockage_epoch,
                reason=relay.switch_reason,
            )
        )
        if self.monitor is not None:
            self.monitor.no_improve_streak = 0

    def _apply_relay_outcome(self, result, decision_label: str) -> None:
        """Reduce attributed search progress, and schedule a recheck of a profile whose API retries ran out."""
        relay = self.run_state.agent_relay
        if result.kept:
            apply_relay_keep(relay)
        else:
            apply_relay_search_outcome(relay, decision=decision_label, producer_profile_id=result.agent_profile_id)
        if decision_label == "API_ERROR" and result.agent_profile_id:
            self._profiles_to_recheck = self._profiles_to_recheck | {result.agent_profile_id}

    def _bind_session_producer(self, session_sink: dict, profile_id: str) -> None:
        """Record the producer and the relay window it ran in, before the outcome can reset that window."""
        relay = self.run_state.agent_relay
        session_sink["agent_profile_id"] = profile_id
        session_sink["blockage_epoch"] = relay.blockage_epoch
        session_sink["intervention_kind"] = relay.intervention_kind

    def _stamp_result_profile(self, result, session_sink: dict) -> None:
        """Attach immutable producer provenance before the outcome is persisted."""
        result.agent_profile_id = str(session_sink.get("agent_profile_id") or "")
        result.blockage_epoch = int(session_sink.get("blockage_epoch") or 0)
        profile = self.run_state.agent_availability_snapshot.profiles.get(result.agent_profile_id)
        if profile is not None:
            result.agent_provider = profile.provider
            result.agent_model = profile.model

    def _handoff_agent_provenance(self, session_sink: dict) -> dict:
        """Resolve producer identity once for the immutable handoff."""
        profile_id = str(session_sink.get("agent_profile_id") or "")
        profile = self.run_state.agent_availability_snapshot.profiles.get(profile_id)
        return {
            "agent_profile_id": profile_id,
            "agent_provider": profile.provider if profile is not None else "",
            "agent_model": profile.model if profile is not None else "",
            "blockage_epoch": int(session_sink.get("blockage_epoch") or 0),
            "intervention_kind": str(session_sink.get("intervention_kind") or ""),
        }


__all__ = ["RelayRuntimeMixin"]
