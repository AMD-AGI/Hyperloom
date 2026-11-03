# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An explore is admitted on something to measure: its own grid, or a seed source that fills one."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.actions.executors.explore import runnable_grid_variants
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop._explore_grid import EXPLORE_GRID_MISSING, explore_grid_denial
from hyperloom.orchestrator.loop.proposals import PendingProposal
from hyperloom.orchestrator.roles import Backend, MockBackend, ScriptedPlan

# A framework the default grid has no branch for, so an explore here measures what it is sent and nothing else.
_SEEDLESS_FRAMEWORK = "sglang"
# The phase where the orchestrator may propose an explore.
_EXPLORE_PHASE = "FRAMEWORK_AGENT"
_GRID: list[dict[str, Any]] = [{"name": "v1", "extra_server_args": "--level 2", "extra_envs": {}}]


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _backends() -> dict[str, Backend]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {name: MockBackend(silent, name=name) for name in ("orchestration", "critic")}


@pytest.fixture
async def coord(session_dir) -> Coordinator:
    """A session in the explore phase whose framework has no cold-start seed grid."""
    c = Coordinator(session_dir, backends=_backends())
    # Past the baseline prerequisite so the sequence gate stays out of the way.
    c.shared_state.baseline_tput = 800.0
    c.shared_state.phase = _EXPLORE_PHASE
    c.shared_state.framework = _SEEDLESS_FRAMEWORK
    yield c
    await c.stop()


def _admission(c: Coordinator, params: dict[str, Any] | None):
    return c.dispatcher.admission_denial_for_action("explore", params=params)


async def _denials(c: Coordinator) -> list[str]:
    observed = await c.bus.tail(topic="observation")
    return [m.payload.get("rule") for m in observed if m.payload.get("kind") == "policy_denied"]


async def _skipped(c: Coordinator) -> list[str]:
    observed = await c.bus.tail(topic="observation")
    return [m.payload["reason"] for m in observed if m.payload.get("kind") == "proposal_materialize_skipped"]


def _approved_explore(msg_id: str, **params: Any) -> PendingProposal:
    payload = {"action_name": "explore", "params": dict(params)}
    return PendingProposal(msg_id, "orchestration", "explore", predicted_gain_pct=5.0, payload=payload)


class TestWhatAdmissionAsksOfAnExplore:
    """The grid question is answered where the proposal is admitted, not where the round reads the config."""

    def test_an_explore_carrying_a_grid_is_admitted(self, coord: Coordinator):
        assert _admission(coord, {"grid": list(_GRID)}) is None

    @pytest.mark.parametrize("params", [None, {}, {"grid": []}, {"grid": None}, {"grid": {"v1": {}}}])
    def test_an_explore_with_nothing_to_measure_is_refused(self, coord: Coordinator, params):
        """Every shape the round would reach ``_resolve_grid``'s ``empty_grid`` refusal with."""
        denied = _admission(coord, params)

        assert denied is not None and denied.rule == EXPLORE_GRID_MISSING

    @pytest.mark.parametrize(
        "rows",
        [["tp8", "tp4"], [{"extra_server_args": "--tp 8"}], [{"name": ""}]],
        ids=["strings", "nameless_dict", "blank_name"],
    )
    def test_a_grid_whose_rows_are_not_variants_is_refused(self, coord: Coordinator, rows):
        """The round builds variants from named dicts only; rows it would skip are nothing to measure."""
        denied = _admission(coord, {"grid": rows})

        assert denied is not None and denied.rule == EXPLORE_GRID_MISSING
        assert f"{len(rows)} row(s)" in str(denied)

    def test_the_refusal_says_what_to_send_instead(self, coord: Coordinator):
        denied = _admission(coord, {})

        assert _SEEDLESS_FRAMEWORK in str(denied)
        assert "params.grid" in (denied.hint or "")

    def test_no_other_action_is_judged_on_a_grid(self, coord: Coordinator):
        """The gate is chained for every action, and only explore measures a grid."""
        for action in ("profile", "report", "specialist"):
            denial = coord.dispatcher.admission_denial_for_action(action, params={})
            assert denial is None or denial.rule != EXPLORE_GRID_MISSING


class TestTheSeedSourcesThatStandInForAGrid:
    """``_resolve_grid`` fills an empty grid from the session; admission asks that same resolution."""

    def test_a_framework_with_a_default_grid_needs_no_grid_of_its_own(self, coord: Coordinator):
        coord.shared_state.framework = "atom"

        assert runnable_grid_variants({}, coord.shared_state)
        assert _admission(coord, {}) is None

    def test_the_framework_the_params_name_answers_before_the_session_s(self, coord: Coordinator):
        assert _admission(coord, {"framework": "atom"}) is None

    @pytest.mark.parametrize(
        "rows", [["tp8", "tp4"], [{"extra_server_args": "--tp 8"}]], ids=["strings", "nameless_dict"]
    )
    def test_a_sent_grid_is_never_replaced_by_the_default_one(self, coord: Coordinator, rows):
        """The round falls through to the seed grid only for an empty payload, atom or not."""
        coord.shared_state.framework = "atom"

        assert runnable_grid_variants({"grid": rows}, coord.shared_state) == []
        denied = _admission(coord, {"grid": rows})

        assert denied is not None and denied.rule == EXPLORE_GRID_MISSING

    def test_registered_framework_levers_are_a_grid_waiting_to_be_built(self, coord: Coordinator):
        """Each unattributed rewrite lever becomes a variant, so the round starts without a sent grid."""
        assert coord.shared_state.record_authored_framework_levers(
            [{"switch": "HL_HOIST", "category": "hoist_loop_invariant"}],
            default_on=False,
        )

        assert runnable_grid_variants({}, coord.shared_state)
        assert _admission(coord, {}) is None

    def test_attributed_levers_are_spent_and_seed_nothing(self, coord: Coordinator):
        coord.shared_state.record_authored_framework_levers(
            [{"switch": "HL_HOIST", "category": "hoist_loop_invariant"}],
            default_on=False,
        )
        rows = [dict(r, attributed_gain_pct=1.5) for r in coord.shared_state.authored_framework_levers]
        coord.shared_state.authored_framework_levers = rows

        assert runnable_grid_variants({}, coord.shared_state) == []
        assert _admission(coord, {}) is not None


class TestBothWaysAnExploreReachesTheRegistry:
    """Delegate and propose are admitted by the same gate, so neither queues a round that cannot start."""

    async def test_a_delegated_explore_with_no_grid_never_queues(self, coord: Coordinator):
        intent = Intent(type=IntentType.DELEGATE, payload={"action_name": "explore", "params": {}})

        await coord.router.handle_delegate("orchestration", intent)

        assert [t for t in await coord.tasks.queued() if t.kind == "explore"] == []
        assert await _denials(coord) == [EXPLORE_GRID_MISSING]

    async def test_a_proposed_explore_with_no_grid_never_reaches_the_critic(self, coord: Coordinator):
        intent = Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "explore", "params": {}, "predicted_gain_pct": 5.0},
        )

        await coord.router._handle_propose_action("orchestration", intent)

        assert not coord.state.pending_proposals
        assert await _denials(coord) == [EXPLORE_GRID_MISSING]

    async def test_a_delegated_explore_carrying_a_grid_still_queues(self, coord: Coordinator, monkeypatch):
        monkeypatch.setattr(coord.shared_state, "is_pruned", lambda a: False)
        intent = Intent(type=IntentType.DELEGATE, payload={"action_name": "explore", "params": {"grid": list(_GRID)}})

        await coord.router.handle_delegate("orchestration", intent)

        assert [t for t in await coord.tasks.queued() if t.kind == "explore"]


class TestTheApprovedProposalIsJudgedAgain:
    """Admission judged the grid the proposal carried; the Critic filter and the seed sources move under it."""

    async def test_an_approved_explore_that_names_no_grid_is_not_materialized(self, coord: Coordinator):
        await coord.proposals.materialize_approved_proposal(_approved_explore("p1"))

        assert [t for t in await coord.tasks.queued() if t.kind == "explore"] == []
        assert await _skipped(coord) == [EXPLORE_GRID_MISSING]

    async def test_an_approved_explore_a_seed_source_still_covers_is_materialized(self, coord: Coordinator):
        coord.shared_state.framework = "atom"

        await coord.proposals.materialize_approved_proposal(_approved_explore("p2"))

        assert await _skipped(coord) == []


def test_the_gate_answers_only_for_explore():
    """A unit check on the predicate itself: no session is consulted for another action."""
    assert explore_grid_denial("profile", {}, None) is None
    assert explore_grid_denial("explore", {"grid": list(_GRID)}, None) is None
    assert explore_grid_denial("explore", {}, None) is not None
