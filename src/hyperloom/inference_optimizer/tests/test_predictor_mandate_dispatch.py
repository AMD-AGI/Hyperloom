# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A first-pass mandate is offered until a specialist task exists for it, then never again.

Nothing used to mark a mandate as acted on. ``_resolve_first_pass_mandate``
substituted the text and returned, so a mandate orchestration had dispatched
exactly as instructed was still the newest unconsumed one on the next render and
was offered a second time -- while a later mandate in the same cycle could never
surface behind it.

These also cover the successful resolve path end to end through a real
Coordinator, which no test did: the only existing one asserted that a dispatch
*without* a mandate id is left alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.predictor import pump as pp
from hyperloom.orchestrator.state.shared_state import SharedState

MANDATE_ID = "primatune-patch-c0-s0-r1"
#: Verbatim from the Olmo-3-7B session of 2026-09-22, degeneration included.
#: The resolver has to hand this over untouched rather than tidy it.
MANDATE = (
    "Gate aiter qualify其一優於 SGLANG_USE_AITER environment Gate aiter qualify "
    "priority and Gate aiter qualify其一 in aiter.py."
)


def _round(round_id: str, mandate_id: str, mandate: str, *, cycle: int = 0) -> dict:
    """A predictor round as ``pump._record_round`` files it."""
    return {
        "round_id": round_id,
        "cycle": cycle,
        "domain": pp.QUEUE_DOMAIN,
        "priority": pp.QUEUE_PRIORITY,
        "task_id": round_id,
        "proposal_set": [],
        "mandate_id": mandate_id,
        "mandate": mandate,
    }


def _state_with(*rounds: dict) -> SharedState:
    state = SharedState()
    state.macro_cycle = 0
    for entry in rounds:
        state.record_specialist_round(entry)
    return state


def _entry(state: SharedState, mandate_id: str) -> dict:
    return next(r for r in state.specialist_rounds if r.get("mandate_id") == mandate_id)


class TestMarkMandateConsumed:
    def test_the_round_carrying_the_id_is_marked(self):
        state = _state_with(_round("c0-s0-r1", MANDATE_ID, MANDATE))
        assert pp.mark_mandate_consumed(state, MANDATE_ID, "task-1") is True
        entry = _entry(state, MANDATE_ID)
        assert entry["mandate_consumed_by"] == "task-1"
        assert entry["mandate_consumed_utc"]

    def test_the_first_consumer_is_kept(self):
        """A wave with one mandate id fans it into every sub-task."""
        state = _state_with(_round("c0-s0-r1", MANDATE_ID, MANDATE))
        pp.mark_mandate_consumed(state, MANDATE_ID, "task-1")
        first_utc = _entry(state, MANDATE_ID)["mandate_consumed_utc"]
        assert pp.mark_mandate_consumed(state, MANDATE_ID, "task-2") is True
        entry = _entry(state, MANDATE_ID)
        assert entry["mandate_consumed_by"] == "task-1"
        assert entry["mandate_consumed_utc"] == first_utc

    def test_an_unknown_or_blank_id_marks_nothing(self):
        state = _state_with(_round("c0-s0-r1", MANDATE_ID, MANDATE))
        assert pp.mark_mandate_consumed(state, "primatune-patch-nope", "task-1") is False
        assert pp.mark_mandate_consumed(state, "", "task-1") is False
        assert "mandate_consumed_by" not in _entry(state, MANDATE_ID)

    def test_a_consumed_mandate_still_resolves(self):
        """An idempotent re-dispatch of the same task still needs the text."""
        state = _state_with(_round("c0-s0-r1", MANDATE_ID, MANDATE))
        pp.mark_mandate_consumed(state, MANDATE_ID, "task-1")
        assert pp.find_mandate(state, MANDATE_ID) == MANDATE


class TestRenderSkipsConsumed:
    def test_a_consumed_mandate_is_not_offered(self):
        state = _state_with(_round("c0-s0-r1", MANDATE_ID, MANDATE))
        assert MANDATE_ID in state.to_untested_proposals_summary()
        pp.mark_mandate_consumed(state, MANDATE_ID, "task-1")
        assert state._untested_patch_mandate() == {}
        assert MANDATE_ID not in state.to_untested_proposals_summary()

    def test_the_next_unconsumed_mandate_in_the_cycle_takes_the_slot(self):
        older = "primatune-patch-c0-s0-r0"
        state = _state_with(
            _round("c0-s0-r0", older, "Bypass the tokenizer lock."),
            _round("c0-s0-r1", MANDATE_ID, MANDATE),
        )
        assert state._untested_patch_mandate()["mandate_id"] == MANDATE_ID
        pp.mark_mandate_consumed(state, MANDATE_ID, "task-1")
        assert state._untested_patch_mandate()["mandate_id"] == older

    def test_a_consumed_mandate_does_not_unlock_an_earlier_cycle(self):
        state = _state_with(
            _round("c0-s0-r0", "primatune-patch-c0-s0-r0", "stale idea", cycle=0),
            _round("c1-s0-r0", "primatune-patch-c1-s0-r0", "fresh idea", cycle=1),
        )
        state.macro_cycle = 1
        pp.mark_mandate_consumed(state, "primatune-patch-c1-s0-r0", "task-1")
        assert state._untested_patch_mandate() == {}


def _build_coord(tmp_path: Path, *rounds: dict) -> Coordinator:
    from hyperloom.orchestrator.roles.agent_role import default_role_registry
    from hyperloom.orchestrator.roles.mock_backend import MockBackend, MockTurn, ScriptedPlan

    state = SharedState(session_id="mandate-dispatch")
    state.save(tmp_path)
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {name: MockBackend(idle) for name in ("orchestration", "critic", "robustness")}
    coord = Coordinator(
        session_dir=tmp_path,
        backends=backends,
        role_registry=default_role_registry(),
        recipe_kb=None,
        knowledge_plane=None,
    )
    coord.shared_state.macro_cycle = 0
    for entry in rounds:
        coord.shared_state.record_specialist_round(entry)
    return coord


def _mandate_delegate() -> Intent:
    """The dispatch the render tells orchestration to emit, placeholder and all."""
    return Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "specialist",
            "params": {
                "scope": "freeform",
                "mode": "patch",
                "primatune_mandate_id": MANDATE_ID,
                "task_description": "act on the first-pass mandate",
            },
        },
    )


class TestDispatchConsumesTheMandate:
    @pytest.mark.asyncio
    async def test_dispatch_resolves_the_mandate_and_marks_it(self, tmp_path):
        coord = _build_coord(tmp_path, _round("c0-s0-r1", MANDATE_ID, MANDATE))
        await coord._handle_delegate("orchestration", _mandate_delegate())

        queued = [t for t in await coord.tasks.queued() if t.kind == "specialist"]
        assert len(queued) == 1
        task = queued[0]
        assert task.params["task_description"] == MANDATE
        assert task.params["mode"] == "patch"
        assert task.params["scope"] == "freeform"
        assert task.params["provenance"] == pp.PROVENANCE
        assert task.params["lever_kind"] == "source_patch"
        assert "domain" not in task.params

        entry = _entry(coord.shared_state, MANDATE_ID)
        assert entry["mandate_consumed_by"] == task.task_id
        assert MANDATE_ID not in coord.shared_state.to_untested_proposals_summary()

    @pytest.mark.asyncio
    async def test_a_failed_create_leaves_the_mandate_on_offer(self, tmp_path, monkeypatch):
        coord = _build_coord(tmp_path, _round("c0-s0-r1", MANDATE_ID, MANDATE))

        async def _refuse(**_kwargs):
            raise RuntimeError("task registry unavailable")

        monkeypatch.setattr(coord.tasks, "create_or_return_existing", _refuse)
        with pytest.raises(RuntimeError):
            await coord._handle_delegate("orchestration", _mandate_delegate())

        assert "mandate_consumed_by" not in _entry(coord.shared_state, MANDATE_ID)
        assert MANDATE_ID in coord.shared_state.to_untested_proposals_summary()

    @pytest.mark.asyncio
    async def test_a_dispatch_without_the_id_consumes_nothing(self, tmp_path):
        """An LLM that re-authors the mandate as its own prose gets no credit for it."""
        coord = _build_coord(tmp_path, _round("c0-s0-r1", MANDATE_ID, MANDATE))
        await coord._handle_delegate(
            "orchestration",
            Intent(
                type=IntentType.DELEGATE,
                payload={
                    "action_name": "specialist",
                    "params": {"scope": "freeform", "task_description": "Trace the AITER gate."},
                },
            ),
        )
        assert "mandate_consumed_by" not in _entry(coord.shared_state, MANDATE_ID)
        assert MANDATE_ID in coord.shared_state.to_untested_proposals_summary()
