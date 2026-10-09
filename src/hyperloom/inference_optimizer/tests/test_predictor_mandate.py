# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the predictor's source-change channel: offered, dispatched verbatim, then withdrawn."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from hyperloom.orchestrator.predictor.mandate import mark_mandate_consumed
from hyperloom.orchestrator.state.shared_state import SharedState

MANDATE_ID = "primatune-patch-c0-s0-r0"


def _state_with_mandate(*, cycle: int = 0) -> SharedState:
    state = SharedState()
    state.record_specialist_round(
        {
            "round_id": "c0-s0-r0",
            "domain": "primatune",
            "priority": 1,
            "cycle": cycle,
            "proposal_set": [],
            "mandate_id": MANDATE_ID,
            "mandate": "Fuse rmsnorm into the QKV GEMM in tuned_gemm.py",
        }
    )
    return state


def test_the_queue_block_offers_the_mandate_with_its_dispatch():
    summary = _state_with_mandate().to_untested_proposals_summary()
    assert f"primatune_mandate_id:'{MANDATE_ID}'" in summary
    assert "Fuse rmsnorm into the QKV GEMM" in summary


def test_a_mandate_from_an_earlier_cycle_is_not_offered():
    state = _state_with_mandate(cycle=0)
    state.macro_cycle = 1
    assert state.to_untested_proposals_summary() == ""


def test_the_router_dispatches_the_predictors_own_text_and_attribution():
    from hyperloom.orchestrator.loop.intent_router import IntentRouter

    state = _state_with_mandate()
    router = IntentRouter(SimpleNamespace(shared_state=state))
    params = {"primatune_mandate_id": MANDATE_ID, "task_description": "x", "domain": "serving", "tags": ["a"]}
    router._resolve_primatune_mandate(params)

    assert params["task_description"] == "Fuse rmsnorm into the QKV GEMM in tuned_gemm.py"
    assert (params["scope"], params["mode"], params["provenance"], params["lever_kind"]) == (
        "freeform",
        "patch",
        "primatune",
        "source_patch",
    )
    assert "domain" not in params and "tags" not in params

    unknown = {"primatune_mandate_id": "nope", "task_description": "x"}
    router._resolve_primatune_mandate(unknown)
    assert unknown == {"primatune_mandate_id": "nope", "task_description": "x"}


async def test_a_mandate_delegate_queues_the_patch_specialist_and_withdraws_the_mandate(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
    from hyperloom.inference_optimizer.session.paths import make_session_dir
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    silent = ScriptedPlan(turns=[])
    coord = Coordinator(
        make_session_dir(),
        backends={"orchestration": MockBackend(silent, name="o"), "critic": MockBackend(silent, name="c")},
    )
    try:
        (tmp_path / "fw").mkdir()
        coord.shared_state.framework, coord.shared_state.framework_repo_path = "sglang", str(tmp_path / "fw")
        coord.shared_state.specialist_rounds = _state_with_mandate().specialist_rounds
        params = {"primatune_mandate_id": MANDATE_ID, "task_description": "x", "domain": "serving_specialist"}
        intent = Intent(
            type=IntentType.DELEGATE, payload={"action_name": "specialist", "idempotency_key": "m1", "params": params}
        )
        await coord.router.handle_delegate("orchestration", intent)

        (task,) = await coord.tasks.by_state("queued")
        assert task.params["task_description"] == "Fuse rmsnorm into the QKV GEMM in tuned_gemm.py"
        assert (task.params["provenance"], task.params["scope"], task.params["mode"]) == (
            "primatune",
            "freeform",
            "patch",
        )
        assert coord.shared_state.specialist_rounds[0]["mandate_consumed_by"] == task.task_id
        assert coord.shared_state.to_untested_proposals_summary() == ""
    finally:
        await coord.stop()


def test_a_consumed_mandate_is_withdrawn_and_keeps_its_first_consumer():
    state = _state_with_mandate()
    assert mark_mandate_consumed(state, MANDATE_ID, "task-1")
    assert mark_mandate_consumed(state, MANDATE_ID, "task-2")
    assert state.specialist_rounds[0]["mandate_consumed_by"] == "task-1"
    assert state.to_untested_proposals_summary() == ""
    assert not mark_mandate_consumed(state, "nope", "task-3")


def test_the_patch_keeps_the_predictors_provenance_on_its_way_to_integrate_patch():
    from hyperloom.orchestrator.phases.framework import _forward_integrate_source

    dst: dict[str, Any] = {}
    _forward_integrate_source({"provenance": "primatune", "lever_kind": "source_patch"}, dst, {})
    assert (dst["provenance"], dst["lever_kind"]) == ("primatune", "source_patch")
    dst = {}
    _forward_integrate_source({"provenance": "primatune", "domain": "serving"}, dst, {})
    assert dst["provenance"] == "specialist:serving"
