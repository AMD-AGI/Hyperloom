# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the predictor's source-change channel: offered, dispatched verbatim, then withdrawn."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.predictor.mandate import mark_mandate_consumed
from hyperloom.orchestrator.predictor.prompt import queue_block
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
    summary = queue_block(_state_with_mandate())
    assert f"primatune_mandate_id:'{MANDATE_ID}'" in summary
    assert "Fuse rmsnorm into the QKV GEMM" in summary


def test_a_mandate_from_an_earlier_cycle_is_not_offered():
    state = _state_with_mandate(cycle=0)
    state.macro_cycle = 1
    assert queue_block(state) == ""


def test_the_router_dispatches_the_predictors_own_text_and_attribution():
    from hyperloom.orchestrator.loop.intent_router import IntentRouter

    state = _state_with_mandate()
    router = IntentRouter(SimpleNamespace(shared_state=state))
    params = {"primatune_mandate_id": MANDATE_ID, "task_description": "x", "domain": "serving", "tags": ["a"]}
    assert router._resolve_primatune_mandate(params) is None

    assert params["task_description"] == "Fuse rmsnorm into the QKV GEMM in tuned_gemm.py"
    assert (params["scope"], params["mode"], params["provenance"], params["lever_kind"]) == (
        "freeform",
        "patch",
        "primatune",
        "source_patch",
    )
    assert "domain" not in params and "tags" not in params

    unknown = {"primatune_mandate_id": "nope", "task_description": "x"}
    refused = router._resolve_primatune_mandate(unknown)
    assert refused is not None and refused.rule == "unknown_primatune_mandate"
    assert unknown == {"primatune_mandate_id": "nope", "task_description": "x"}


@pytest.fixture
async def coordinator(tmp_path, monkeypatch):
    """A real Coordinator holding the mandate, with a framework checkout for patch specialists."""
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
        # Mandates are offered in FRAMEWORK_AGENT, where a specialist delegate is admitted.
        coord.shared_state.phase = "FRAMEWORK_AGENT"
        yield coord
    finally:
        await coord.stop()


async def _delegate_specialist(coord: Any, params: dict[str, Any], key: str) -> None:
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    payload = {"action_name": "specialist", "idempotency_key": key, "params": params}
    await coord.router.handle_delegate("orchestration", Intent(type=IntentType.DELEGATE, payload=payload))


async def test_a_mandate_delegate_queues_the_patch_specialist_and_withdraws_the_mandate(coordinator):
    params = {"primatune_mandate_id": MANDATE_ID, "task_description": "x", "domain": "serving_specialist"}
    await _delegate_specialist(coordinator, params, "m1")

    (task,) = await coordinator.tasks.by_state("queued")
    assert task.params["task_description"] == "Fuse rmsnorm into the QKV GEMM in tuned_gemm.py"
    assert (task.params["provenance"], task.params["scope"], task.params["mode"]) == ("primatune", "freeform", "patch")
    assert coordinator.shared_state.specialist_rounds[0]["mandate_consumed_by"] == task.task_id
    assert queue_block(coordinator.shared_state) == ""


async def test_a_delegate_citing_no_known_mandate_is_refused_rather_than_run_on_its_placeholder(coordinator):
    params = {"primatune_mandate_id": "primatune-patch-c9-s9-r9", "task_description": "<one line>"}
    await _delegate_specialist(coordinator, params, "m2")

    assert await coordinator.tasks.by_state("queued") == []
    assert coordinator.shared_state.policy_denial_streak == {"specialist:unknown_primatune_mandate": 1}
    assert "mandate_consumed_by" not in coordinator.shared_state.specialist_rounds[0]


def test_a_consumed_mandate_is_withdrawn_and_keeps_its_first_consumer():
    state = _state_with_mandate()
    assert mark_mandate_consumed(state, MANDATE_ID, "task-1")
    assert mark_mandate_consumed(state, MANDATE_ID, "task-2")
    assert state.specialist_rounds[0]["mandate_consumed_by"] == "task-1"
    assert queue_block(state) == ""
    assert not mark_mandate_consumed(state, "nope", "task-3")


def test_the_patch_keeps_the_predictors_provenance_on_its_way_to_integrate_patch():
    from hyperloom.orchestrator.phases.framework import _forward_integrate_source

    dst: dict[str, Any] = {}
    _forward_integrate_source({"provenance": "primatune", "lever_kind": "source_patch"}, dst, {})
    assert (dst["provenance"], dst["lever_kind"]) == ("primatune", "source_patch")
    dst = {}
    _forward_integrate_source({"provenance": "primatune", "domain": "serving"}, dst, {})
    assert dst["provenance"] == "specialist:serving"
