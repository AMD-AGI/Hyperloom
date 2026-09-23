# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""CLOSE measures a working stack that changed after its last validation before publishing it."""

from __future__ import annotations

import asyncio

import pytest

from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult


@pytest.fixture
def coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    from hyperloom.inference_optimizer.session.paths import make_session_dir
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import MockBackend, MockCriticBackend, ScriptedPlan

    backends = {
        "orchestration": MockBackend(ScriptedPlan(turns=[]), name="orchestration"),
        "critic": MockCriticBackend(),
    }
    c = Coordinator(make_session_dir(), backends=backends)
    state = c.shared_state
    state.baseline_tput = 1000.0
    state.current_best = {"action": "baseline", "tput": 1000.0, "extra_server_args": "", "extra_envs": {}}
    assert c._lift_to_current_best(
        "explore",
        1100.0,
        {"name": "page16", "extra_server_args": "--page-size 16", "candidate_extra_server_args": "--page-size 16"},
    )
    assert state.optimization_stack_has_unvalidated_keeps()
    state.phase_history = [{"to_phase": "CLOSE"}]
    return c


def _steps(c) -> list[dict]:
    rows = (c.shared_state.phase_history or [{}])[-1].get("evidence", {}).get("close_steps", [])
    return [row for row in rows if row.get("step") == "stack_revalidation"]


def _run_rebench(c, *, measured: float | None, calls: list):
    async def _run(task, *, on_complete=None, **_kwargs):
        calls.append(task)
        await c.tasks.transition(task.task_id, "running")
        state = "succeeded" if measured else "failed"
        await c.tasks.transition(task.task_id, state)
        result = SubAgentResult(
            task_id=task.task_id,
            state=state,
            result={"status": state, "output_throughput": measured} if measured else {"status": "failed"},
        )
        assert on_complete is not None, "the rebench must promote through the dispatcher's reap"
        await on_complete(result)
        return result

    return _run


@pytest.mark.asyncio
async def test_a_validated_stack_is_not_rebenched(coordinator) -> None:
    c = coordinator
    assert c._update_cumulative_gain_validated(1100.0, {"output_throughput": 1100.0})
    calls: list = []
    c.run_task_registered = _run_rebench(c, measured=1100.0, calls=calls)

    await c._revalidate_stack_for_close()

    assert calls == []


@pytest.mark.asyncio
async def test_a_successful_rebench_validates_the_stack_close_then_publishes(coordinator) -> None:
    c = coordinator
    c.shared_state.stop_reason = "global_converged"
    calls: list = []
    c.run_task_registered = _run_rebench(c, measured=1120.0, calls=calls)

    await c._revalidate_stack_for_close()

    [task] = calls
    assert task.idempotency_key == "close-stack-revalidate-g1"
    assert task.params["source"] == "resume_stack_revalidate"
    assert "geak_fallback" not in task.params
    state = c.shared_state
    assert not state.optimization_stack_has_unvalidated_keeps()
    assert state.cumulative_gain_validated == pytest.approx(12.0)
    assert state.optimization_stack[-1]["variant_name"] == "page16" and len(state.optimization_stack) == 1
    assert c.finalize_recipe_and_journal()["reason"] != "unvalidated_recipe_stack"


@pytest.mark.asyncio
async def test_a_failed_rebench_leaves_close_publishing_nothing(coordinator) -> None:
    c = coordinator
    calls: list = []
    c.run_task_registered = _run_rebench(c, measured=None, calls=calls)

    await c._revalidate_stack_for_close()

    assert len(calls) == 1
    assert c.shared_state.optimization_stack_has_unvalidated_keeps()
    assert _steps(c)[-1]["status"] == "failed"
    assert c.finalize_recipe_and_journal()["reason"] == "unvalidated_recipe_stack"


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["signal", "coordinator_exception"])
async def test_an_interrupted_run_is_not_rebenched(coordinator, stop_reason) -> None:
    c = coordinator
    c.shared_state.stop_reason = stop_reason
    calls: list = []
    c.run_task_registered = _run_rebench(c, measured=1100.0, calls=calls)

    await c._revalidate_stack_for_close()

    assert calls == []
    assert _steps(c)[-1]["status"] == "skipped"


@pytest.mark.asyncio
async def test_a_budget_that_cannot_fit_one_measurement_is_not_rebenched(coordinator) -> None:
    c = coordinator
    c.shared_state.baseline_runtime_sec = 900.0
    c.shared_state.session_budget_usable_sec = lambda **_kwargs: 60.0
    calls: list = []
    c.run_task_registered = _run_rebench(c, measured=1100.0, calls=calls)

    await c._revalidate_stack_for_close()

    assert calls == []
    assert _steps(c)[-1]["detail"].startswith("session_budget")


@pytest.mark.asyncio
async def test_busy_lanes_cancel_the_rebench_instead_of_leaving_it_queued(coordinator) -> None:
    c = coordinator
    seen: list = []

    async def _lanes_busy(task, **_kwargs):
        seen.append(task)
        return None

    c.run_task_registered = _lanes_busy

    await c._revalidate_stack_for_close()

    [task] = seen
    assert (await c.tasks.get(task.task_id)).state == "cancelled"
    assert _steps(c)[-1]["detail"] == "lanes_busy"


@pytest.mark.asyncio
async def test_a_rebench_that_outlives_its_bound_is_abandoned(coordinator) -> None:
    c = coordinator
    c.CLOSE_STACK_REVALIDATION_TIMEOUT_SEC = 0.05
    seen: list = []

    async def _hang(task, **_kwargs):
        seen.append(task)
        await c.tasks.transition(task.task_id, "running")
        await asyncio.sleep(5)

    c.run_task_registered = _hang

    await c._revalidate_stack_for_close()

    [task] = seen
    assert (await c.tasks.get(task.task_id)).state == "failed"
    assert _steps(c)[-1]["detail"] == "timeout"
    assert c.shared_state.optimization_stack_has_unvalidated_keeps()
