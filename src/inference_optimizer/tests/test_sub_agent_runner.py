"""Tests for orchestrator/sub_agent_runner.py — F3b."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from inference_optimizer.paths import asset_actions_dir
from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.backends import MockBackend, ScriptStep
from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.feature_flags import build_feature_flags
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.policy import PolicyGate
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
)
from inference_optimizer.orchestrator.sub_agent_runner import (
    SubAgentRunner,
    TaskResult,
    dispatch_pending_delegates,
)
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.storage.connection import SqliteConnection


PACKAGE_ACTIONS_DIR = asset_actions_dir()


# ---------------------------------------------------------------------------
@pytest.fixture
async def runner_setup(session_dir):
    """Returns (runner, tasks, db) wired with a real ActionRegistry."""
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    tasks = TaskRegistry(db)
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    actions = ActionRegistry(PACKAGE_ACTIONS_DIR).load()
    policy = PolicyGate(
        flags=build_feature_flags(ExecutionMode.GUIDED_KERNEL_OPT),
        mode=ExecutionMode.GUIDED_KERNEL_OPT,
        role_registry=default_role_registry(),
        action_registry=actions,
    )
    backend = MockBackend()
    runner = SubAgentRunner(
        backend=backend,
        policy=policy,
        locks=locks,
        action_registry=actions,
        tasks=tasks,
        workspace=session_dir,
        agent_name="bench-runner-1",
    )
    yield runner, tasks, db, backend
    db.close()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_succeeds_for_valid_action(runner_setup):
    runner, tasks, db, backend = runner_setup
    # Pre-script a useful intent so metrics extraction succeeds.
    backend.script.append(
        ScriptStep(
            intents=[
                Intent(
                    type=IntentType.UPDATE_STATE,
                    payload={"changes": {"current_tput": 9876.5}},
                ),
                Intent(
                    type=IntentType.SEND_MESSAGE,
                    payload={"topic": "event", "body_md": "bench done"},
                ),
            ],
            note="bench script",
        )
    )
    task = await tasks.create(
        kind="delegate",
        params={
            "action_name": "bench_runner",
            "params": {"server": "sglang", "iters": 30},
            "requested_by": "executor",
        },
        idempotency_key="t1",
    )
    result = await runner.run(task)
    assert isinstance(result, TaskResult)
    assert result.status == "succeeded"
    assert result.metrics.get("tput") == 9876.5
    # Task moved to succeeded.
    refreshed = await tasks.get(task.task_id)
    assert refreshed.state == "succeeded"


@pytest.mark.asyncio
async def test_run_passes_action_allowed_tools_to_backend(runner_setup):
    runner, tasks, db, backend = runner_setup
    task = await tasks.create(
        kind="delegate",
        params={
            "action_name": "kernel_opt",
            "params": {"target_kernel": "rms_norm", "variant": "split_k"},
            "requested_by": "executor",
        },
        idempotency_key="t-kernel",
    )
    await runner.run(task)
    # MockBackend records calls in self.calls.
    assert backend.calls
    last = backend.calls[-1]
    assert "Edit" in last["allowed_tools"]
    assert "Bash" in last["allowed_tools"]
    assert "emit_intent" in last["allowed_tools"]


@pytest.mark.asyncio
async def test_run_composes_prompt_with_action_md(runner_setup):
    """Prompt should include the action.md body so the sub-agent has context."""
    runner, tasks, db, backend = runner_setup
    task = await tasks.create(
        kind="delegate",
        params={
            "action_name": "bench_runner",
            "params": {"warmup": 3},
            "requested_by": "executor",
        },
        idempotency_key="t-prompt",
    )
    await runner.run(task)
    # We can't see the raw prompt from MockBackend.calls (it logs prompt_chars
    # only); but we can verify _compose_prompt directly.
    actions = runner.actions
    action = actions.get("bench_runner")
    body = runner._compose_prompt(task, action)
    assert "Sub-agent: bench_runner" in body
    assert "warmup" in body  # task params injected
    assert "bench_runner" in body  # action md body


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_fails_safely_for_unknown_action(runner_setup):
    runner, tasks, db, backend = runner_setup
    task = await tasks.create(
        kind="delegate",
        params={
            "action_name": "_does_not_exist_",
            "params": {},
            "requested_by": "executor",
        },
        idempotency_key="t-bad",
    )
    result = await runner.run(task)
    assert result.status == "safely_failed"
    assert "_does_not_exist_" in result.notes
    # Task transitioned to failed.
    refreshed = await tasks.get(task.task_id)
    assert refreshed.state == "failed"


@pytest.mark.asyncio
async def test_run_marks_failed_on_backend_error(session_dir):
    """If the backend raises, the runner should release lanes + fail cleanly."""
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    tasks = TaskRegistry(db)
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    actions = ActionRegistry(PACKAGE_ACTIONS_DIR).load()
    policy = PolicyGate(
        flags=build_feature_flags(ExecutionMode.GUIDED_KERNEL_OPT),
        mode=ExecutionMode.GUIDED_KERNEL_OPT,
        role_registry=default_role_registry(),
        action_registry=actions,
    )

    class _BoomBackend(MockBackend):
        async def run(self, *a, **kw):
            raise RuntimeError("boom")

    backend = _BoomBackend()
    runner = SubAgentRunner(
        backend=backend,
        policy=policy,
        locks=locks,
        action_registry=actions,
        tasks=tasks,
    )
    task = await tasks.create(
        kind="delegate",
        params={"action_name": "bench_runner", "params": {}, "requested_by": "x"},
        idempotency_key="boom",
    )
    result = await runner.run(task)
    assert result.status == "failed"
    assert "boom" in result.notes
    # Task marked failed.
    refreshed = await tasks.get(task.task_id)
    assert refreshed.state == "failed"
    # And lanes were released — lock manager has no active leases.
    active = await locks.summary()
    assert active == []
    db.close()


@pytest.mark.asyncio
async def test_run_marks_needs_manual_review_when_no_intents(runner_setup):
    """If the sub-agent returns no intents, we mark needs_manual_review."""
    runner, tasks, db, backend = runner_setup
    backend.script.append(ScriptStep(intents=[]))
    task = await tasks.create(
        kind="delegate",
        params={
            "action_name": "bench_runner",
            "params": {},
            "requested_by": "x",
        },
        idempotency_key="quiet",
    )
    result = await runner.run(task)
    assert result.status == "needs_manual_review"
    refreshed = await tasks.get(task.task_id)
    assert refreshed.state == "needs_manual_review"


# ---------------------------------------------------------------------------
# Dispatcher pump
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dispatch_pending_delegates_drains_queue(runner_setup):
    runner, tasks, db, backend = runner_setup
    # Pre-stage one queued task.
    await tasks.create(
        kind="delegate",
        params={
            "action_name": "bench_runner",
            "params": {},
            "requested_by": "executor",
        },
        idempotency_key="q1",
    )
    await tasks.create(
        kind="delegate",
        params={
            "action_name": "bench_runner",
            "params": {"iters": 50},
            "requested_by": "executor",
        },
        idempotency_key="q2",
    )
    n = await dispatch_pending_delegates(runner, db=db)
    assert n == 2
    # Both completed.
    rows = await db.fetchall(
        "SELECT state FROM tasks WHERE kind=?", ("delegate",)
    )
    states = {r["state"] for r in rows}
    assert states <= {"succeeded", "needs_manual_review"}


@pytest.mark.asyncio
async def test_dispatch_pending_delegates_runs_with_stop_event(runner_setup):
    runner, tasks, db, backend = runner_setup
    stop = asyncio.Event()

    async def kick():
        await asyncio.sleep(0.3)
        stop.set()

    n, _ = await asyncio.gather(
        dispatch_pending_delegates(runner, db=db, stop=stop, poll_interval_s=0.1),
        kick(),
    )
    assert n == 0  # nothing queued


# ---------------------------------------------------------------------------
# Metrics / artifact extraction
# ---------------------------------------------------------------------------
def test_extract_metrics_reads_propose_action_gain():
    intents = [
        Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "x", "predicted_gain_pct": 5.5},
        )
    ]
    metrics = SubAgentRunner._extract_metrics(intents)
    assert metrics.get("predicted_gain_pct") == 5.5


def test_extract_artifacts_reads_known_keys():
    intents = [
        Intent(
            type=IntentType.SEND_MESSAGE,
            payload={
                "topic": "event",
                "body_md": "done",
                "artifact_path": "/tmp/a",
                "log_path": "/tmp/b.log",
            },
        )
    ]
    arts = SubAgentRunner._extract_artifacts(intents)
    assert "/tmp/a" in arts
    assert "/tmp/b.log" in arts
