# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for targeted-build enqueue, executor lifecycle, and resume recovery."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.framework.build_actions import TargetedBuildAction
from hyperloom.orchestrator.loop.build_lifecycle import _driver_command


def _action(cmd, **kw):
    base = dict(
        gap_id="g", framework="vllm", component="aiter", capability="fp4_moe",
        build_command=tuple(cmd),
    )
    base.update(kw)
    return TargetedBuildAction(**base)


def _real_action(**kw):
    base = dict(
        gap_id="g", framework="vllm", component="aiter", capability="fp4_moe",
        ref="v0.1.0", repo_url="https://github.com/ROCm/aiter", gpu_arch="gfx950",
    )
    base.update(kw)
    return TargetedBuildAction(**base)


def _fake_action(**kw):
    base = dict(
        gap_id="g", framework="vllm", component="aiter", capability="fp4_moe",
        build_command=(sys.executable, "-c", "print('fake')"),
    )
    base.update(kw)
    return TargetedBuildAction(**base)


async def _enqueue_and_run(build_lifecycle, executor, *, action, session_dir) -> tuple:
    """Run one build through SubAgentRunner; return (task_row, runner_result)."""
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner

    tid = await build_lifecycle.enqueue_targeted_build(action)
    task_obj = await build_lifecycle.tasks.get(tid)
    runner = SubAgentRunner(
        locks=build_lifecycle.locks,
        tasks=build_lifecycle.tasks,
        executor_registry={"targeted_build": executor},
        session_dir=Path(session_dir),
        shared_state=build_lifecycle.shared_state,
    )
    lease = await build_lifecycle.locks.try_acquire_many(
        ["build_lane"],
        holder_id=tid, task_id=tid, action="targeted_build",
        ttl_sec=task_obj.lease_ttl_sec or 60,
    )
    assert lease is not None
    result = await runner.run_task(task_obj, prebound_lease=lease)
    return await build_lifecycle.tasks.get(tid), result


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_idempotent_enqueue_no_double_row(build_coord, build_lifecycle):
    a = _action([sys.executable, "-c", "print('x')"], ref="v1", gpu_arch="gfx950")
    t1 = await build_lifecycle.enqueue_targeted_build(a)
    t2 = await build_lifecycle.enqueue_targeted_build(a)
    assert t1 == t2
    all_builds = [t for t in await build_lifecycle.tasks.queued() if t.kind == "targeted_build"]
    assert len(all_builds) == 1


@pytest.mark.asyncio
async def test_build_lane_serializes_two_builds(build_coord, build_lifecycle):
    """Capacity-1 build_lane: second build stays queued while first runs."""
    a1 = await build_lifecycle.enqueue_targeted_build(
        _action([sys.executable, "-c", "import time; time.sleep(60)"], ref="v1")
    )
    a2 = await build_lifecycle.enqueue_targeted_build(
        _action([sys.executable, "-c", "print('two')"], ref="v2")
    )
    assert a1 != a2
    t1 = await build_lifecycle.tasks.get(a1)
    lease = await build_lifecycle.locks.try_acquire_many(
        ["build_lane"], holder_id=a1, task_id=a1, action="targeted_build",
        ttl_sec=t1.lease_ttl_sec or 60,
    )
    assert lease is not None
    await build_lifecycle.tasks.transition(a1, "running")
    running = [t.task_id for t in await build_lifecycle.tasks.by_state("running") if t.kind == "targeted_build"]
    queued = [t.task_id for t in await build_lifecycle.tasks.queued() if t.kind == "targeted_build"]
    assert len(running) == 1
    assert len(queued) == 1
    await build_lifecycle.locks.release(lease)


@pytest.mark.asyncio
async def test_build_lane_does_not_conflict_with_serving(build_coord, build_lifecycle):
    """build_lane must not mutex the serving/benchmark lanes."""
    tid = await build_lifecycle.enqueue_targeted_build(
        _action([sys.executable, "-c", "import time; time.sleep(2)"])
    )
    t = await build_lifecycle.tasks.get(tid)
    build_lease = await build_lifecycle.locks.try_acquire_many(
        ["build_lane"], holder_id=tid, task_id=tid, action="targeted_build",
        ttl_sec=t.lease_ttl_sec or 60,
    )
    assert build_lease is not None
    bench_lease = await build_coord.locks.try_acquire_many(
        ["benchmark_lane"], holder_id="bench", task_id="bench", action="bench", ttl_sec=60
    )
    assert bench_lease is not None
    await build_coord.locks.release(bench_lease)
    await build_coord.locks.release(build_lease)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

@pytest.fixture
def executor(tmp_path):
    from hyperloom.orchestrator.actions.executors.targeted_build_executor import TargetedBuildExecutor
    return TargetedBuildExecutor()


@pytest.mark.asyncio
async def test_build_succeeds(build_coord, build_lifecycle, executor, tmp_path):
    task, _ = await _enqueue_and_run(
        build_lifecycle, executor,
        action=_action([sys.executable, "-c", "print('ok')"]),
        session_dir=tmp_path,
    )
    assert task.state == "succeeded"
    assert build_coord.shared_state.enablement.build_manifest
    assert build_coord.shared_state.pending_targeted_build == {}
    holders = await build_coord.locks.lane_holders()
    assert holders.get("build_lane", 0) == 0 or "build_lane" not in holders


@pytest.mark.asyncio
async def test_nonzero_exit_records_compile_error(build_coord, build_lifecycle, executor, tmp_path):
    task, _ = await _enqueue_and_run(
        build_lifecycle, executor,
        action=_action([sys.executable, "-c", "import sys; sys.exit(2)"]),
        session_dir=tmp_path,
    )
    assert task.state == "failed"
    assert build_coord.shared_state.enablement.last_build_failure["failure_class"] == "compile_error"


@pytest.mark.asyncio
async def test_timeout_kills_and_records_timeout(build_coord, build_lifecycle, tmp_path):
    from hyperloom.orchestrator.actions.executors.targeted_build_executor import TargetedBuildExecutor
    action = _action([sys.executable, "-c", "import time; time.sleep(600)"], build_budget_sec=1)
    task, _ = await _enqueue_and_run(
        build_lifecycle, TargetedBuildExecutor(),
        action=action, session_dir=tmp_path,
    )
    assert task.state == "failed"
    assert build_coord.shared_state.enablement.last_build_failure["failure_class"] == "timeout"
    holders = await build_coord.locks.lane_holders()
    assert holders.get("build_lane", 0) == 0 or "build_lane" not in holders


@pytest.mark.asyncio
async def test_cancel_kills_the_compile_before_releasing_the_lane(
    build_coord, build_lifecycle, executor, tmp_path
):
    """A cancelled build must not leave the compile running.

    The lane is released as this coroutine unwinds, so a surviving process group
    would compile on while the next build holds build_lane.
    """
    import asyncio

    from hyperloom.orchestrator.actions.executors import targeted_build_executor as tbe_mod

    spawned: list = []
    real_spawn = tbe_mod.spawn_build

    def _capture(*a, **kw):
        handle = real_spawn(*a, **kw)
        spawned.append(handle)
        return handle

    tbe_mod.spawn_build = _capture
    try:
        run = asyncio.create_task(
            _enqueue_and_run(
                build_lifecycle, executor,
                action=_action([sys.executable, "-c", "import time; time.sleep(600)"]),
                session_dir=tmp_path,
            )
        )
        for _ in range(200):
            await asyncio.sleep(0.02)
            if spawned:
                break
        assert spawned, "build never spawned"
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
    finally:
        tbe_mod.spawn_build = real_spawn

    handle = spawned[0]
    with pytest.raises(ProcessLookupError):
        os.killpg(handle.pgid, 0)
    assert build_coord.shared_state.pending_targeted_build == {}


@pytest.mark.asyncio
async def test_a_failed_sentinel_write_still_kills_the_compile(
    build_coord, build_lifecycle, executor, tmp_path
):
    """The spawn is inside the teardown's scope, so a raise cannot orphan it."""
    from hyperloom.orchestrator.actions.executors import targeted_build_executor as tbe_mod

    spawned: list = []
    real_spawn = tbe_mod.spawn_build

    def _capture(*a, **kw):
        handle = real_spawn(*a, **kw)
        spawned.append(handle)
        return handle

    original_save = type(build_coord.shared_state).save
    calls = {"n": 0}

    def _fail_first_save(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("state dir is read-only")
        return original_save(self, *a, **kw)

    tbe_mod.spawn_build = _capture
    type(build_coord.shared_state).save = _fail_first_save
    try:
        task, _ = await _enqueue_and_run(
            build_lifecycle, executor,
            action=_action([sys.executable, "-c", "import time; time.sleep(600)"]),
            session_dir=tmp_path,
        )
    finally:
        tbe_mod.spawn_build = real_spawn
        type(build_coord.shared_state).save = original_save

    assert spawned, "build never spawned"
    assert task.state == "failed"
    with pytest.raises(ProcessLookupError):
        os.killpg(spawned[0].pgid, 0)


# ---------------------------------------------------------------------------
# Driver wiring
# ---------------------------------------------------------------------------

def test_driver_command_real_component_uses_driver_module(tmp_path):
    action = _real_action()
    attempt_root = str(tmp_path / "attempt")
    cmd = _driver_command(action, attempt_root)
    assert cmd[0] == sys.executable
    assert "targeted_build" in " ".join(cmd)
    assert "--attempt-root" in cmd
    plan = tmp_path / "attempt" / "plan.json"
    assert plan.exists()
    assert json.loads(plan.read_text())["component"] == "aiter"


def test_driver_command_explicit_build_command_passthrough(tmp_path):
    action = _fake_action()
    attempt_root = str(tmp_path / "attempt2")
    cmd = _driver_command(action, attempt_root)
    assert cmd == list(action.build_command)
    assert not (tmp_path / "attempt2" / "plan.json").exists()


@pytest.mark.asyncio
async def test_real_component_writes_plan_json_before_spawn(build_coord, build_lifecycle, tmp_path):
    """plan.json must exist before spawn_build is called."""
    from hyperloom.orchestrator.actions.executors.targeted_build_executor import TargetedBuildExecutor
    from hyperloom.orchestrator.actions.executors import targeted_build_executor as tbe_mod
    from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext

    action = _real_action()
    executor = TargetedBuildExecutor()
    tid = await build_lifecycle.enqueue_targeted_build(action)
    task_obj = await build_lifecycle.tasks.get(tid)
    await build_lifecycle.tasks.transition(tid, "running")

    spawned_commands: list[list[str]] = []

    def _capture_spawn(action_inner, *, attempt_root, command, **kw):
        spawned_commands.append(list(command) if command else [])
        plan = Path(attempt_root) / "plan.json"
        assert plan.exists()
        assert json.loads(plan.read_text())["component"] == "aiter"
        raise RuntimeError("captured")

    original = tbe_mod.spawn_build
    tbe_mod.spawn_build = _capture_spawn
    try:
        with pytest.raises(RuntimeError, match="captured"):
            ctx = RunnerContext(
                task=task_obj, lease=None,
                extra={"shared_state": build_coord.shared_state, "session_dir": str(tmp_path)},
            )
            await executor(ctx)
    finally:
        tbe_mod.spawn_build = original

    assert spawned_commands
    cmd = spawned_commands[0]
    assert sys.executable in cmd
    assert "targeted_build" in " ".join(cmd)


@pytest.mark.asyncio
async def test_explicit_build_command_passed_verbatim(build_coord, build_lifecycle, tmp_path):
    from hyperloom.orchestrator.actions.executors.targeted_build_executor import TargetedBuildExecutor
    from hyperloom.orchestrator.actions.executors import targeted_build_executor as tbe_mod
    from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext

    action = _fake_action()
    executor = TargetedBuildExecutor()
    tid = await build_lifecycle.enqueue_targeted_build(action)
    task_obj = await build_lifecycle.tasks.get(tid)
    await build_lifecycle.tasks.transition(tid, "running")

    spawned_commands: list[list[str]] = []

    def _capture_spawn(action_inner, *, attempt_root, command, **kw):
        spawned_commands.append(list(command) if command else [])
        raise RuntimeError("captured")

    original = tbe_mod.spawn_build
    tbe_mod.spawn_build = _capture_spawn
    try:
        with pytest.raises(RuntimeError, match="captured"):
            ctx = RunnerContext(
                task=task_obj, lease=None,
                extra={"shared_state": build_coord.shared_state, "session_dir": str(tmp_path)},
            )
            await executor(ctx)
    finally:
        tbe_mod.spawn_build = original

    assert spawned_commands
    assert spawned_commands[0] == list(action.build_command)


# ---------------------------------------------------------------------------
# Policy gate
# ---------------------------------------------------------------------------

def test_targeted_build_in_coordinator_internal_actions():
    from hyperloom.inference_optimizer.protocol.action_surfaces import COORDINATOR_INTERNAL_ACTIONS
    assert "targeted_build" in COORDINATOR_INTERNAL_ACTIONS


def test_targeted_build_params_pass_policy_gate(tmp_path):
    """Build params must not be misidentified as path-like fields."""
    from hyperloom.orchestrator.policy.gate import PolicyGate
    from hyperloom.orchestrator.roles.agent_role import AgentRole, BackendType
    from hyperloom.inference_optimizer.protocol.intent import IntentType

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    orch_role = AgentRole(
        name="orchestration",
        backend_type=BackendType.CLAUDE,
        model="claude-3-5-sonnet",
        api_key_env="ANTHROPIC_API_KEY",
        allowed_intents=frozenset(IntentType),
    )
    gate = PolicyGate(
        role_registry={"orchestration": orch_role},
        session_dir=session_dir,
        strict_paths=True,
    )
    action = TargetedBuildAction(
        gap_id="gap.enablement.fp4_moe",
        framework="vllm",
        component="aiter",
        capability="fp4_moe",
        repo_url="https://github.com/ROCm/aiter",
        ref="v0.1.0",
        gpu_arch="gfx950",
        build_command=("/usr/bin/gcc", "-O2", "/tmp/build.c"),
    )
    gate.validate_dispatched_task("targeted_build", action.to_state())


# ---------------------------------------------------------------------------
# P1-12 regression: spawn failure lands row in failed state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spawn_failure_marks_row_failed(build_coord, build_lifecycle, tmp_path):
    """spawn failure via sub_agent_runner writes failed terminal state and releases lane."""
    from hyperloom.orchestrator.actions.executors.targeted_build_executor import TargetedBuildExecutor
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner

    action = _action(["/nonexistent_compiler_xyz_P1_12"])
    tid = await build_lifecycle.enqueue_targeted_build(action)
    task_obj = await build_lifecycle.tasks.get(tid)
    executor = TargetedBuildExecutor()
    runner = SubAgentRunner(
        locks=build_coord.locks,
        tasks=build_coord.tasks,
        executor_registry={"targeted_build": executor},
        session_dir=tmp_path,
        shared_state=build_coord.shared_state,
    )
    lease = await build_lifecycle.locks.try_acquire_many(
        ["build_lane"], holder_id=tid, task_id=tid, action="targeted_build",
        ttl_sec=task_obj.lease_ttl_sec or 60,
    )
    await runner.run_task(task_obj, prebound_lease=lease)
    t = await build_coord.tasks.get(tid)
    assert t.state == "failed"
    holders = await build_coord.locks.lane_holders()
    assert holders.get("build_lane", 0) == 0 or "build_lane" not in holders


# ---------------------------------------------------------------------------
# Resume recovery
# ---------------------------------------------------------------------------

def _silent_plan():
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
    from hyperloom.orchestrator.roles import ScriptedPlan
    return ScriptedPlan(
        turns=[],
        default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"}),
    )


def _build_backends():
    from hyperloom.orchestrator.roles import MockBackend
    return {
        name: MockBackend(_silent_plan(), name=name)
        for name in ("orchestration", "critic", "robustness")
    }


@pytest.fixture
def resume_session_dir(tmp_path, monkeypatch) -> Path:
    from hyperloom.inference_optimizer.session.paths import make_session_dir
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


@pytest.fixture
def resume_coord(resume_session_dir):
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    return Coordinator(resume_session_dir, backends=_build_backends())


@pytest.mark.asyncio
async def test_resume_kills_orphan_and_clears_sentinel(resume_coord):
    import subprocess
    resume_coord._resumed_from["is_resume"] = True

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True
    )
    pgid = os.getpgid(proc.pid)

    attempt_root = Path(resume_coord.session_dir) / "enablement" / "builds" / "t-orphan"
    jit_dir = attempt_root / "aiter_jit"
    jit_dir.mkdir(parents=True, exist_ok=True)
    stale_lock = jit_dir / "lock"
    stale_lock.write_text("")
    os.utime(stale_lock, (time.time() - 3600,) * 2)

    action = TargetedBuildAction(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe")
    task, _ = await resume_coord.tasks.create_or_return_existing(
        kind="targeted_build", params=action.to_state(), idempotency_key="k-orphan",
        requires_lanes=["build_lane"],
    )
    await resume_coord.tasks.transition(task.task_id, "running")
    resume_coord.shared_state.pending_targeted_build = {
        "task_id": task.task_id,
        "pid": proc.pid,
        "pgid": pgid,
        "attempt_root": str(attempt_root),
        "aiter_jit_dir": str(jit_dir),
    }

    report = await resume_coord._resume_consistency_pass()

    fix = next(f for f in report["fixes"] if isinstance(f, dict) and f["kind"] == "reclaimed_pending_targeted_build")
    assert fix["task_id"] == task.task_id
    assert resume_coord.shared_state.pending_targeted_build == {}
    assert resume_coord.shared_state.enablement.last_build_failure["failure_class"] == "timeout"
    assert (await resume_coord.tasks.get(task.task_id)).state == "failed"
    assert not attempt_root.exists()
    for _ in range(40):
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


@pytest.mark.asyncio
async def test_resume_no_pending_is_noop(resume_coord):
    resume_coord._resumed_from["is_resume"] = True
    resume_coord.shared_state.pending_targeted_build = {}
    report = await resume_coord._resume_consistency_pass()
    assert not any(
        isinstance(f, dict) and f.get("kind") == "reclaimed_pending_targeted_build"
        for f in report["fixes"]
    )
