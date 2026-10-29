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

from hyperloom.orchestrator.enablement.runtime.build_actions import TargetedBuildAction
from hyperloom.orchestrator.loop.build_lifecycle import _driver_command
from hyperloom.orchestrator.state.task_registry import TaskSpec


def _action(**kw):
    base = dict(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe")
    base.update(kw)
    return TargetedBuildAction(**base)


def _drive_with(monkeypatch, cmd):
    """Make the executor spawn ``cmd`` in place of the plan.json driver."""
    from hyperloom.orchestrator.actions.executors import targeted_build_executor as tbe

    monkeypatch.setattr(tbe, "_driver_command", lambda _action, _root: list(cmd))


#: Exits 0 the way the driver does: only after writing an ok ``result.json``.
_WRITES_OK_RESULT = [sys.executable, "-c", "import json; open('result.json', 'w').write(json.dumps({'ok': True}))"]


def _real_action(**kw):
    base = dict(
        gap_id="g",
        framework="vllm",
        component="aiter",
        capability="fp4_moe",
        ref="v0.1.0",
        repo_url="https://github.com/ROCm/aiter",
        gpu_arch="gfx950",
    )
    base.update(kw)
    return TargetedBuildAction(**base)


async def _enqueue_and_run(build_coord, executor, *, action, session_dir) -> tuple:
    """Run one build through SubAgentRunner; return (task_row, runner_result)."""
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner

    tid = await build_coord.build_lifecycle.enqueue_targeted_build(action)
    task_obj = await build_coord.tasks.get(tid)
    runner = SubAgentRunner(locks=build_coord.locks, tasks=build_coord.tasks, session_dir=Path(session_dir))
    runner.shared_state = build_coord.shared_state
    runner.register_executor("targeted_build", executor)
    lease = await build_coord.locks.acquire_many(
        ["build_lane"],
        holder_id=tid,
        task_id=tid,
        action="targeted_build",
        ttl_sec=task_obj.lease_ttl_sec or 60,
    )
    assert lease is not None
    result = await runner.run_task(task_obj, prebound_lease=lease)
    return await build_coord.tasks.get(tid), result


# Enqueue


@pytest.mark.asyncio
async def test_idempotent_enqueue_no_double_row(build_coord):
    a = _action(ref="v1", gpu_arch="gfx950")
    t1 = await build_coord.build_lifecycle.enqueue_targeted_build(a)
    t2 = await build_coord.build_lifecycle.enqueue_targeted_build(a)
    assert t1 == t2
    all_builds = [t for t in await build_coord.tasks.queued() if t.kind == "targeted_build"]
    assert len(all_builds) == 1


def test_idempotency_key_keeps_the_shape_the_task_registry_persisted():
    """A resumed session must still recognise the builds it already attempted."""
    import hashlib

    from hyperloom.orchestrator.loop.build_lifecycle import _novelty_idempotency_key

    action = _action(repo_url="https://github.com/ROCm/aiter.git", ref="v1", gpu_arch="gfx950")
    persisted = json.dumps(["aiter", "https://github.com/rocm/aiter", "fp4_moe", "v1", "gfx950", []], sort_keys=True)
    digest = hashlib.sha1(persisted.encode(), usedforsecurity=False).hexdigest()[:16]
    assert _novelty_idempotency_key(action) == f"targeted_build:aiter:{digest}"


@pytest.mark.asyncio
async def test_build_lane_serializes_two_builds(build_coord):
    """Capacity-1 build_lane: second build stays queued while first runs."""
    a1 = await build_coord.build_lifecycle.enqueue_targeted_build(_action(ref="v1"))
    a2 = await build_coord.build_lifecycle.enqueue_targeted_build(_action(ref="v2"))
    assert a1 != a2
    t1 = await build_coord.tasks.get(a1)
    lease = await build_coord.locks.acquire_many(
        ["build_lane"],
        holder_id=a1,
        task_id=a1,
        action="targeted_build",
        ttl_sec=t1.lease_ttl_sec or 60,
    )
    assert lease is not None
    await build_coord.tasks.transition(a1, "running")
    running = [t.task_id for t in await build_coord.tasks.by_state("running") if t.kind == "targeted_build"]
    queued = [t.task_id for t in await build_coord.tasks.queued() if t.kind == "targeted_build"]
    assert len(running) == 1
    assert len(queued) == 1
    await build_coord.locks.release(lease)


@pytest.mark.asyncio
async def test_build_lane_does_not_conflict_with_serving(build_coord):
    """build_lane must not mutex the serving/benchmark lanes."""
    tid = await build_coord.build_lifecycle.enqueue_targeted_build(_action())
    t = await build_coord.tasks.get(tid)
    build_lease = await build_coord.locks.acquire_many(
        ["build_lane"],
        holder_id=tid,
        task_id=tid,
        action="targeted_build",
        ttl_sec=t.lease_ttl_sec or 60,
    )
    assert build_lease is not None
    bench_lease = await build_coord.locks.acquire_many(
        ["benchmark_lane"], holder_id="bench", task_id="bench", action="bench", ttl_sec=60
    )
    assert bench_lease is not None
    await build_coord.locks.release(bench_lease)
    await build_coord.locks.release(build_lease)


# Executor


@pytest.fixture
def executor(tmp_path):
    from hyperloom.orchestrator.actions.executors.targeted_build_executor import TargetedBuildExecutor

    return TargetedBuildExecutor()


@pytest.mark.asyncio
async def test_build_succeeds(build_coord, executor, tmp_path, monkeypatch):
    _drive_with(monkeypatch, _WRITES_OK_RESULT)
    task, _ = await _enqueue_and_run(build_coord, executor, action=_action(), session_dir=tmp_path)
    assert task.state == "succeeded"
    assert build_coord.shared_state.enablement.build_manifest
    assert build_coord.shared_state.pending_targeted_build == {}
    holders = await build_coord.locks.lane_holders()
    assert holders.get("build_lane", 0) == 0 or "build_lane" not in holders


@pytest.mark.asyncio
async def test_nonzero_exit_records_compile_error(build_coord, executor, tmp_path, monkeypatch):
    _drive_with(monkeypatch, [sys.executable, "-c", "import sys; sys.exit(2)"])
    task, _ = await _enqueue_and_run(build_coord, executor, action=_action(), session_dir=tmp_path)
    assert task.state == "failed"
    assert build_coord.shared_state.enablement.last_build_failure["failure_class"] == "compile_error"


@pytest.mark.asyncio
async def test_timeout_kills_and_records_timeout(build_coord, tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import targeted_build_executor as tbe

    _drive_with(monkeypatch, [sys.executable, "-c", "import time; time.sleep(600)"])
    monkeypatch.setattr(tbe, "default_budget_sec", lambda _component: 1)
    task, _ = await _enqueue_and_run(build_coord, tbe.TargetedBuildExecutor(), action=_action(), session_dir=tmp_path)
    assert task.state == "failed"
    assert build_coord.shared_state.enablement.last_build_failure["failure_class"] == "timeout"
    holders = await build_coord.locks.lane_holders()
    assert holders.get("build_lane", 0) == 0 or "build_lane" not in holders


@pytest.mark.asyncio
async def test_cancel_kills_the_compile_before_releasing_the_lane(build_coord, executor, tmp_path, monkeypatch):
    """A cancelled build must not leave the compile running."""
    import asyncio

    _drive_with(monkeypatch, [sys.executable, "-c", "import time; time.sleep(600)"])
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
                build_coord,
                executor,
                action=_action(),
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
async def test_a_failed_sentinel_write_still_kills_the_compile(build_coord, executor, tmp_path, monkeypatch):
    """The spawn is inside the teardown's scope, so a raise cannot orphan it."""
    from hyperloom.orchestrator.actions.executors import targeted_build_executor as tbe_mod

    _drive_with(monkeypatch, [sys.executable, "-c", "import time; time.sleep(600)"])
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
            build_coord,
            executor,
            action=_action(),
            session_dir=tmp_path,
        )
    finally:
        tbe_mod.spawn_build = real_spawn
        type(build_coord.shared_state).save = original_save

    assert spawned, "build never spawned"
    assert task.state == "failed"
    with pytest.raises(ProcessLookupError):
        os.killpg(spawned[0].pgid, 0)


@pytest.mark.asyncio
async def test_scope_cancelled_build_does_not_spawn(monkeypatch, tmp_path):
    from concurrent.futures import CancelledError
    from types import SimpleNamespace
    from unittest.mock import Mock

    from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope
    from hyperloom.orchestrator.actions.executors import targeted_build_executor as tbe
    from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext

    scope = CancelScope()
    scope.cancel(reason="session_stopped")
    spawn = Mock(side_effect=AssertionError("cancelled build must not spawn"))
    monkeypatch.setattr(tbe, "spawn_build", spawn)
    ctx = RunnerContext(
        task=SimpleNamespace(task_id="cancelled-build", params=_action().to_state()),
        lease=None,
        extra={"session_dir": str(tmp_path)},
    )
    with use_cancel_scope(scope), pytest.raises(CancelledError, match="session_stopped"):
        await tbe.TargetedBuildExecutor()(ctx)
    spawn.assert_not_called()
    assert not scope.has_listeners


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed_dead", [True, False])
@pytest.mark.parametrize("cancel_kind", ["scope", "direct", "success", "compile_error", "timeout"])
async def test_scope_cancellation_reaps_build_and_preserves_cleanup_evidence(
    monkeypatch, tmp_path, confirmed_dead, cancel_kind
):
    import asyncio
    import subprocess
    from concurrent.futures import CancelledError
    from types import SimpleNamespace
    from unittest.mock import Mock

    from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope
    from hyperloom.orchestrator.actions.executors import targeted_build_executor as tbe
    from hyperloom.orchestrator.enablement.runtime.targeted_build import BuildHandle
    from hyperloom.orchestrator.loop.sub_agent_runner import ExecutionCleanupUnconfirmed, RunnerContext

    entered = asyncio.Event()
    action = _action(attempt_root=str(tmp_path))
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(int(sys.stdin.readline()))"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    handle = BuildHandle(
        action, str(tmp_path), str(tmp_path / "aiter_jit"), str(tmp_path / "build.log"), child, child.pid, child.pid
    )
    shared = SimpleNamespace(pending_targeted_build={}, save=Mock())
    scope = CancelScope()
    cleanup = Mock()

    def spawn(*args, **kwargs):
        entered.set()
        return handle

    def ensure_dead(build):
        cleanup(build)
        if confirmed_dead:
            child.terminate()
            child.wait(timeout=3)
        return confirmed_dead

    monkeypatch.setattr(tbe, "spawn_build", spawn)
    monkeypatch.setattr(tbe, "ensure_build_dead", ensure_dead)
    if cancel_kind == "timeout":
        monkeypatch.setattr(tbe, "default_budget_sec", lambda _component: 0.05)
    record = Mock()
    monkeypatch.setattr(tbe.TargetedBuildExecutor, "_record_result", record)
    ctx = RunnerContext(
        task=SimpleNamespace(task_id="running-build", params=action.to_state()),
        lease=None,
        extra={"session_dir": str(tmp_path), "shared_state": shared},
    )
    run = None
    try:
        with use_cancel_scope(scope):
            run = asyncio.create_task(tbe.TargetedBuildExecutor()(ctx))
        await asyncio.wait_for(entered.wait(), 3)
        assert scope.has_listeners
        if cancel_kind == "scope":
            scope.cancel(reason="session_stopped")
        elif cancel_kind == "direct":
            run.cancel()
        elif cancel_kind in {"success", "compile_error"}:
            if cancel_kind == "success":
                (tmp_path / "result.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            child.stdin.write(b"0\n" if cancel_kind == "success" else b"2\n")
            child.stdin.flush()
        if confirmed_dead and cancel_kind == "success":
            result = await asyncio.wait_for(asyncio.shield(run), 3)
            assert result["ok"] is True
        else:
            expected = (
                ExecutionCleanupUnconfirmed
                if not confirmed_dead
                else CancelledError
                if cancel_kind == "scope"
                else asyncio.CancelledError
                if cancel_kind == "direct"
                else RuntimeError
            )
            with pytest.raises(expected) as raised:
                await asyncio.wait_for(asyncio.shield(run), 3)
        cleanup.assert_called_once_with(handle)
        if confirmed_dead:
            assert child.poll() is not None
            assert shared.pending_targeted_build == {}
        else:
            assert (child.poll() is not None) is (cancel_kind in {"success", "compile_error"})
            assert shared.pending_targeted_build["pgid"] == child.pid
            if cancel_kind == "direct":
                assert raised.value.result is None
            else:
                expected_class = (
                    "cancelled" if cancel_kind == "scope" else "ok" if cancel_kind == "success" else cancel_kind
                )
                assert raised.value.result.result["failure_class"] == expected_class
                assert raised.value.result.state == (
                    "cancelled" if cancel_kind == "scope" else "succeeded" if cancel_kind == "success" else "failed"
                )
        if confirmed_dead and cancel_kind != "direct":
            record.assert_called_once()
            assert record.call_args.args[0].failure_class == (
                "cancelled" if cancel_kind == "scope" else "ok" if cancel_kind == "success" else cancel_kind
            )
        assert not scope.has_listeners
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=3)
        child.stdin.close()
        if run is not None:
            await asyncio.gather(run, return_exceptions=True)


# Driver wiring


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


@pytest.mark.asyncio
async def test_real_component_writes_plan_json_before_spawn(build_coord, tmp_path):
    """plan.json must exist before spawn_build is called."""
    from hyperloom.orchestrator.actions.executors.targeted_build_executor import TargetedBuildExecutor
    from hyperloom.orchestrator.actions.executors import targeted_build_executor as tbe_mod
    from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext

    action = _real_action()
    executor = TargetedBuildExecutor()
    tid = await build_coord.build_lifecycle.enqueue_targeted_build(action)
    task_obj = await build_coord.tasks.get(tid)
    await build_coord.tasks.transition(tid, "running")

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
                task=task_obj,
                lease=None,
                extra={"shared_state": build_coord.shared_state, "session_dir": str(tmp_path)},
            )
            await executor(ctx)
    finally:
        tbe_mod.spawn_build = original

    assert spawned_commands
    cmd = spawned_commands[0]
    assert sys.executable in cmd
    assert "targeted_build" in " ".join(cmd)


# Policy gate


def test_targeted_build_in_coordinator_internal_actions():
    from hyperloom.inference_optimizer.protocol.action_surfaces import COORDINATOR_INTERNAL_ACTIONS

    assert "targeted_build" in COORDINATOR_INTERNAL_ACTIONS


def test_targeted_build_params_pass_policy_gate(tmp_path):
    """Build params must not be misidentified as path-like fields."""
    from hyperloom.orchestrator.policy.gate import PolicyGate
    from hyperloom.orchestrator.roles.agent_role import AgentRole
    from hyperloom.inference_optimizer.protocol.intent import IntentType

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    orch_role = AgentRole(
        name="orchestration",
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
    )
    gate.validate_dispatched_task("targeted_build", action.to_state())


# P1-12 regression: spawn failure lands row in failed state


@pytest.mark.asyncio
async def test_spawn_failure_marks_row_failed(build_coord, tmp_path, monkeypatch):
    """spawn failure via sub_agent_runner writes failed terminal state and releases lane."""
    from hyperloom.orchestrator.actions.executors.targeted_build_executor import TargetedBuildExecutor
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner

    _drive_with(monkeypatch, ["/nonexistent_compiler_xyz_P1_12"])
    action = _action()
    tid = await build_coord.build_lifecycle.enqueue_targeted_build(action)
    task_obj = await build_coord.tasks.get(tid)
    runner = SubAgentRunner(locks=build_coord.locks, tasks=build_coord.tasks, session_dir=tmp_path)
    runner.shared_state = build_coord.shared_state
    runner.register_executor("targeted_build", TargetedBuildExecutor())
    lease = await build_coord.locks.acquire_many(
        ["build_lane"],
        holder_id=tid,
        task_id=tid,
        action="targeted_build",
        ttl_sec=task_obj.lease_ttl_sec or 60,
    )
    await runner.run_task(task_obj, prebound_lease=lease)
    t = await build_coord.tasks.get(tid)
    assert t.state == "failed"
    holders = await build_coord.locks.lane_holders()
    assert holders.get("build_lane", 0) == 0 or "build_lane" not in holders


# Resume recovery


def _silent_plan():
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
    from hyperloom.orchestrator.roles import ScriptedPlan

    return ScriptedPlan(
        turns=[],
        default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"}),
    )


def _build_backends():
    from hyperloom.orchestrator.roles import MockBackend

    return {name: MockBackend(_silent_plan(), name=name) for name in ("orchestration", "critic")}


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

    resume_coord.writeback._resumed_from["is_resume"] = True

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True)
    pgid = os.getpgid(proc.pid)

    attempt_root = Path(resume_coord.session_dir) / "enablement" / "builds" / "t-orphan"
    jit_dir = attempt_root / "aiter_jit"
    jit_dir.mkdir(parents=True, exist_ok=True)
    stale_lock = jit_dir / "lock"
    stale_lock.write_text("")
    os.utime(stale_lock, (time.time() - 3600,) * 2)

    action = TargetedBuildAction(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe")
    task, _ = await resume_coord.tasks.create_or_return_existing(
        TaskSpec(
            kind="targeted_build",
            params=action.to_state(),
            idempotency_key="k-orphan",
            requires_lanes=["build_lane"],
        ),
    )
    await resume_coord.tasks.transition(task.task_id, "running")
    resume_coord.shared_state.pending_targeted_build = {
        "task_id": task.task_id,
        "pgid": pgid,
        "attempt_root": str(attempt_root),
        "aiter_jit_dir": str(jit_dir),
    }

    report = await resume_coord.writeback._resume_consistency_pass()

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


def _jit_sweep_bug(coord, monkeypatch, tmp_path):
    from hyperloom.orchestrator.actions.executors import _aiter_jit

    def _bug(**_kwargs):
        raise KeyError("lock table")

    monkeypatch.setattr(_aiter_jit, "sweep_stale_aiter_locks_if_dead", _bug)
    coord.shared_state.pending_targeted_build = {"task_id": "", "aiter_jit_dir": str(tmp_path / "aiter_jit")}
    return coord.writeback._resume_recover_pending_targeted_build, KeyError, "lock table"


def _revalidation_row_unreadable(coord, monkeypatch, tmp_path):
    import sqlite3

    async def _locked(_task_id):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(coord.tasks, "get", _locked)
    coord.shared_state.enablement.validation_pending = True
    coord.shared_state.enablement.revalidation_task_id = "t-reval"
    return coord.writeback._resume_recover_pending_revalidation, sqlite3.OperationalError, "database is locked"


def _keep_log_unreadable(coord, monkeypatch, tmp_path):
    import sqlite3

    async def _locked(**_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(coord.bus, "tail", _locked)
    return coord.writeback._resume_recover_orphaned_keeps, sqlite3.OperationalError, "database is locked"


@pytest.mark.asyncio
@pytest.mark.parametrize("broken", [_jit_sweep_bug, _revalidation_row_unreadable, _keep_log_unreadable])
async def test_a_failed_resume_recovery_step_is_not_skipped_silently(resume_coord, monkeypatch, tmp_path, broken):
    recover, error_type, message = broken(resume_coord, monkeypatch, tmp_path)
    with pytest.raises(error_type, match=message):
        await recover({"fixes": [], "warnings": []})


@pytest.mark.asyncio
async def test_resume_no_pending_is_noop(resume_coord):
    resume_coord.writeback._resumed_from["is_resume"] = True
    resume_coord.shared_state.pending_targeted_build = {}
    report = await resume_coord.writeback._resume_consistency_pass()
    assert not any(isinstance(f, dict) and f.get("kind") == "reclaimed_pending_targeted_build" for f in report["fixes"])


@pytest.mark.asyncio
async def test_an_unconfirmed_build_records_the_group_not_the_dead_root(tmp_path, monkeypatch):
    """The operator's lead has to outlive the process that failed.

    ``spawn_build`` detaches the build with ``start_new_session`` and resolves
    the group it leads onto the handle. ``proc.pid`` stops naming anything the
    moment that root exits, which on this path it may well have; the group id
    still names the group while any member of it runs.

    Nothing probes it -- the lane is retained either way -- but it is printed to
    whoever has to clear that lane by hand, and a number that named a dead root
    would send them nowhere.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    from hyperloom.orchestrator.actions.executors import targeted_build_executor as tbe
    from hyperloom.orchestrator.enablement.runtime.targeted_build import BuildHandle
    from hyperloom.orchestrator.loop.sub_agent_runner import ExecutionCleanupUnconfirmed, RunnerContext

    action = _action(attempt_root=str(tmp_path))
    handle = BuildHandle(
        action,
        str(tmp_path),
        str(tmp_path / "aiter_jit"),
        str(tmp_path / "build.log"),
        SimpleNamespace(pid=4242, poll=lambda: 0, wait=lambda **_kw: 0),
        4242,
        1717,
    )
    monkeypatch.setattr(tbe, "spawn_build", lambda *a, **kw: handle)
    monkeypatch.setattr(tbe, "ensure_build_dead", lambda _build: False)
    monkeypatch.setattr(tbe, "default_budget_sec", lambda _component: 0.05)
    monkeypatch.setattr(tbe.TargetedBuildExecutor, "_record_result", Mock())
    ctx = RunnerContext(
        task=SimpleNamespace(task_id="pgid-build", params=action.to_state()),
        lease=None,
        extra={"session_dir": str(tmp_path)},
    )

    with pytest.raises(ExecutionCleanupUnconfirmed) as caught:
        await tbe.TargetedBuildExecutor()(ctx)

    assert caught.value.tree_pgid == 1717
