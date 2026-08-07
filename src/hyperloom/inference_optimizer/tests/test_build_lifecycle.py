# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the off-loop targeted-build lifecycle collaborator.

Drives the pump/reaper against the shared ``build_coord`` fake coordinator (real
TaskRegistry + ResourceLockManager + SharedState): the build runs off the tick
loop, ``build_lane`` serializes, timeout kills the group, idempotency avoids a
double spawn, and real components spawn the driver argv (writing plan.json)
while explicit build_commands pass through verbatim.
"""

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
        gap_id="g", framework="vllm", component="aiter", capability="fp4_moe", build_command=tuple(cmd)
    )
    base.update(kw)
    return TargetedBuildAction(**base)


def _real_action(**kw):
    base = dict(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe",
                ref="v0.1.0", repo_url="https://github.com/ROCm/aiter", gpu_arch="gfx950")
    base.update(kw)
    return TargetedBuildAction(**base)


def _fake_action(**kw):
    """Action with an explicit build_command (fake-builder path)."""
    base = dict(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe",
                build_command=(sys.executable, "-c", "print('fake')"))
    base.update(kw)
    return TargetedBuildAction(**base)


async def _drain(bl, *, ticks=200, sleep=0.05):
    """Pump + reap across simulated ticks until the build row is terminal."""
    for i in range(ticks):
        await bl._maybe_reap_targeted_build(tick=i)
        await bl._maybe_pump_targeted_build(tick=i)
        running = [t for t in await bl.tasks.by_state("running") if t.kind == "targeted_build"]
        succeeded = [t for t in await bl.tasks.by_state("succeeded") if t.kind == "targeted_build"]
        failed = [t for t in await bl.tasks.by_state("failed") if t.kind == "targeted_build"]
        if (succeeded or failed) and not running:
            return succeeded, failed
        time.sleep(sleep)
    raise AssertionError("build did not reach a terminal state")


# ---------------------------------------------------------------------------
# pump/reap lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enqueue_then_build_succeeds(build_coord, build_lifecycle):
    tid = await build_lifecycle.enqueue_targeted_build(_action([sys.executable, "-c", "print('ok')"]))
    assert tid
    succeeded, failed = await _drain(build_lifecycle)
    assert succeeded and not failed
    assert build_coord.shared_state.enablement.build_manifest
    assert build_coord.shared_state.pending_targeted_build == {}
    holders = await build_coord.locks.lane_holders()
    assert holders == {} or "build_lane" not in holders


@pytest.mark.asyncio
async def test_no_tick_freeze_build_runs_off_loop(build_coord, build_lifecycle):
    """Ticks keep advancing while a long build is in flight (no freeze)."""
    await build_lifecycle.enqueue_targeted_build(_action([sys.executable, "-c", "import time; time.sleep(3)"]))
    ticks_observed = 0
    started = time.monotonic()
    for i in range(20):
        t0 = time.monotonic()
        await build_lifecycle._maybe_reap_targeted_build(tick=i)
        await build_lifecycle._maybe_pump_targeted_build(tick=i)
        assert time.monotonic() - t0 < 1.0
        ticks_observed += 1
        running = [t for t in await build_lifecycle.tasks.by_state("running") if t.kind == "targeted_build"]
        if i >= 1:
            assert running, "build should be in flight without blocking the tick"
        if time.monotonic() - started > 1.0:
            break
    assert ticks_observed >= 2
    running = [t for t in await build_lifecycle.tasks.by_state("running") if t.kind == "targeted_build"]
    assert running


@pytest.mark.asyncio
async def test_nonzero_build_fails_and_records_failure(build_coord, build_lifecycle):
    await build_lifecycle.enqueue_targeted_build(_action([sys.executable, "-c", "import sys; sys.exit(2)"]))
    succeeded, failed = await _drain(build_lifecycle)
    assert failed and not succeeded
    lbf = build_coord.shared_state.enablement.last_build_failure
    assert lbf["failure_class"] == "compile_error"


@pytest.mark.asyncio
async def test_timeout_kills_and_marks_failed(build_coord, build_lifecycle):
    action = _action([sys.executable, "-c", "import time; time.sleep(600)"], build_budget_sec=1)
    await build_lifecycle.enqueue_targeted_build(action)
    succeeded, failed = await _drain(build_lifecycle, ticks=400)
    assert failed and not succeeded
    assert build_coord.shared_state.enablement.last_build_failure["failure_class"] == "timeout"
    holders = await build_coord.locks.lane_holders()
    assert holders.get("build_lane", 0) in (0, None) or "build_lane" not in holders


@pytest.mark.asyncio
async def test_build_lane_serializes_two_builds(build_coord, build_lifecycle):
    """Capacity-1 build_lane: the second build stays queued until the first ends."""
    a1 = await build_lifecycle.enqueue_targeted_build(_action([sys.executable, "-c", "import time; time.sleep(2)"], ref="v1"))
    a2 = await build_lifecycle.enqueue_targeted_build(_action([sys.executable, "-c", "print('two')"], ref="v2"))
    assert a1 != a2
    await build_lifecycle._maybe_pump_targeted_build(tick=0)
    running = [t.task_id for t in await build_lifecycle.tasks.by_state("running") if t.kind == "targeted_build"]
    queued = [t.task_id for t in await build_lifecycle.tasks.queued() if t.kind == "targeted_build"]
    assert len(running) == 1
    assert len(queued) == 1


@pytest.mark.asyncio
async def test_idempotent_enqueue_no_double_row(build_coord, build_lifecycle):
    a = _action([sys.executable, "-c", "print('x')"], ref="v1", gpu_arch="gfx950")
    t1 = await build_lifecycle.enqueue_targeted_build(a)
    t2 = await build_lifecycle.enqueue_targeted_build(a)
    assert t1 == t2
    all_builds = [t for t in await build_lifecycle.tasks.queued() if t.kind == "targeted_build"]
    assert len(all_builds) == 1


@pytest.mark.asyncio
async def test_build_lane_empty_conflict_does_not_block_serving(build_coord, build_lifecycle):
    """build_lane must not mutex the serving/benchmark lanes."""
    await build_lifecycle.enqueue_targeted_build(_action([sys.executable, "-c", "import time; time.sleep(2)"]))
    await build_lifecycle._maybe_pump_targeted_build(tick=0)
    lease = await build_coord.locks.try_acquire_many(
        ["benchmark_lane"], holder_id="bench", task_id="bench", action="bench", ttl_sec=60
    )
    assert lease is not None
    await build_coord.locks.release(lease)


# ---------------------------------------------------------------------------
# lifecycle -> driver wiring
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
async def test_pump_real_component_writes_plan_json(build_coord, build_lifecycle, tmp_path):
    """Pump for an aiter action with empty build_command writes plan.json before spawn."""
    action = _real_action()
    await build_lifecycle.enqueue_targeted_build(action)

    spawned_commands: list[list[str]] = []

    def _capture_spawn(action_inner, *, attempt_root, command, run=None, **kw):
        spawned_commands.append(list(command) if command else [])
        plan = Path(attempt_root) / "plan.json"
        assert plan.exists(), "plan.json must exist before spawn_build is called"
        assert json.loads(plan.read_text())["component"] == "aiter"
        raise RuntimeError("captured")

    from hyperloom.orchestrator.loop import build_lifecycle as blc_mod
    original = blc_mod.spawn_build
    blc_mod.spawn_build = _capture_spawn
    try:
        await build_lifecycle._maybe_pump_targeted_build(tick=0)
    except RuntimeError:
        # _capture_spawn raises after recording the command to stop the pump.
        pass
    finally:
        blc_mod.spawn_build = original

    assert spawned_commands
    cmd = spawned_commands[0]
    assert sys.executable in cmd
    assert "targeted_build" in " ".join(cmd)
    assert "--attempt-root" in cmd


@pytest.mark.asyncio
async def test_pump_fake_component_runs_literal_command(build_coord, build_lifecycle):
    """Pump for an action with explicit build_command uses that command directly."""
    action = _fake_action()
    await build_lifecycle.enqueue_targeted_build(action)

    spawned_commands: list[list[str]] = []

    def _capture_spawn(action_inner, *, attempt_root, command, run=None, **kw):
        spawned_commands.append(list(command) if command else [])
        raise RuntimeError("captured")

    from hyperloom.orchestrator.loop import build_lifecycle as blc_mod
    original = blc_mod.spawn_build
    blc_mod.spawn_build = _capture_spawn
    try:
        await build_lifecycle._maybe_pump_targeted_build(tick=0)
    except RuntimeError:
        # _capture_spawn raises after recording the command to stop the pump.
        pass
    finally:
        blc_mod.spawn_build = original

    assert spawned_commands
    assert spawned_commands[0] == list(action.build_command)


# ---------------------------------------------------------------------------
# resume recovery for an in-flight build
# ---------------------------------------------------------------------------

def _silent_plan():
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    from hyperloom.orchestrator.roles import ScriptedPlan

    return ScriptedPlan(
        turns=[], default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})
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
    resume_coord._resumed_from["is_resume"] = True
    import subprocess

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True
    )
    pgid = os.getpgid(proc.pid)

    attempt_root = Path(resume_coord.session_dir) / "enablement" / "builds" / "t-orphan"
    jit_dir = attempt_root / "aiter_jit"
    jit_dir.mkdir(parents=True, exist_ok=True)
    stale_lock = jit_dir / "lock"
    stale_lock.write_text("")
    old = time.time() - 3600
    os.utime(stale_lock, (old, old))

    action = TargetedBuildAction(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe")
    task, _ = await resume_coord.tasks.create_or_return_existing(
        kind="targeted_build", params=action.to_state(), idempotency_key="k-orphan", requires_lanes=["build_lane"]
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

    fix = next(
        f for f in report["fixes"] if isinstance(f, dict) and f["kind"] == "reclaimed_pending_targeted_build"
    )
    assert fix["task_id"] == task.task_id
    assert resume_coord.shared_state.pending_targeted_build == {}
    assert resume_coord.shared_state.enablement.last_build_failure["failure_class"] == "timeout"
    reclaimed = await resume_coord.tasks.get(task.task_id)
    assert reclaimed.state == "failed"
    assert not attempt_root.exists()
    for _ in range(40):
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


@pytest.mark.asyncio
async def test_resume_no_pending_targeted_build_is_noop(resume_coord):
    resume_coord._resumed_from["is_resume"] = True
    resume_coord.shared_state.pending_targeted_build = {}
    report = await resume_coord._resume_consistency_pass()
    assert not any(
        isinstance(f, dict) and f.get("kind") == "reclaimed_pending_targeted_build"
        for f in report["fixes"]
    )
