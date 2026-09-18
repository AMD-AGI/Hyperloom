# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for specialist_subprocess process teardown: the SIGTERM/SIGKILL ``_kill`` ladder."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from concurrent.futures import CancelledError
from types import SimpleNamespace
from unittest.mock import Mock

from hyperloom.common.deadline import Deadline

import pytest

from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope
from hyperloom.orchestrator.loop.sub_agent_runner import ExecutionCleanupUnconfirmed

from hyperloom.orchestrator.specialists import subprocess_ as ss
from hyperloom.orchestrator.specialists.subprocess_ import (
    SpecialistSubprocessDispatcher,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["local", "pending"])
async def test_pre_cancelled_specialist_never_starts(monkeypatch, tmp_path, backend):
    scope = CancelScope()
    scope.cancel(reason="session_stopped")
    local_spawn = Mock(side_effect=AssertionError("cancelled specialist must not spawn"))
    lease = Mock() if backend == "pending" else None
    if lease is not None:
        lease.start_async.side_effect = AssertionError("cancelled specialist must not start actor")
    dispatcher = ss.SpecialistSubprocessDispatcher(ss.SpecialistSubprocessConfig(agent_backend="claude"))
    monkeypatch.setattr(dispatcher, "_build_claude_cmd", lambda **kw: ["unused"])
    monkeypatch.setattr(ss.subprocess, "Popen", local_spawn)
    with use_cancel_scope(scope), pytest.raises(CancelledError, match="session_stopped"):
        await dispatcher.run(
            task_id="cancelled-specialist",
            workspace=tmp_path,
            worktree=None,
            worktree_base=None,
            system_prompt="system",
            user_prompt="user",
            max_turns=1,
            gpu_lease=lease,
        )
    local_spawn.assert_not_called()
    if lease is not None:
        lease.start_async.assert_not_called()
    assert not scope.has_listeners


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["running", "done_grace", "pending"])
@pytest.mark.parametrize("confirmed_dead", [True, False])
@pytest.mark.parametrize("cancel_kind", ["scope", "direct"])
async def test_specialist_scope_cancel_stops_before_harvest(monkeypatch, tmp_path, phase, confirmed_dead, cancel_kind):
    scope = CancelScope()
    dispatcher = ss.SpecialistSubprocessDispatcher(
        ss.SpecialistSubprocessConfig(agent_backend="claude", poll_interval_seconds=0.01)
    )
    child = None
    run = None
    entered = asyncio.Event()
    lease = Mock() if phase == "pending" else None
    kill = Mock()
    harvest = Mock(side_effect=AssertionError("cancelled specialist must not harvest results"))
    monkeypatch.setattr(dispatcher, "_collect_patches", harvest)
    monkeypatch.setattr(dispatcher, "_build_claude_cmd", lambda **kw: ["unused"])
    if lease is not None:

        def poll_started():
            entered.set()
            return None

        lease.poll_started.side_effect = poll_started
        lease.close.return_value = confirmed_dead
        monkeypatch.setattr(ss, "_RAY_PENDING_POLL_INTERVAL_SEC", 0.01)
    else:
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.readline()"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def spawn(*args, **kwargs):
            entered.set()
            return child

        def kill_child(proc):
            kill(proc)
            if confirmed_dead:
                proc.terminate()
                proc.wait(timeout=3)
            return confirmed_dead

        monkeypatch.setattr(ss.subprocess, "Popen", spawn)
        monkeypatch.setattr(dispatcher, "_kill", kill_child)
        if phase == "done_grace":
            (tmp_path / "specialist_done.json").write_text('{"summary": "not accepted"}', encoding="utf-8")
    try:
        with use_cancel_scope(scope):
            run = asyncio.create_task(
                dispatcher.run(
                    task_id="running-specialist",
                    workspace=tmp_path,
                    worktree=None,
                    worktree_base=None,
                    system_prompt="system",
                    user_prompt="user",
                    max_turns=1,
                    gpu_lease=lease,
                )
            )
        await asyncio.wait_for(entered.wait(), 3)
        await asyncio.sleep(0.03)
        assert scope.has_listeners
        if cancel_kind == "scope":
            scope.cancel(reason="session_stopped")
        else:
            run.cancel()
        cancelled_error = CancelledError if cancel_kind == "scope" else asyncio.CancelledError
        expected = cancelled_error if confirmed_dead else ExecutionCleanupUnconfirmed
        with pytest.raises(expected):
            await asyncio.wait_for(asyncio.shield(run), 3)
        if lease is not None:
            assert lease.close.call_count == (1 if confirmed_dead else 2)
        else:
            kill.assert_called_once_with(child)
            assert (child.poll() is not None) is confirmed_dead
        harvest.assert_not_called()
        assert not scope.has_listeners
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=3)
            child.stdin.close()
        if run is not None:
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)


def test_kill_exited_root_cannot_confirm_descendants(monkeypatch):
    proc = Mock(pid=4242)
    proc.poll.return_value = 0
    collect = Mock()
    monkeypatch.setattr(ss, "collect_tree", collect)
    with pytest.raises(ExecutionCleanupUnconfirmed, match="exited root"):
        SpecialistSubprocessDispatcher._kill(proc)
    collect.assert_not_called()


@pytest.mark.parametrize("confirmed_dead", [True, False])
def test_kill_reuses_tree_cleanup_confirmation(monkeypatch, confirmed_dead):
    from hyperloom.common.proctree import Tree

    proc = Mock(pid=4242)
    proc.poll.return_value = None
    tree = Tree(((4242, 1), (4243, 2)), (4242,), frozenset({4242}))
    collect = Mock(return_value=tree)
    kill = Mock(return_value=confirmed_dead)
    monkeypatch.setattr(ss, "collect_tree", collect)
    monkeypatch.setattr(ss, "kill_tree", kill)
    if confirmed_dead:
        confirmed = SpecialistSubprocessDispatcher._kill(proc)
        assert confirmed is True
        proc.wait.assert_called_once_with(timeout=1.0)
    else:
        with pytest.raises(ExecutionCleanupUnconfirmed, match="tree cleanup unconfirmed"):
            SpecialistSubprocessDispatcher._kill(proc)
        proc.wait.assert_not_called()
    collect.assert_called_once_with([4242])
    kill.assert_called_once_with(tree)


@pytest.mark.parametrize("failure", [OSError("procfs unreadable"), subprocess.TimeoutExpired("specialist", 1)])
def test_kill_does_not_confirm_failed_collection_or_wait(monkeypatch, failure):
    from hyperloom.common.proctree import Tree

    proc = Mock(pid=4242)
    proc.poll.return_value = None
    if isinstance(failure, OSError):
        collect = Mock(side_effect=failure)
    else:
        collect = Mock(return_value=Tree(((4242, 1),), (4242,), frozenset({4242})))
        proc.wait.side_effect = failure
    monkeypatch.setattr(ss, "collect_tree", collect)
    monkeypatch.setattr(ss, "kill_tree", Mock(return_value=True))
    with pytest.raises(ExecutionCleanupUnconfirmed, match="tree cleanup failed"):
        SpecialistSubprocessDispatcher._kill(proc)


def test_kill_missing_process_identity_cannot_confirm_tree(monkeypatch):
    from hyperloom.common.proctree import Tree

    proc = Mock(pid=4242)
    proc.poll.return_value = None
    monkeypatch.setattr(ss, "collect_tree", Mock(return_value=Tree((), (), frozenset({4242}))))
    kill = Mock()
    monkeypatch.setattr(ss, "kill_tree", kill)
    with pytest.raises(ExecutionCleanupUnconfirmed, match="tree cleanup unconfirmed"):
        SpecialistSubprocessDispatcher._kill(proc)
    kill.assert_not_called()


@pytest.mark.parametrize("confirmed_dead", [True, False])
def test_kill_ray_process_requires_cleanup_ack(confirmed_dead):
    lease = Mock()
    lease.stop.return_value = confirmed_dead
    lease.close.return_value = confirmed_dead
    lease.exit_code.return_value = None
    proc = ss._RayLeaseProcess(lease, 4242)
    if confirmed_dead:
        confirmed = SpecialistSubprocessDispatcher._kill(proc)
        assert confirmed is True
    else:
        with pytest.raises(ExecutionCleanupUnconfirmed, match="actor cleanup unconfirmed"):
            SpecialistSubprocessDispatcher._kill(proc)
    lease.stop.assert_called_once()
    assert lease.close.call_count == int(not confirmed_dead)


@pytest.mark.parametrize("exit_code", [None, 0, 7, -15])
def test_ray_reap_followup_ack_preserves_observed_exit_code(exit_code):
    actor = SimpleNamespace(closed=False)
    lease = Mock()
    lease.stop.return_value = False
    lease.exit_code.side_effect = lambda: None if actor.closed else exit_code

    def close():
        actor.closed = True
        return True

    lease.close.side_effect = close
    proc = ss._RayLeaseProcess(lease, 4242)
    confirmed = SpecialistSubprocessDispatcher._kill(proc)
    assert confirmed is True
    assert proc.returncode == exit_code
    lease.stop.assert_called_once()
    lease.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["timeout", "stale", "done", "natural"])
@pytest.mark.parametrize("exit_code", [None, 0, 7])
async def test_ray_followup_ack_preserves_reap_outcome(monkeypatch, tmp_path, phase, exit_code):
    dispatcher = ss.SpecialistSubprocessDispatcher(
        ss.SpecialistSubprocessConfig(
            agent_backend="claude", poll_interval_seconds=0.01, heartbeat_stale_seconds=1 if phase == "stale" else 300
        )
    )
    monkeypatch.setattr(dispatcher, "_build_claude_cmd", lambda **kw: ["unused"])
    monkeypatch.setattr(dispatcher, "_collect_patches", Mock(return_value=([], {})))
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(ss, "time", SimpleNamespace(monotonic=lambda: clock.now, time=lambda: 100.0))

    async def advance(_delay):
        clock.now += 31.0

    monkeypatch.setattr(ss.asyncio, "sleep", advance)
    lease = Mock()
    lease.poll_started.return_value = 4242
    actor = SimpleNamespace(closed=False, alive=phase != "natural")
    lease.is_alive.side_effect = lambda: actor.alive and not actor.closed
    lease.exit_code.side_effect = lambda: None if actor.closed else exit_code
    lease.stop.return_value = False

    def close():
        actor.closed = True
        return True

    lease.close.side_effect = close
    payload = {"summary": "preserved", "proposal_set": []}
    filename = "specialist_done.json" if phase in {"done", "natural"} else "specialist_done.partial.json"
    (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")
    result = await dispatcher.run(
        task_id="ray-reap-outcome",
        workspace=tmp_path,
        worktree=None,
        worktree_base=None,
        system_prompt="system",
        user_prompt="user",
        max_turns=1,
        gpu_lease=lease,
        deadline=Deadline.after(0 if phase == "timeout" else 600, now=100),
    )
    assert result.timed_out is (phase == "timeout")
    assert result.stale_heartbeat is (phase == "stale")
    if phase == "timeout":
        assert "deadline" in result.error
    elif phase == "stale":
        assert "heartbeat stale" in result.error
    else:
        assert result.error == ""
    assert result.done_payload["summary"] == "preserved"
    assert result.done_payload.get("_recovered_from_partial", False) is (phase in {"timeout", "stale"})
    if phase == "natural" and exit_code is None:
        from hyperloom.orchestrator.actions.executors._ray_serving import _RAY_ACTOR_DIED_RC

        assert result.exit_code == _RAY_ACTOR_DIED_RC
    else:
        assert result.exit_code == exit_code
    assert lease.stop.call_count == int(phase != "natural")
    assert lease.close.call_count == int(phase != "natural")
