# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for specialist_subprocess process teardown: the SIGTERM/SIGKILL ``_kill`` ladder."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from concurrent.futures import CancelledError
from unittest.mock import Mock

from hyperloom.common.deadline import Deadline

import pytest

from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope
from hyperloom.orchestrator.loop.sub_agent_runner import ExecutionCleanupUnconfirmed

from hyperloom.common.agent_isolation import AccountIsolation
from hyperloom.orchestrator.specialists import launch_env
from hyperloom.orchestrator.specialists import subprocess_ as ss
from hyperloom.orchestrator.specialists.launch_env import PreparedLaunch
from hyperloom.orchestrator.specialists.subprocess_ import (
    SpecialistSubprocessDispatcher,
    SpecialistSpawn,
)


@pytest.mark.asyncio
async def test_pre_cancelled_specialist_never_starts(monkeypatch, tmp_path):
    scope = CancelScope()
    scope.cancel(reason="session_stopped")
    local_spawn = Mock(side_effect=AssertionError("cancelled specialist must not spawn"))
    dispatcher = ss.SpecialistSubprocessDispatcher(ss.SpecialistSubprocessConfig(agent_backend="claude"))
    monkeypatch.setattr(dispatcher, "_build_claude_cmd", lambda **kw: ["unused"])
    monkeypatch.setattr(ss.subprocess, "Popen", local_spawn)
    with use_cancel_scope(scope), pytest.raises(CancelledError, match="session_stopped"):
        await dispatcher.run(
            SpecialistSpawn(
                task_id="cancelled-specialist",
                workspace=tmp_path,
                worktree=None,
                worktree_base=None,
                system_prompt="system",
                user_prompt="user",
            )
        )
    local_spawn.assert_not_called()
    assert not scope.has_listeners


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["running", "done_grace"])
@pytest.mark.parametrize("confirmed_dead", [True, False])
@pytest.mark.parametrize("cancel_kind", ["scope", "direct"])
async def test_specialist_scope_cancel_stops_before_harvest(monkeypatch, tmp_path, phase, confirmed_dead, cancel_kind):
    scope = CancelScope()
    dispatcher = ss.SpecialistSubprocessDispatcher(
        ss.SpecialistSubprocessConfig(agent_backend="claude", poll_interval_seconds=0.01)
    )
    run = None
    entered = asyncio.Event()
    kill = Mock()
    harvest = Mock(side_effect=AssertionError("cancelled specialist must not harvest results"))
    monkeypatch.setattr(dispatcher, "_collect_patches", harvest)
    monkeypatch.setattr(dispatcher, "_build_claude_cmd", lambda **kw: ["unused"])
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
                    SpecialistSpawn(
                        task_id="running-specialist",
                        workspace=tmp_path,
                        worktree=None,
                        worktree_base=None,
                        system_prompt="system",
                        user_prompt="user",
                    )
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
        kill.assert_called_once_with(child)
        assert (child.poll() is not None) is confirmed_dead
        harvest.assert_not_called()
        assert not scope.has_listeners
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=3)
        child.stdin.close()
        if run is not None:
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)


class _SameUidRunAs:
    """A demoted launch whose account is the test's own uid, so no privilege is needed to run it."""

    user = "specialist"
    uid = os.getuid()

    def command(self, cmd):
        return list(cmd)

    def popen_kwargs(self):
        return {}


@pytest.mark.skipif(sys.platform == "win32", reason="procfs is Linux-only")
@pytest.mark.skipif(os.getuid() == 0, reason="pid 1 has to belong to another uid")
def test_demoted_reap_signals_only_pids_the_specialist_account_owns(monkeypatch):
    """The account can write into the scan's stdout, so a pid it prints is believed only if that account owns it."""
    from hyperloom_common.proctree import Tree

    own = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    roots: list[list[int]] = []
    monkeypatch.setattr(launch_env, "LIST_MARKED_SCRIPT", f"echo 1; echo {own.pid}")
    monkeypatch.setattr(launch_env, "collect_tree", lambda pids: roots.append(list(pids)) or Tree((), (), frozenset()))
    monkeypatch.setattr(launch_env, "kill_tree", lambda tree: True)
    try:
        PreparedLaunch(AccountIsolation(_SameUidRunAs())).reap_marked("mark")
    finally:
        own.kill()
        own.wait(timeout=5)
    assert roots == [[own.pid]]


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="procfs is Linux-only")
@pytest.mark.parametrize("demoted", [False, True], ids=["app_user", "demoted"])
async def test_natural_exit_reaps_detached_descendant(monkeypatch, tmp_path, demoted):
    """A heartbeat loop detached before the CLI exits does not outlive the specialist; another run's loop does."""
    import os
    import signal
    import time
    from hyperloom_common.proctree import running

    pid_path = tmp_path / "heartbeat.pid"
    child_code = (
        "import pathlib, subprocess, sys\n"
        f"pathlib.Path({str(pid_path)!r}).write_text(subprocess.run("
        "['sh','-c','setsid nohup \"$0\" -c \"$1\" </dev/null >/dev/null 2>&1 & echo $!',"
        "sys.executable,'import time; time.sleep(300)'],"
        "capture_output=True,text=True,check=True).stdout.strip())\n"
    )
    real_popen = ss.subprocess.Popen

    def spawn(cmd, *args, **kwargs):
        if cmd != ["unused"]:
            return real_popen(cmd, *args, **kwargs)
        child = real_popen(
            [sys.executable, "-c", child_code],
            env=kwargs["env"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        child.wait(timeout=5)
        return child

    monkeypatch.setattr(ss.subprocess, "Popen", spawn)
    if demoted:
        monkeypatch.setattr(ss, "resolve_launch_isolation", lambda: AccountIsolation(_SameUidRunAs()))
        monkeypatch.setattr(launch_env, "_demote", lambda *args: {})
        # The app's uid cannot read a demoted child's environ.
        monkeypatch.setattr(launch_env, "marked_pids", lambda key, value: [])
    dispatcher = SpecialistSubprocessDispatcher(
        ss.SpecialistSubprocessConfig(agent_backend="claude", poll_interval_seconds=0.05)
    )
    monkeypatch.setattr(dispatcher, "_build_claude_cmd", lambda **kw: ["unused"])
    monkeypatch.setattr(dispatcher, "_collect_patches", lambda *args, **kwargs: ([], {}))
    bystander = real_popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        env={**os.environ, "HYPERLOOM_REAP_MARK": "another-run"},
        start_new_session=True,
    )
    heartbeat_pid = 0
    try:
        result = await dispatcher.run(
            SpecialistSpawn(
                task_id="detached-heartbeat",
                workspace=tmp_path,
                worktree=None,
                worktree_base=None,
                system_prompt="system",
                user_prompt="user",
            )
        )
        deadline = time.monotonic() + 5
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        heartbeat_pid = int(pid_path.read_text())
        assert result.exit_code == 0
        assert running(heartbeat_pid) is False
        assert running(bystander.pid) is True
    finally:
        if heartbeat_pid and running(heartbeat_pid):
            os.kill(heartbeat_pid, signal.SIGKILL)
        bystander.kill()
        bystander.wait(timeout=5)


@pytest.mark.asyncio
async def test_a_sweep_that_never_returns_does_not_hold_back_the_done_file(monkeypatch, tmp_path, caplog):
    """The specialist wrote its done file and exited; a straggler sweep stuck in the kernel must not keep it unbooked."""
    import threading

    done = {"intent_type": "specialist_done", "payload": {"summary": "finished", "proposal_set": []}}
    child_code = (
        f"import pathlib; pathlib.Path({str(tmp_path / 'specialist_done.json')!r}).write_text({json.dumps(done)!r})"
    )
    real_popen = ss.subprocess.Popen

    def spawn(cmd, *args, **kwargs):
        return real_popen([sys.executable, "-c", child_code], stdout=subprocess.DEVNULL, start_new_session=True)

    stuck = threading.Event()
    monkeypatch.setattr(ss.subprocess, "Popen", spawn)
    monkeypatch.setattr(PreparedLaunch, "reap_marked", lambda self, mark: stuck.wait(30))
    monkeypatch.setattr(launch_env, "STRAGGLER_SWEEP_BOUND_SEC", 0.2)
    dispatcher = SpecialistSubprocessDispatcher(
        ss.SpecialistSubprocessConfig(agent_backend="claude", poll_interval_seconds=0.05)
    )
    monkeypatch.setattr(dispatcher, "_build_claude_cmd", lambda **kw: ["unused"])
    monkeypatch.setattr(dispatcher, "_collect_patches", lambda *args, **kwargs: ([], {}))
    try:
        with caplog.at_level("WARNING", logger=launch_env.log.name):
            result = await asyncio.wait_for(
                dispatcher.run(
                    SpecialistSpawn(
                        task_id="stuck-sweep",
                        workspace=tmp_path,
                        worktree=None,
                        worktree_base=None,
                        system_prompt="system",
                        user_prompt="user",
                    )
                ),
                10,
            )
    finally:
        stuck.set()
    assert result.done_payload["summary"] == "finished"
    assert result.exit_code == 0
    assert "stuck-sweep: the sweep for processes its launch left behind outlasted" in caplog.text


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
    from hyperloom_common.proctree import Tree

    proc = Mock(pid=4242)
    proc.poll.return_value = None
    monkeypatch.setattr(ss, "collect_tree", Mock(return_value=Tree(((4242, 1), (4243, 2)), (4242,), frozenset({4242}))))
    monkeypatch.setattr(ss, "kill_tree", Mock(return_value=confirmed_dead))
    if confirmed_dead:
        assert SpecialistSubprocessDispatcher._kill(proc) is True
    else:
        with pytest.raises(ExecutionCleanupUnconfirmed, match="tree cleanup unconfirmed"):
            SpecialistSubprocessDispatcher._kill(proc)


@pytest.mark.parametrize("failure", [OSError("procfs unreadable"), subprocess.TimeoutExpired("specialist", 1)])
def test_kill_does_not_confirm_failed_collection_or_wait(monkeypatch, failure):
    from hyperloom_common.proctree import Tree

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
    from hyperloom_common.proctree import Tree

    proc = Mock(pid=4242)
    proc.poll.return_value = None
    monkeypatch.setattr(ss, "collect_tree", Mock(return_value=Tree((), (), frozenset({4242}))))
    kill = Mock()
    monkeypatch.setattr(ss, "kill_tree", kill)
    with pytest.raises(ExecutionCleanupUnconfirmed, match="tree cleanup unconfirmed"):
        SpecialistSubprocessDispatcher._kill(proc)
    kill.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["deadline", "silence"])
async def test_reap_loop_kill_left_unconfirmed_raises_the_kills_error_carrying_the_decision(
    monkeypatch, tmp_path, ending
):
    """A caller of the base class sees the kill's own message and group; the loop's decision rides on the raise."""

    def doubt(proc):
        raise ExecutionCleanupUnconfirmed(f"specialist pid={proc.pid}: tree cleanup unconfirmed", tree_pgid=proc.pid)

    monkeypatch.setattr(SpecialistSubprocessDispatcher, "_kill", staticmethod(doubt))
    monkeypatch.setattr(ss, "cli_alive", lambda *args, **kwargs: ending == "deadline")
    dispatcher = SpecialistSubprocessDispatcher(
        ss.SpecialistSubprocessConfig(
            poll_interval_seconds=0.01, heartbeat_stale_seconds=0.0 if ending == "silence" else 300.0
        )
    )
    proc = Mock(pid=4242, returncode=None)
    proc.poll.return_value = None
    with pytest.raises(ExecutionCleanupUnconfirmed, match=r"^specialist pid=4242: tree cleanup unconfirmed$") as caught:
        await dispatcher._reap_loop(
            proc,
            ss.ReapWatch(workspace=tmp_path, done_files=()),
            deadline=Deadline.after(0.0 if ending == "deadline" else 600.0),
            started=time.monotonic(),
        )
    assert isinstance(caught.value, ss.ReapKillUnconfirmed)
    assert caught.value.tree_pgid == 4242
    outcome = caught.value.outcome
    assert (outcome["timed_out"], outcome["stale_heartbeat"]) == (ending == "deadline", ending == "silence")
    assert ("at its deadline" if ending == "deadline" else "heartbeat stale") in outcome["error"]


class _NeverExits:
    pid = -1
    returncode = None

    def poll(self):
        return None


@pytest.mark.parametrize("done_file", [False, True])
async def test_a_cancel_is_noticed_within_the_stop_gate_poll_not_the_reap_poll(tmp_path, done_file):
    # The dispatcher's cancel grace budgets one STOP_GATE_POLL_SECONDS for noticing, so neither the reap poll nor the
    # done-file grace may hold a cancelled specialist longer than that before its teardown starts.
    from hyperloom_common.proctree import STOP_GATE_POLL_SECONDS
    from hyperloom.orchestrator.specialists.reap_watch import ReapWatch

    done = tmp_path / "done.json"
    if done_file:
        done.write_text("{}", encoding="utf-8")
    # Without a done-file the loop sits in its reap poll; with one it sits in the done-file grace.
    poll = 0.01 if done_file else 30.0
    dispatcher = SpecialistSubprocessDispatcher(ss.SpecialistSubprocessConfig(poll_interval_seconds=poll))
    scope = CancelScope()
    watch = ReapWatch(workspace=tmp_path, done_files=(done,), task_id="t")
    with use_cancel_scope(scope):
        reaping = asyncio.create_task(
            dispatcher._reap_loop(_NeverExits(), watch, deadline=None, started=time.monotonic())
        )
        await asyncio.sleep(0.2)
        cancelled_at = time.monotonic()
        scope.cancel(reason="coordinator_stop")
        with pytest.raises(CancelledError):
            await asyncio.wait_for(reaping, timeout=5.0)
    assert time.monotonic() - cancelled_at <= STOP_GATE_POLL_SECONDS + 0.25
