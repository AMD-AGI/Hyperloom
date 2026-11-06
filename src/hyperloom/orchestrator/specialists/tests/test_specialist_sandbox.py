# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialists launched in a real bubblewrap sandbox, through the dispatcher, the agent runner and the Ray actor's
process manager: what the agent reaches, how it is stopped, and how a sandbox that could not be set up reads."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom_common.git_safety import recorded_admin_dir, trusted_admin_dir, worktree_head
from hyperloom_common.tests._bwrap_host import usable_bwrap
from hyperloom.common.agent_isolation import BwrapIsolation
from hyperloom.common.deadline import Deadline
from hyperloom.common.proctree import REAP_MARK_ENV
from hyperloom.orchestrator.actions.executors._ray_serving import ManagedServerProcess
from hyperloom.orchestrator.specialists import agent_runner, launch_env
from hyperloom.orchestrator.specialists import subprocess_ as ss
from hyperloom.orchestrator.specialists.launch_env import LaunchPlace, SandboxedProcess, prepare_launch
from hyperloom.orchestrator.specialists.subprocess_ import (
    SpecialistSpawn,
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
)
from hyperloom.orchestrator.specialists.tests._agent_fixtures import (
    FORGES_ITS_GIT,
    make_runner,
    make_spec,
    wait_finished,
)
from hyperloom.orchestrator.specialists.worktrees import _setup_worktree
from hyperloom.orchestrator.tests._helpers import init_git_repo

pytestmark = pytest.mark.agent_isolation

#: The stand-in claude CLI: it records what it reaches, then writes its done file where it works.
_REACH = """#!/bin/sh
cat > got_prompt
{{
  echo "home=$HOME"
  echo "tmpdir=$TMPDIR"
  echo "mark=${{{mark}:+set}}"
  cat {secret} >/dev/null 2>&1 && echo "secret=read" || echo "secret=unreachable"
  test -e {app_tmp} && echo "app_tmp=seen" || echo "app_tmp=absent"
  kill -0 {app_pid} 2>/dev/null && echo "app=signalled" || echo "app=unreachable"
  echo "pids=$(ls /proc | grep -c '^[0-9]')"
  echo "model=$(cat {model}/config.json)"
}} > reach.txt
echo '{{"summary": "sandboxed", "proposal_set": []}}' > specialist_done.json
"""
_TRAPPED = """#!/bin/sh
trap 'echo stopped > "{out}"; exit 0' TERM
sleep 1000 &
wait
"""


def _script(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def sandboxed(monkeypatch: pytest.MonkeyPatch) -> BwrapIsolation:
    """Every specialist launch here goes through the host's real bubblewrap; the CLI's login is a token."""
    isolation = BwrapIsolation(usable_bwrap())
    monkeypatch.setattr(launch_env, "choose_agent_isolation", lambda **_kwargs: isolation)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "token-for-the-test")
    return isolation


def _dispatcher(cli: Path) -> SpecialistSubprocessDispatcher:
    return SpecialistSubprocessDispatcher(
        SpecialistSubprocessConfig(agent_backend="claude", claude_executable=str(cli), poll_interval_seconds=0.1)
    )


def _spawn(workspace: Path, **overrides: object) -> SpecialistSpawn:
    fields: dict = {
        "task_id": "t-sandboxed",
        "workspace": workspace,
        "worktree": None,
        "worktree_base": None,
        "system_prompt": "system",
        "user_prompt": "the task",
        "deadline": Deadline.after(60.0),
    }
    fields.update(overrides)
    return SpecialistSpawn(**fields)


async def test_a_sandboxed_specialist_writes_its_workspace_reads_its_model_and_reaches_nothing_of_the_app(
    tmp_path: Path, sandboxed: BwrapIsolation
) -> None:
    secret = tmp_path / "app" / "state" / "exit.json"
    secret.parent.mkdir(parents=True)
    secret.write_text("APP-SECRET", encoding="utf-8")
    model = tmp_path / "models" / "llama"
    model.mkdir(parents=True)
    (model / "config.json").write_text("weights-config", encoding="utf-8")
    with tempfile.NamedTemporaryFile(prefix="hl-b3-app-") as app_tmp:
        cli = _script(
            tmp_path / "cli" / "bin" / "claude",
            _REACH.format(mark=REAP_MARK_ENV, secret=secret, app_tmp=app_tmp.name, app_pid=os.getpid(), model=model),
        )
        workspace = tmp_path / "ws"
        result = await _dispatcher(cli).run(_spawn(workspace, readable=(model,)))

    reach = dict(line.split("=", 1) for line in (workspace / "reach.txt").read_text(encoding="utf-8").splitlines())
    assert result.error == "" and result.exit_code == 0 and result.done_payload["summary"] == "sandboxed"
    assert (workspace / "got_prompt").read_text(encoding="utf-8") == "the task"
    assert reach == {
        "home": str(workspace / ".home"),
        "tmpdir": "/tmp",
        "mark": "set",
        "secret": "unreachable",
        "app_tmp": "absent",
        "app": "unreachable",
        "pids": reach["pids"],
        "model": "weights-config",
    }
    assert int(reach["pids"]) < 10


async def test_the_harvest_of_a_sandboxed_specialist_never_takes_the_git_dir_it_points_its_worktree_at(
    tmp_path: Path, sandboxed: BwrapIsolation
) -> None:
    """In the sandbox the agent is the app's own uid and writes its worktree's ``.git``: the admin dir it makes there
    passes every ownership check, so the app's git goes through the one the repository records."""
    repo, workspace = tmp_path / "repo", tmp_path / "ws"
    init_git_repo(repo, seed_file="mod.py", seed_text="old = 1\n")
    worktree, err = _setup_worktree(repo, workspace / "worktree", "specialist-t-sandboxed")
    assert worktree is not None, err
    admin = recorded_admin_dir(repo, worktree)
    head = worktree_head(worktree, admin=admin)
    cli = _script(
        tmp_path / "cli" / "bin" / "claude",
        FORGES_ITS_GIT.format(workspace=workspace, worktree=worktree, objects=repo / ".git" / "objects", head=head),
    )

    result = await _dispatcher(cli).run(_spawn(workspace, worktree=worktree, worktree_base=repo, worktree_admin=admin))

    assert trusted_admin_dir(worktree) == workspace / "forged.git" / "worktrees" / "w"
    assert (workspace / "git_seen").read_text(encoding="utf-8") == f"{head}\n"  # its repository is shown read-only
    assert result.error == "" and result.done_payload["summary"] == "forged"
    assert len(result.patches) == 1 and result.patch_roots == {result.patches[0]: str(repo)}
    patch = Path(result.patches[0]).read_text(encoding="utf-8")
    assert "--- a/mod.py\n+++ b/mod.py\n" in patch and "+new = 2" in patch


async def test_a_specialist_killed_at_its_deadline_runs_its_own_stop_first(
    tmp_path: Path, sandboxed: BwrapIsolation
) -> None:
    workspace = tmp_path / "ws"
    cli = _script(tmp_path / "cli" / "bin" / "claude", _TRAPPED.format(out=workspace / "trapped"))

    result = await _dispatcher(cli).run(_spawn(workspace, deadline=Deadline.after(1.0)))

    assert result.timed_out
    assert (workspace / "trapped").read_text(encoding="utf-8") == "stopped\n"
    assert result.exit_code == 0  # bubblewrap ended with the agent's own exit


async def test_a_sandboxed_cli_that_goes_silent_is_reaped_as_stale_not_taken_for_its_own_tool(
    tmp_path: Path, sandboxed: BwrapIsolation
) -> None:
    """Liveness looks below the agent's own process: bubblewrap's pid 1 and the CLI itself are not tools it waits on."""
    hang = f'#!/bin/sh\necho started\nexec {sys.executable} -I -S -c "import signal; signal.pause()"\n'
    cli = _script(tmp_path / "cli" / "bin" / "claude", hang)
    dispatcher = SpecialistSubprocessDispatcher(
        SpecialistSubprocessConfig(
            agent_backend="claude", claude_executable=str(cli), poll_interval_seconds=0.1, heartbeat_stale_seconds=1.0
        )
    )

    result = await dispatcher.run(_spawn(tmp_path / "ws", deadline=Deadline.after(30.0)))

    assert result.stale_heartbeat and not result.timed_out, result.error


async def test_an_agent_killed_by_a_signal_reads_as_bubblewraps_128_plus_n(
    tmp_path: Path, sandboxed: BwrapIsolation
) -> None:
    cli = _script(tmp_path / "cli" / "bin" / "claude", "#!/bin/sh\nkill -KILL $$\n")

    result = await _dispatcher(cli).run(_spawn(tmp_path / "ws"))

    assert result.exit_code == 137 and "sandbox" not in result.error


async def test_a_sandbox_that_cannot_be_set_up_fails_the_task_as_that_with_bubblewraps_line(
    tmp_path: Path, sandboxed: BwrapIsolation, monkeypatch: pytest.MonkeyPatch
) -> None:
    gone = tmp_path / "removed-before-the-launch"
    real = launch_env.prepare_launch

    def prepare(*args: object) -> launch_env.PreparedLaunch:
        prepared = real(*args)
        prepared.policy = replace(prepared.policy, writable=(*prepared.policy.writable, str(gone)))
        return prepared

    monkeypatch.setattr(ss, "prepare_launch", prepare)
    cli = _script(tmp_path / "cli" / "bin" / "claude", "#!/bin/sh\nexit 3\n")

    result = await _dispatcher(cli).run(_spawn(tmp_path / "ws"))

    assert result.exit_code is None
    assert result.error.startswith("specialist sandbox could not be set up: bwrap: Can't find source path")
    assert str(gone) in result.error


async def test_a_cancelled_sandboxed_cli_runs_its_own_stop_and_a_failed_setup_reads_as_that(
    tmp_path: Path, sandboxed: BwrapIsolation, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "w"
    cli = _script(tmp_path / "cli" / "bin" / "agent", _TRAPPED.format(out=workspace / "trapped"))
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(workspace, argv=(str(cli),), env={"CLAUDE_CODE_OAUTH_TOKEN": "t"}))
    try:
        await asyncio.sleep(0.5)
        await runner.cancel(handle)
        status = await wait_finished(runner, handle)
        await runner.collect(handle)
    finally:
        await runner.cancel(handle)
    assert isinstance(handle.session.proc, SandboxedProcess)
    assert (workspace / "trapped").read_text(encoding="utf-8") == "stopped\n" and status.returncode == 0

    real = launch_env.prepare_launch
    gone = str(tmp_path / "gone")

    def prepare(*args: object) -> launch_env.PreparedLaunch:
        prepared = real(*args)
        prepared.policy = replace(prepared.policy, writable=(*prepared.policy.writable, gone))
        return prepared

    monkeypatch.setattr(agent_runner, "prepare_launch", prepare)
    failed = await runner.launch(make_spec(tmp_path / "w2", argv=(str(cli),), env={"CLAUDE_CODE_OAUTH_TOKEN": "t"}))
    status = await wait_finished(runner, failed)
    await runner.collect(failed)
    assert status.returncode is None and "sandbox could not be set up: bwrap: Can't find source path" in status.error


def test_a_ray_actors_process_manager_runs_the_sandbox_and_its_status_reaches_this_host(
    tmp_path: Path, sandboxed: BwrapIsolation
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    for code, failure in ((5, ""), (0, "bwrap: Can't find source path")):
        env = {"PATH": "/usr/bin:/bin", REAP_MARK_ENV: f"ray-{code}"}
        prepared = prepare_launch(sandboxed, LaunchPlace(workspace), ["/bin/sh", "-c", f"exit {code}"], env)
        if failure:
            prepared.policy = replace(prepared.policy, writable=(*prepared.policy.writable, str(tmp_path / "gone")))
        spawn = prepared.actor(["/bin/sh", "-c", f"exit {code}"], env)
        log = workspace / f"process-{code}.log"
        manager = ManagedServerProcess()
        try:
            pid = manager.start(spawn.argv, env=spawn.env, cwd=str(workspace), log_path=str(log))
            proc = prepared.started(ss._RayLeaseProcess(manager, pid))
            deadline = time.monotonic() + 30
            while manager.exit_code() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert proc.setup_failure(log).startswith(failure) if failure else proc.setup_failure(log) == ""
            assert manager.exit_code() == (1 if failure else code), log.read_text(encoding="utf-8")
        finally:
            manager.stop()
            prepared.release()
