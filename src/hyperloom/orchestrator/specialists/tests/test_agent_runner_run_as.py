# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local agent runner in each isolation tier: unconfined launches as before, an account demotes, a bubblewrap sandbox
wraps the CLI, and a launch that must be isolated and cannot be is refused."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom_common.bwrap import BwrapProbe
from hyperloom.common.agent_isolation import AccountIsolation, BwrapIsolation, NoIsolation
from hyperloom.common.env_safety import RunAs
from hyperloom.common import agent_isolation
from hyperloom_common.proctree import REAP_MARK_ENV
from hyperloom.orchestrator.specialists import agent_runner
from hyperloom.orchestrator.specialists.agent_runner import AgentConfigError
from hyperloom.orchestrator.specialists.launch_env import (
    CLAUDE_LOGIN_REFUSAL,
    CLAUDE_LOGIN_UNCONFINED,
    CODEX_BYPASS_FLAG,
    LaunchPlace,
    LaunchStatus,
    SandboxedProcess,
)
from hyperloom.orchestrator.specialists.tests._agent_fixtures import make_runner, make_spec, wait_finished

_USER_ENV = "HYPERLOOM_SPECIALIST_USER"
_TODAYS_KWARGS = {"stdin", "stdout", "stderr", "env", "cwd", "start_new_session"}
_DEMOTE_KWARGS = {"user", "group", "extra_groups", "umask"}


class _Refused(OSError):
    """The recording ``Popen`` never starts a child."""


def _recording_popen(calls: list[tuple[list[str], dict[str, Any]]]):
    def popen(argv: list[str], **kwargs: Any) -> Any:
        calls.append((list(argv), kwargs))
        raise _Refused("spawn refused by test")

    return popen


def _modes(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p): (p.lstat().st_mode, p.lstat().st_gid) for p in [root, *root.rglob("*")]}


async def _launch(tmp_path: Path, **spec_overrides: Any) -> tuple[Path, list[tuple[list[str], dict[str, Any]]]]:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    runner = make_runner(tmp_path / "session", popen=_recording_popen(calls))
    workspace = tmp_path / "w"
    spec_overrides.setdefault("cwd", workspace / "worktree")
    spec = make_spec(workspace, **spec_overrides)
    with pytest.raises(_Refused):
        await runner.launch(spec)
    return workspace, calls


@pytest.fixture
def demote_to_self_group(monkeypatch: pytest.MonkeyPatch) -> RunAs:
    """A resolvable user whose group is the test's own, so the share runs without root."""
    gid = os.getgid()
    run_as = RunAs(user="specialist", uid=os.getuid() + 1, gid=gid, groups=(gid,), home="/home/specialist")
    monkeypatch.setenv(_USER_ENV, "specialist")
    monkeypatch.setattr(agent_isolation, "resolve_run_as", lambda name: run_as)
    monkeypatch.setattr(agent_isolation, "missing_launch_capabilities", list)
    return run_as


_FAKE_BWRAP = "/opt/fake/bwrap"


def _tier(name: str) -> Any:
    gid = os.getgid()
    run_as = RunAs(user="specialist", uid=os.getuid() + 1, gid=gid, groups=(gid,), home="/home/specialist")
    tiers = {
        "none": NoIsolation(("no tier here",)),
        "account": AccountIsolation(run_as),
        "bwrap": BwrapIsolation(BwrapProbe(_FAKE_BWRAP, (0, 13, 1), ())),
    }
    return tiers[name]


@pytest.mark.parametrize("tier", ["none", "account", "bwrap"])
async def test_each_tier_starts_the_cli_its_own_way(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier: str) -> None:
    monkeypatch.setattr(agent_runner, "_launch_isolation", lambda: _tier(tier))
    model = tmp_path / "model"
    model.mkdir()

    workspace, calls = await _launch(tmp_path, env={"CLAUDE_CODE_OAUTH_TOKEN": "tok"}, readable=(model,))

    [(argv, kwargs)] = calls
    env = kwargs["env"]
    if tier == "bwrap":
        [status_fd] = kwargs.pop("pass_fds")
        assert argv[0] == _FAKE_BWRAP and argv[argv.index("--json-status-fd") + 1] == str(status_fd)
        assert REAP_MARK_ENV not in env and argv[argv.index("--setenv") + 1] == REAP_MARK_ENV
        assert argv[argv.index("--chdir") + 1] == str(workspace / "worktree")
        assert argv[argv.index(str(model)) - 1] == "--ro-bind-try"
        cli = argv[argv.index("--") + 1 :]
    else:
        assert env[REAP_MARK_ENV] and str(model) not in argv
        cli = argv
    expected = _TODAYS_KWARGS | (_DEMOTE_KWARGS if tier == "account" else set())
    assert kwargs.keys() == expected
    staged = workspace / ("runtime" if tier == "none" else "") / "specialist_mcp.json"
    assert cli[-2:] == ["--mcp-config", str(staged)]


async def test_a_gpu_session_is_placed_with_the_gpu_devices_and_the_git_metadata_its_repository_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_runner, "_launch_isolation", lambda: _tier("bwrap"))
    places: list[LaunchPlace] = []
    real = agent_runner.prepare_launch

    def prepare(isolation: Any, place: LaunchPlace, *args: Any) -> Any:
        places.append(place)
        return real(isolation, place, *args)

    monkeypatch.setattr(agent_runner, "prepare_launch", prepare)
    admin = tmp_path / "repo" / ".git" / "worktrees" / "worktree"

    await _launch(tmp_path, env={"CLAUDE_CODE_OAUTH_TOKEN": "tok"}, gpus=1, git_admin=admin)
    await _launch(tmp_path / "cpu", env={"CLAUDE_CODE_OAUTH_TOKEN": "tok"})

    assert [(place.gpu, place.git_admin) for place in places] == [(True, admin), (False, None)]


async def test_a_sandbox_launch_whose_policy_cannot_be_used_fails_as_a_config_error_and_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_runner, "_launch_isolation", lambda: _tier("bwrap"))
    monkeypatch.setattr(
        agent_runner, "prepare_launch", lambda isolation, *_args: agent_runner.PreparedLaunch(isolation)
    )
    calls: list[tuple[list[str], dict[str, Any]]] = []
    runner = make_runner(tmp_path / "session", popen=_recording_popen(calls))

    with pytest.raises(AgentConfigError, match="no sandbox policy"):
        await runner.launch(make_spec(tmp_path / "w", env={"CLAUDE_CODE_OAUTH_TOKEN": "tok"}))
    assert calls == []


async def test_a_finished_session_releases_what_its_launch_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sandbox's scratch (its /tmp and /dev/shm) and status descriptor go once its session ends."""
    monkeypatch.setattr(agent_runner, "_launch_isolation", lambda: _tier("none"))
    released: list[bool] = []
    real = agent_runner.prepare_launch

    def prepare(*args: Any) -> Any:
        prepared = real(*args)
        monkeypatch.setattr(prepared, "release", lambda: released.append(True))
        return prepared

    monkeypatch.setattr(agent_runner, "prepare_launch", prepare)
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(tmp_path / "w", code="pass"))
    await wait_finished(runner, handle)
    await runner.collect(handle)

    assert released == [True]


async def test_a_sandboxed_sessions_liveness_looks_below_the_agents_own_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bubblewrap's pid has the sandbox's init and the CLI below it, which would read as a tool the CLI waits on."""

    class Sandboxed(SandboxedProcess):
        agent_pid = 4242

    asked: list[int] = []

    def alive(_workspace: Path, pid: int, **_kwargs: Any) -> bool:
        asked.append(pid)
        return False

    monkeypatch.setattr(agent_runner, "cli_alive", alive)
    runner = make_runner(tmp_path / "session", wall=lambda: 1e9)
    proc = Sandboxed(SimpleNamespace(pid=7, poll=lambda: None), "mark", LaunchStatus())
    session = SimpleNamespace(spec=SimpleNamespace(workspace=tmp_path), started_unix=0.0, proc=proc)

    assert await runner._silent(session, None)
    assert asked == [4242]


async def test_a_claude_without_a_login_a_sandbox_can_show_starts_unconfined_where_it_need_not_be_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_runner, "_launch_isolation", lambda: _tier("bwrap"))

    _workspace, calls = await _launch(tmp_path)

    [(argv, kwargs)] = calls
    assert _FAKE_BWRAP not in argv and "pass_fds" not in kwargs  # the CLI itself, not a sandbox
    assert agent_isolation.session_isolation_lines() == (
        f"agent isolation: 1 agent launch ran unconfined: {CLAUDE_LOGIN_UNCONFINED}",
    )


async def test_a_claude_without_a_login_a_sandbox_can_show_is_refused_before_anything_starts_where_it_must_be_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_runner, "_launch_isolation", lambda: _tier("bwrap"))
    monkeypatch.setattr(agent_isolation._STATE, "hosted", True)
    calls: list[tuple[list[str], dict[str, Any]]] = []
    runner = make_runner(tmp_path / "session", popen=_recording_popen(calls))

    with pytest.raises(AgentConfigError) as refused:
        await runner.launch(make_spec(tmp_path / "w"))

    assert str(refused.value) == CLAUDE_LOGIN_REFUSAL and calls == []


async def test_a_sandboxed_codex_runs_with_its_own_sandbox_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_runner, "_launch_isolation", lambda: _tier("bwrap"))
    codex = ("codex", "exec", "--json", "--sandbox", "workspace-write", "-")

    _workspace, calls = await _launch(tmp_path, argv=codex, backend="codex", mcp=False)

    [(argv, _kwargs)] = calls
    assert argv[argv.index("--") + 1 :] == ["codex", "exec", CODEX_BYPASS_FLAG, "--json", "-"]


@pytest.mark.parametrize("value", [None, "", "   "])
async def test_an_unset_or_blank_user_launches_exactly_as_today(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(_USER_ENV, raising=False)
    else:
        monkeypatch.setenv(_USER_ENV, value)
    probe = tmp_path / "probe"
    make_spec(probe, cwd=probe / "worktree")
    before = _modes(probe)

    workspace, calls = await _launch(tmp_path)

    [(argv, kwargs)] = calls
    assert kwargs.keys() == _TODAYS_KWARGS
    assert argv[-2:] == ["--mcp-config", str(workspace / "runtime" / "specialist_mcp.json")]
    assert not (workspace / "specialist_mcp.json").exists()
    assert "HOME" in kwargs["env"] and kwargs["env"]["HOME"] == str(workspace)
    assert "GIT_CONFIG_COUNT" not in kwargs["env"]
    expected = {k.replace(str(probe), str(workspace)): v for k, v in before.items()}
    assert {k: v for k, v in _modes(workspace).items() if k in expected} == expected


async def test_a_resolvable_user_gets_the_demote_kwargs_and_a_staged_mcp_config(
    tmp_path: Path, demote_to_self_group: RunAs
) -> None:
    workspace, calls = await _launch(tmp_path)

    [(argv, kwargs)] = calls
    assert kwargs.keys() == _TODAYS_KWARGS | _DEMOTE_KWARGS
    assert kwargs["user"] == demote_to_self_group.uid and kwargs["group"] == demote_to_self_group.gid
    staged = workspace / "specialist_mcp.json"
    assert argv[argv.index("--mcp-config") + 1] == str(staged)
    assert staged.read_text() == (workspace / "runtime" / "specialist_mcp.json").read_text()
    assert kwargs["env"]["HOME"] == "/home/specialist"
    assert kwargs["env"]["GIT_CONFIG_VALUE_0"] == str(workspace / "worktree")
    assert workspace.stat().st_mode & stat.S_ISVTX
    assert (workspace / "worktree").stat().st_mode & stat.S_IWGRP
    for private in (workspace / "prompt.md", workspace / "process.log"):
        assert stat.S_IMODE(private.stat().st_mode) == 0o600


async def test_a_worktree_outside_the_workspace_is_shared_too(tmp_path: Path, demote_to_self_group: RunAs) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    os.chmod(outside, 0o700)

    await _launch(tmp_path, cwd=outside)

    assert outside.stat().st_mode & stat.S_IRWXG == stat.S_IRWXG


@pytest.mark.parametrize("planted", ["process.log", "runtime/specialist_mcp.json", ".claude-config/settings.json"])
async def test_a_link_a_previous_child_left_in_the_workspace_is_not_written_through(
    tmp_path: Path, demote_to_self_group: RunAs, planted: str
) -> None:
    target = tmp_path / "apps-file"
    link = tmp_path / "w" / planted
    link.parent.mkdir(parents=True)
    link.symlink_to(target)

    await _launch(tmp_path)

    assert not target.exists()


async def test_a_beat_does_not_follow_a_link_left_at_the_heartbeat(tmp_path: Path) -> None:
    beats: list = []
    target = tmp_path / "apps-file"
    spec = make_spec(tmp_path / "w", heartbeat_sink=beats.append)
    spec.heartbeat_path.symlink_to(target)
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(spec)
    try:
        beats[0]()
    finally:
        await runner.cancel(handle)
        await runner.collect(handle)

    assert not target.exists()


@pytest.mark.parametrize("user", ["no-such-user-d16", "root"])
async def test_an_unusable_user_fails_the_launch_without_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, user: str
) -> None:
    monkeypatch.setenv(_USER_ENV, user)
    calls: list[tuple[list[str], dict[str, Any]]] = []
    runner = make_runner(tmp_path / "session", popen=_recording_popen(calls))

    with pytest.raises(AgentConfigError, match="specialist isolation refused"):
        await runner.launch(make_spec(tmp_path / "w"))

    assert calls == []


async def test_a_user_the_app_cannot_signal_fails_the_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demote_to_self_group: RunAs
) -> None:
    monkeypatch.setattr(agent_isolation, "missing_launch_capabilities", lambda: ["CAP_KILL"])
    calls: list[tuple[list[str], dict[str, Any]]] = []
    runner = make_runner(tmp_path / "session", popen=_recording_popen(calls))

    with pytest.raises(AgentConfigError, match="CAP_KILL"):
        await runner.launch(make_spec(tmp_path / "w"))

    assert calls == []


def _account_exists(name: str) -> bool:
    import pwd

    try:
        pwd.getpwnam(name)
    except KeyError:
        return False
    return True


_LIVE_USER = "nobody"
_CHILD = """#!/bin/sh
id -u > proof_uid.txt
cat > proof_stdin.txt
cat {prompt} > proof_prompt.txt 2>/dev/null; echo $? > proof_prompt.rc
"""


@pytest.mark.skipif(
    os.geteuid() != 0 or not _account_exists(_LIVE_USER),
    reason=f"demoting needs root and an existing {_LIVE_USER!r} account; the test creates no users",
)
async def test_a_demoted_child_runs_as_the_user_writes_its_worktree_and_reads_only_its_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pwd
    import shutil
    import tempfile

    from hyperloom.orchestrator.specialists.tests._agent_fixtures import wait_finished

    monkeypatch.setenv(_USER_ENV, _LIVE_USER)
    base = Path(tempfile.mkdtemp(prefix="agent-runas."))
    try:
        base.chmod(0o755)  # the demoted account must traverse to the fake CLI and the workspace
        workspace = base / "w"
        cli = base / "cli.sh"
        cli.write_text(_CHILD.format(prompt=workspace / "prompt.md"), encoding="utf-8")
        cli.chmod(0o755)
        runner = make_runner(base / "session")
        spec = make_spec(workspace, argv=(str(cli),), cwd=workspace / "worktree", mcp=False)

        handle = await runner.launch(spec)
        status = await wait_finished(runner, handle)
        await runner.collect(handle)

        proof = {p.name: p.read_text(encoding="utf-8").strip() for p in (workspace / "worktree").glob("proof_*")}
        assert status.returncode == 0, (workspace / "process.log").read_text()
        assert proof["proof_uid.txt"] == str(pwd.getpwnam(_LIVE_USER).pw_uid)
        assert proof["proof_stdin.txt"] == "do the task"
        assert proof["proof_prompt.rc"] != "0"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _reading_popen(seen: list[bytes]):
    def popen(argv: list[str], **kwargs: Any) -> Any:
        seen.append(kwargs["stdin"].read())
        raise _Refused("spawn refused by test")

    return popen


async def test_the_prompt_reaches_the_cli_on_stdin(tmp_path: Path) -> None:
    seen: list[bytes] = []
    runner = make_runner(tmp_path / "session", popen=_reading_popen(seen))
    with pytest.raises(_Refused):
        await runner.launch(make_spec(tmp_path / "w"))
    assert seen == [b"do the task\n"]


async def test_a_symlinked_prompt_is_refused_not_followed(tmp_path: Path) -> None:
    seen: list[bytes] = []
    runner = make_runner(tmp_path / "session", popen=_reading_popen(seen))
    spec = make_spec(tmp_path / "w")
    secret = tmp_path / "runner-only.txt"
    secret.write_text("not for the agent\n", encoding="utf-8")
    (spec.workspace / "prompt.md").unlink()
    (spec.workspace / "prompt.md").symlink_to(secret)
    with pytest.raises(OSError) as refused:
        await runner.launch(spec)
    assert refused.value.errno == errno.ELOOP
    assert seen == []
