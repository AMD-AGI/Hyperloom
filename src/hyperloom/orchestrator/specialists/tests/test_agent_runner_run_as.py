# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local agent runner under HYPERLOOM_SPECIALIST_USER: unset launches as before, set demotes, unusable refuses."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from hyperloom.common.env_safety import RunAs
from hyperloom.orchestrator.specialists import launch_env
from hyperloom.orchestrator.specialists.agent_runner import AgentConfigError
from hyperloom.orchestrator.specialists.tests._agent_fixtures import make_runner, make_spec

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
    monkeypatch.setattr(launch_env, "specialist_run_as", lambda: run_as)
    monkeypatch.setattr(launch_env, "missing_launch_capabilities", list)
    return run_as


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

    with pytest.raises(AgentConfigError, match="specialist user refused"):
        await runner.launch(make_spec(tmp_path / "w"))

    assert calls == []


async def test_a_user_the_app_cannot_signal_fails_the_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demote_to_self_group: RunAs
) -> None:
    monkeypatch.setattr(launch_env, "missing_launch_capabilities", lambda: ["CAP_KILL"])
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
