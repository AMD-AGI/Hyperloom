# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a specialist launch is readied in each isolation tier, with a stand-in bubblewrap (nothing is sandboxed here)."""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from hyperloom_common.bwrap import BwrapProbe
from hyperloom_common.git_safety import recorded_admin_dir, trusted_admin_dir
from hyperloom.common import agent_isolation
from hyperloom.common.agent_isolation import (
    AccountIsolation,
    BwrapIsolation,
    IsolationRefused,
    NoIsolation,
    session_isolation_lines,
)
from hyperloom.common.deadline import Deadline
from hyperloom.common.env_safety import RunAs
from hyperloom_common.proctree import REAP_MARK_ENV, running
from hyperloom.orchestrator.specialists import launch_env
from hyperloom.orchestrator.specialists import subprocess_ as ss
from hyperloom.orchestrator.specialists.agent_cli_config import SpecialistAgentUnavailableError
from hyperloom.orchestrator.specialists.agent_runner import AgentConfigError
from hyperloom.orchestrator.specialists.launch_env import (
    CLAUDE_LOGIN_REFUSAL,
    CLAUDE_LOGIN_UNCONFINED,
    CLAUDE_TOKEN_VARIABLES,
    CODEX_BYPASS_FLAG,
    LaunchPlace,
    LaunchStatus,
    PreparedLaunch,
    SandboxedProcess,
    bedrock_files,
    claude_login_isolation,
    prepare_launch,
    specialist_confinement,
    stage_mcp_config,
    without_codex_sandbox,
)
from hyperloom.orchestrator.specialists.tests._agent_fixtures import make_runner, make_spec, wait_finished

FAKE_BWRAP = "/opt/fake/bwrap"


def _bwrap(version: tuple[int, int, int] = (0, 13, 1)) -> BwrapIsolation:
    return BwrapIsolation(BwrapProbe(FAKE_BWRAP, version, ()))


def _place(tmp_path: Path, **overrides: Any) -> LaunchPlace:
    workspace = tmp_path / "ws"
    (workspace / "worktree").mkdir(parents=True, exist_ok=True)
    fields: dict[str, Any] = {"workspace": workspace, "worktree": workspace / "worktree"}
    fields.update(overrides)
    return LaunchPlace(**fields)


def _env(**extra: str) -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", REAP_MARK_ENV: "mark-1", **extra}


def test_every_launch_is_isolated_as_the_chooser_says_and_a_ray_actor_never_as_an_account(monkeypatch):
    asked: list[bool] = []
    monkeypatch.setattr(launch_env, "choose_agent_isolation", lambda *, allow_account: asked.append(allow_account))

    launch_env.resolve_launch_isolation(None)
    launch_env.resolve_launch_isolation(object())

    assert asked == [True, False]


def test_an_unconfined_launch_runs_the_command_as_it_is(tmp_path):
    prepared = prepare_launch(NoIsolation(("why",)), _place(tmp_path), ["claude", "--print"], env := _env())
    spawn = prepared.local(["claude", "--print"], env)

    assert (spawn.argv, spawn.env, spawn.kwargs) == (["claude", "--print"], env, {})
    assert prepared.started(proc := object()) is proc


def test_an_account_launch_starts_through_the_drop_tool_with_the_accounts_ids(tmp_path):
    gid = os.getgid()
    run_as = RunAs("specialist", os.getuid() + 1, gid, (gid,), "/home/specialist", drop_tool="/usr/bin/setpriv")
    env = _env()

    prepared = prepare_launch(AccountIsolation(run_as), _place(tmp_path), ["claude"], env)
    spawn = prepared.local(["claude"], env)

    assert spawn.argv[0] == "/usr/bin/setpriv" and spawn.argv[-1] == "claude"
    assert spawn.kwargs == run_as.popen_kwargs()
    assert spawn.env[REAP_MARK_ENV] == "mark-1" and spawn.env["HOME"] == "/home/specialist"


def test_a_sandboxed_launch_sets_its_mark_inside_only_and_passes_its_status_descriptor(tmp_path):
    place = _place(tmp_path)
    env = _env(HOME="/root", CLAUDE_CODE_OAUTH_TOKEN="token-value")
    prepared = prepare_launch(_bwrap(), place, ["claude", "--print"], env)
    try:
        spawn = prepared.local(["claude", "--print"], env)
        argv = spawn.argv
        [status_fd] = spawn.kwargs["pass_fds"]
        assert argv[0] == FAKE_BWRAP and argv[argv.index("--") + 1 :] == ["claude", "--print"]
        assert {"--unshare-user", "--unshare-pid", "--cap-drop"} <= set(argv)
        assert argv[argv.index("--json-status-fd") + 1] == str(status_fd)
        assert argv[argv.index("--setenv") + 1 : argv.index("--setenv") + 3] == [REAP_MARK_ENV, "mark-1"]
        assert REAP_MARK_ENV not in spawn.env and "token-value" not in argv
        assert argv[argv.index("--chdir") + 1] == str(place.worktree)
        assert ["--bind", str(place.workspace), str(place.workspace)] == argv[argv.index(str(place.workspace)) - 1 :][
            :3
        ]
        assert spawn.env["HOME"] == str(place.workspace / ".home") and spawn.env["TMPDIR"] == "/tmp"
        assert (place.workspace / ".home").is_dir()
        scratch = prepared.scratch
        assert scratch is not None and os.path.isdir(scratch.tmp) and os.path.isdir(scratch.shm)
    finally:
        prepared.release()
    assert not os.path.exists(scratch.tmp) and not os.path.exists(scratch.shm)


def test_a_sandbox_points_cli_state_and_caches_named_elsewhere_into_its_workspace(tmp_path):
    place = _place(tmp_path)
    inside = str(place.workspace / "mine")
    env = _env(
        HOME=str(place.workspace / "h"),
        CLAUDE_CONFIG_DIR="/root/.claude",
        TRITON_CACHE_DIR="/shared/triton",
        AITER_JIT_DIR=inside,
        TMP="/var/tmp",
    )
    prepared = prepare_launch(_bwrap(), place, ["claude"], env)
    prepared.release()

    assert env["HOME"] == str(place.workspace / "h")
    assert env["CLAUDE_CONFIG_DIR"] == str(place.workspace / ".claude-config")
    assert env["TRITON_CACHE_DIR"] == str(place.workspace / ".cache" / "triton")
    assert env["AITER_JIT_DIR"] == inside
    assert "TORCHINDUCTOR_CACHE_DIR" not in env  # unset: the tool's default is under the home or /tmp
    assert env["TMP"] == env["TMPDIR"] == "/tmp"


def test_a_sandboxed_gpu_task_gets_its_jit_caches_in_its_workspace_not_in_the_read_only_frameworks(tmp_path):
    """Unset, AITER builds its JIT modules in its installed tree, which the sandbox shows read-only; only a CPU task
    had them set (``apply_gpu_env``)."""
    place = _place(tmp_path, gpu=True)
    env = _env()
    launch_env.apply_gpu_env(env, place.workspace, (0,), None)
    prepare_launch(_bwrap(), place, ["claude"], env).release()

    cache = place.workspace / ".cache"
    assert env["AITER_JIT_DIR"] == env["INFERENCE_OPTIMIZER_AITER_JIT_DIR"] == str(cache / "aiter_jit")
    assert env["FLYDSL_RUNTIME_CACHE_DIR"] == str(cache / "flydsl")
    assert "TRITON_CACHE_DIR" not in env  # its default is under the sandbox's home

    unconfined = _env()
    prepare_launch(NoIsolation(("why",)), place, ["claude"], unconfined)
    assert "AITER_JIT_DIR" not in unconfined


def test_hugging_faces_default_directory_stays_visible_read_only_once_the_home_moves(tmp_path):
    home = tmp_path / "app-home"
    hub = home / ".cache" / "huggingface"
    (hub / "hub" / "models--org--m").mkdir(parents=True)
    (hub / "token").write_text("hf_secret", encoding="utf-8")
    place = _place(tmp_path)
    env = _env(HOME=str(home))

    prepared = prepare_launch(_bwrap(), place, ["claude"], env)
    prepared.release()

    assert env["HF_HOME"] == str(hub) and env["HOME"] == str(place.workspace / ".home")
    assert str(hub / "hub") in prepared.policy.read_only, "the cache is bound entry by entry, without the login"
    assert not [path for path in prepared.policy.read_only if str(hub / "token").startswith(path)]
    assert str(hub / "token") not in prepared.policy.masked, "a file is left out, not masked"
    named = _env(HOME=str(home), HF_HUB_CACHE=str(tmp_path / "hub"))
    prepare_launch(_bwrap(), place, ["claude"], named).release()
    assert "HF_HOME" not in named


def _cannot_search(monkeypatch: pytest.MonkeyPatch, locked: Path) -> None:
    """Fail every look below ``locked`` as it fails for a uid that cannot search it (``EACCES``), where the suite runs
    as root, which can."""
    accessor = type(getattr(Path(), "_accessor", None))  # Python 3.10's Path calls the os functions through it
    for name in ("stat", "lstat"):
        real = getattr(os, name)

        def look(path: Any, *args: Any, _real: Any = real, **kwargs: Any) -> Any:
            if not isinstance(path, int) and os.fsdecode(path).startswith(f"{locked}{os.sep}"):
                raise PermissionError(errno.EACCES, "Permission denied", os.fsdecode(path))
            return _real(path, *args, **kwargs)

        monkeypatch.setattr(os, name, look)
        if hasattr(accessor, name):
            monkeypatch.setattr(accessor, name, staticmethod(look))


@pytest.mark.parametrize("whose", ["the child's", "the app's too"])
def test_a_home_the_app_cannot_search_has_no_hugging_face_cache_to_show_and_the_launch_goes_on(
    tmp_path, monkeypatch, whose
):
    """CI's runner is not root, and its HOME may name ``/root``: the cache, the model and the logins below a home the
    app cannot search read as absent (nothing of the app's shows them either), never as a ``PermissionError``."""
    locked = tmp_path / "locked-home"
    (locked / ".cache" / "huggingface").mkdir(parents=True)
    (locked / "model").mkdir()
    _cannot_search(monkeypatch, locked)
    if whose == "the app's too":
        monkeypatch.setenv("HOME", str(locked))
    place = _place(tmp_path, readable=(locked / "model",))
    env = _env(HOME=str(locked), CLAUDE_CODE_OAUTH_TOKEN="token-value")

    prepared = prepare_launch(_bwrap(), place, ["claude", "--print"], env)
    prepared.release()

    assert "HF_HOME" not in env and env["HOME"] == str(place.workspace / ".home")
    assert prepared.policy is not None and not any(path.startswith(str(locked)) for path in prepared.policy.read_only)


_LAUNCH_AS_IT_IS = """
import json, sys
from pathlib import Path
from hyperloom_common.bwrap import BwrapProbe
from hyperloom.common.agent_isolation import BwrapIsolation
from hyperloom.orchestrator.specialists.launch_env import LaunchPlace, prepare_launch
home, ws = sys.argv[1], Path(sys.argv[2])
env = {"PATH": "/usr/bin:/bin", "HOME": home, "CLAUDE_CODE_OAUTH_TOKEN": "token-value"}
place = LaunchPlace(ws, ws / "worktree", readable=(Path(home, "model"),))
prepared = prepare_launch(BwrapIsolation(BwrapProbe("/opt/fake/bwrap", (0, 13, 1), ())), place, ["claude"], env)
prepared.release()
print(json.dumps({"home": env["HOME"], "hf": "HF_HOME" in env, "read_only": prepared.policy.read_only}))
"""


def test_a_launch_by_a_uid_that_cannot_search_its_home_goes_on_without_what_lies_there():
    """The same, with no stand-in: the app runs as a uid that cannot search the home it and the child are given
    (as root here: user 65534 through ``setpriv``, against a root-owned 0700 home; else the suite's own uid against
    a 0000 one)."""
    src = Path(launch_env.__file__).resolve().parents[3]
    as_root = os.geteuid() == 0
    drop = ["setpriv", "--reuid", "65534", "--regid", "65534", "--clear-groups"] if as_root else []
    if as_root and shutil.which("setpriv") is None:
        pytest.skip("needs setpriv to run as another uid")
    base = Path(tempfile.mkdtemp(prefix="hl-unsearchable-"))
    home, ws = base / "home", base / "ws"
    try:
        base.chmod(0o755)
        (home / ".cache" / "huggingface").mkdir(parents=True)
        (home / "model").mkdir()
        (ws / "worktree").mkdir(parents=True)
        if as_root:
            os.chown(ws, 65534, 65534)
            os.chown(ws / "worktree", 65534, 65534)
        env = {"PATH": os.defpath, "HOME": str(home), "PYTHONPATH": str(src)}
        reached = subprocess.run([*drop, sys.executable, "-c", "import hyperloom"], env=env, capture_output=True)
        if reached.returncode != 0:
            pytest.skip(f"this interpreter or tree cannot be run as the other uid: {reached.stderr[-300:]!r}")
        home.chmod(0o700 if as_root else 0)
        done = subprocess.run(
            [*drop, sys.executable, "-c", _LAUNCH_AS_IT_IS, str(home), str(ws)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        home.chmod(0o700)
        shutil.rmtree(base)

    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout)
    assert seen["home"] == str(ws / ".home") and not seen["hf"]
    assert not any(path.startswith(str(home)) for path in seen["read_only"])


def test_the_apps_state_directory_is_masked_where_a_bind_would_show_it(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    (shared / "state").mkdir(parents=True)
    monkeypatch.setenv("HYPERLOOM_STATE_DIR", str(shared / "state"))

    confinement = specialist_confinement(_place(tmp_path, readable=(shared,)), sys.executable, _env())

    assert shared in confinement.readable and shared / "state" in confinement.masked


def test_a_sandbox_launch_with_no_policy_is_refused_never_run_unconfined(tmp_path):
    prepared = PreparedLaunch(_bwrap())

    with pytest.raises(IsolationRefused, match="no sandbox policy"):
        prepared.local(["claude"], _env())
    with pytest.raises(IsolationRefused, match="no sandbox policy"):
        prepared.actor(["claude"], _env())
    assert prepared.status is None


def test_an_unconfined_launch_is_counted_for_the_session_report_and_said_once(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="hyperloom.common.agent_isolation"):
        for _ in range(2):
            prepare_launch(NoIsolation(("no tier here",)), _place(tmp_path), ["claude"], _env())

    assert len(caplog.records) == 1 and caplog.records[0].getMessage().endswith("(no tier here)")
    assert session_isolation_lines() == ("agent isolation: 2 agent launches ran unconfined: no tier here",)
    prepare_launch(_bwrap(), _place(tmp_path), ["claude"], _env()).release()
    assert session_isolation_lines() == ("agent isolation: 2 agent launches ran unconfined: no tier here",)


def test_the_parents_copy_of_the_status_pipe_is_closed_once_the_launch_holds_its_own(tmp_path):
    """bubblewrap's status reads end of file only when every write end is closed, the app's own copy included."""
    prepared = prepare_launch(_bwrap(), _place(tmp_path), ["claude"], env := _env())
    try:
        prepared.local(["claude"], env)
        status = prepared.status
        write_fd = status.write_fd
        pipe = os.fstat(write_fd)
        prepared.started(object())
        assert status.write_fd == -1
        try:
            still = os.fstat(write_fd)
        except OSError:
            still = None
        assert still is None or (still.st_dev, still.st_ino) != (pipe.st_dev, pipe.st_ino)
    finally:
        prepared.release()


@pytest.mark.parametrize(("version", "bound"), [((0, 9, 0), False), ((0, 13, 1), True)])
def test_an_older_bubblewrap_gets_a_fresh_tmp_when_something_is_mounted_below_it(tmp_path, version, bound):
    root = Path("/tmp") / f"hl-b3-{os.getpid()}-{version[1]}"
    place = LaunchPlace(root / "ws")
    place.workspace.mkdir(parents=True)
    try:
        prepared = prepare_launch(_bwrap(version), place, ["claude"], _env())
        try:
            assert prepared.policy is not None
            assert (prepared.policy.tmp is not None) is bound
            assert prepared.policy.shm == prepared.scratch.shm
            spawn = prepared.local(["claude"], _env())
            assert ("--bind" in spawn.argv[spawn.argv.index("/tmp") - 2 : spawn.argv.index("/tmp")]) is bound
        finally:
            prepared.release()
    finally:
        import shutil

        shutil.rmtree(root)


def _linked_worktree(repo: Path, worktree: Path) -> Path:
    """Make ``repo`` with one commit and ``worktree`` a linked worktree of it; return its admin dir as ``repo`` records
    it."""
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "--allow-empty", "-m", "base"], check=True, env=_git_env())
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", str(worktree)], check=True)
    return recorded_admin_dir(repo, worktree)


def test_the_confinement_reads_the_model_and_the_worktrees_git_and_writes_only_its_trees(tmp_path):
    repo, model = tmp_path / "repo", tmp_path / "models" / "llama"
    model.mkdir(parents=True)
    place = LaunchPlace(tmp_path / "ws", tmp_path / "ws" / "worktree", readable=(model, tmp_path / "ws" / "x"))
    place.workspace.mkdir()
    place = replace(place, git_admin=_linked_worktree(repo, place.worktree))

    confinement = specialist_confinement(place, sys.executable, _env())

    assert confinement.writable == (place.worktree, place.workspace)
    assert model in confinement.readable and (repo / ".git") in confinement.readable
    assert not any(path == place.workspace / "x" for path in confinement.readable)
    assert confinement.devices == ()


def test_the_sandbox_shows_the_git_metadata_the_repository_records_never_what_the_worktrees_git_names(tmp_path):
    """An agent of the app's uid can point ``.git`` at a repository it made, ownership and back pointer and all."""
    repo, other = tmp_path / "repo", tmp_path / "other"
    place = LaunchPlace(tmp_path / "ws", tmp_path / "ws" / "worktree")
    place.workspace.mkdir()
    admin = _linked_worktree(repo, place.worktree)
    forged = _linked_worktree(other, tmp_path / "other-worktree")
    (forged / "gitdir").write_text(f"{place.worktree / '.git'}\n", encoding="utf-8")
    (place.worktree / ".git").write_text(f"gitdir: {forged}\n", encoding="utf-8")
    assert trusted_admin_dir(place.worktree) == forged  # every ownership check passes for the app's own uid

    pinned = specialist_confinement(replace(place, git_admin=admin), sys.executable, _env()).readable
    unpinned = specialist_confinement(place, sys.executable, _env()).readable

    assert repo / ".git" in pinned and not any(other in path.parents or path == other for path in pinned)
    assert not any(path.name == ".git" for path in unpinned)


def test_a_repository_in_a_tree_the_sandbox_writes_is_refused_not_bound(tmp_path):
    place = LaunchPlace(tmp_path / "ws", tmp_path / "ws" / "worktree")
    place.workspace.mkdir()
    place = replace(place, git_admin=_linked_worktree(place.workspace / "repo", place.worktree))

    with pytest.raises(IsolationRefused, match="reached through a tree the agent writes"):
        prepare_launch(_bwrap(), place, ["claude"], _env())


def _git_env() -> dict[str, str]:
    names = ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL")
    return {**os.environ, **dict.fromkeys(names, "t"), "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_EMAIL": "t@t"}


def test_a_gpu_task_gets_the_gpu_devices_the_host_has(tmp_path, monkeypatch):
    kfd, dri = tmp_path / "kfd", tmp_path / "dri-missing"
    kfd.touch()
    monkeypatch.setattr(launch_env, "GPU_DEVICES", (kfd, dri))

    assert specialist_confinement(_place(tmp_path, gpu=True), sys.executable, _env()).devices == (kfd,)
    assert specialist_confinement(_place(tmp_path), sys.executable, _env()).devices == ()


def test_a_bedrock_credentials_file_is_read_in_the_sandbox_and_the_rest_of_the_home_is_masked(tmp_path):
    home = tmp_path / "home"
    credentials = home / ".aws" / "credentials"
    credentials.parent.mkdir(parents=True)
    credentials.write_text("[default]\n", encoding="utf-8")
    (home / ".aws" / "config").write_text("[default]\n", encoding="utf-8")
    (home / ".netrc").write_text("machine x\n", encoding="utf-8")
    env = _env(HOME=str(home), CLAUDE_CODE_USE_BEDROCK="1", AWS_SHARED_CREDENTIALS_FILE=str(credentials))

    confinement = specialist_confinement(_place(tmp_path), sys.executable, env)

    assert bedrock_files(env) == (credentials,)
    assert credentials in confinement.readable and credentials not in confinement.masked
    assert {home / ".netrc", home / ".aws" / "config"} <= set(confinement.masked)
    assert bedrock_files({**env, "CLAUDE_CODE_USE_BEDROCK": "0"}) == ()


def test_a_stdio_mcp_servers_program_is_bound_and_named_once(tmp_path, caplog):
    server = tmp_path / "tools" / "bin" / "mcp-server"
    server.parent.mkdir(parents=True)
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    config = tmp_path / "mcp.json"
    servers = {"local": {"command": str(server)}, "remote": {"type": "http", "url": "http://127.0.0.1:9/mcp"}}
    config.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    place = _place(tmp_path, mcp_config=str(config))

    with caplog.at_level(logging.WARNING, logger=launch_env.__name__):
        confinement = specialist_confinement(place, sys.executable, _env())
        specialist_confinement(place, sys.executable, _env())

    assert server.parent in confinement.readable
    named = [record.getMessage() for record in caplog.records if "stdio MCP servers" in record.getMessage()]
    assert len(named) == 1 and f"local ({server})" in named[0] and "remote" not in named[0]


@pytest.mark.parametrize(
    ("env", "refused"),
    [
        ({}, True),
        ({"CLAUDE_CODE_OAUTH_TOKEN": "t"}, False),
        ({"ANTHROPIC_API_KEY": "k"}, False),
        ({"CLAUDE_CODE_USE_BEDROCK": "1"}, True),
        ({"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "s"}, False),
        ({"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": "p"}, True),
        ({"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_CONFIG_FILE": "FILE"}, False),
        ({"CLAUDE_CODE_USE_BEDROCK": "0", "AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "s"}, True),
    ],
)
def test_a_claude_cli_with_no_login_a_sandbox_can_show_runs_unconfined_unless_it_must_be_isolated(
    tmp_path, monkeypatch, env, refused
):
    aws_config = tmp_path / "aws-config"
    aws_config.write_text("[default]\n", encoding="utf-8")
    env = {name: str(aws_config) if value == "FILE" else value for name, value in env.items()}
    sandbox = _bwrap()

    kept = [NoIsolation(("why",)), AccountIsolation(RunAs("u", 1, 1, (1,), "/h")), *([] if refused else [sandbox])]
    for isolation in kept:
        assert claude_login_isolation(isolation, env) is isolation
    if not refused:
        return
    assert claude_login_isolation(sandbox, env) == NoIsolation((CLAUDE_LOGIN_UNCONFINED,))
    for must in ({"HYPERLOOM_SPECIALIST_USER": "specialist"}, {"hosted": True}):
        with monkeypatch.context() as patched:
            if "hosted" in must:
                patched.setattr(agent_isolation._STATE, "hosted", True)
            else:
                patched.setenv("HYPERLOOM_SPECIALIST_USER", must["HYPERLOOM_SPECIALIST_USER"])
            with pytest.raises(SpecialistAgentUnavailableError) as refusal:
                claude_login_isolation(sandbox, env)
        assert str(refusal.value) == CLAUDE_LOGIN_REFUSAL


def test_the_claude_login_reasons_say_why_and_how_to_run_it_confined():
    for text in (CLAUDE_LOGIN_UNCONFINED, CLAUDE_LOGIN_REFUSAL):
        assert "login file" in text and "without showing that file to the agent" in text
        assert "CLAUDE_CODE_OAUTH_TOKEN (`claude setup-token` prints one)" in text
    assert CLAUDE_LOGIN_UNCONFINED.startswith("a claude CLI with no login in its environment runs unconfined")
    assert CLAUDE_LOGIN_UNCONFINED.endswith("to run it confined")
    assert "must be isolated" in CLAUDE_LOGIN_REFUSAL


#: The session report's line once two such launches ran unconfined.
TWO_UNCONFINED_LOGINS = f"agent isolation: 2 agent launches ran unconfined: {CLAUDE_LOGIN_UNCONFINED}"
#: A claude CLI whose one login is the file under the app's home: it notes whether it could read that file.
_LOGIN_FILE_ONLY = """#!/bin/sh
cat > /dev/null
{{ cat {login} > /dev/null 2>&1 && echo read || echo unreadable; }} > login_seen
echo '{{"summary": "ran", "proposal_set": []}}' > specialist_done.json
"""


@pytest.fixture
def login_file_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The stand-in claude CLI of :data:`_LOGIN_FILE_ONLY`, launched where bubblewrap is chosen (a stand-in one: a
    launch that tried to sandbox it would fail) and with no login in the environment."""
    login = tmp_path / "home" / ".claude" / ".credentials.json"
    login.parent.mkdir(parents=True)
    login.write_text('{"claudeAiOauth": {}}', encoding="utf-8")
    monkeypatch.setenv("HOME", str(login.parent.parent))
    for name in (*CLAUDE_TOKEN_VARIABLES, "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CONFIG_DIR", "HYPERLOOM_SPECIALIST_USER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(launch_env, "choose_agent_isolation", lambda **_kwargs: _bwrap())
    cli = tmp_path / "cli" / "bin" / "claude"
    cli.parent.mkdir(parents=True)
    cli.write_text(_LOGIN_FILE_ONLY.format(login=login), encoding="utf-8")
    cli.chmod(0o755)
    return cli


def _login_dispatcher(cli: Path) -> ss.SpecialistSubprocessDispatcher:
    return ss.SpecialistSubprocessDispatcher(
        ss.SpecialistSubprocessConfig(agent_backend="claude", claude_executable=str(cli), poll_interval_seconds=0.05)
    )


def _login_spawn(workspace: Path) -> ss.SpecialistSpawn:
    return ss.SpecialistSpawn(
        task_id=workspace.name,
        workspace=workspace,
        worktree=None,
        worktree_base=None,
        system_prompt="system",
        user_prompt="the task",
        deadline=Deadline.after(60.0),
    )


def _unconfined_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]


async def test_the_dispatcher_runs_a_claude_cli_with_only_its_login_file_unconfined_and_says_so_once(
    tmp_path, login_file_only, caplog
):
    with caplog.at_level(logging.WARNING):
        results = [await _login_dispatcher(login_file_only).run(_login_spawn(tmp_path / f"ws{n}")) for n in range(2)]

    ran = [(result.error, result.exit_code, (result.done_payload or {}).get("summary")) for result in results]
    assert ran == [("", 0, "ran")] * 2
    assert [(tmp_path / f"ws{n}" / "login_seen").read_text(encoding="utf-8") for n in range(2)] == ["read\n"] * 2
    warnings = _unconfined_warnings(caplog)
    assert len(warnings) == 1 and warnings[0].endswith(f"({CLAUDE_LOGIN_UNCONFINED})"), warnings
    assert session_isolation_lines() == (TWO_UNCONFINED_LOGINS,)


async def test_the_dispatcher_refuses_a_claude_cli_with_only_its_login_file_where_it_must_be_isolated(
    tmp_path, login_file_only, monkeypatch
):
    monkeypatch.setattr(agent_isolation._STATE, "hosted", True)

    result = await _login_dispatcher(login_file_only).run(_login_spawn(workspace := tmp_path / "ws"))

    assert result.exit_code is None
    assert result.error == f"specialist agent runtime unavailable (backend=claude): {CLAUDE_LOGIN_REFUSAL}"
    assert not (workspace / "login_seen").exists() and session_isolation_lines() == ()


async def test_the_agent_runner_runs_a_claude_cli_with_only_its_login_file_unconfined_and_says_so_once(
    tmp_path, login_file_only, caplog
):
    runner = make_runner(tmp_path / "session")
    statuses, procs = [], []
    with caplog.at_level(logging.WARNING):
        for n in range(2):
            handle = await runner.launch(make_spec(tmp_path / f"w{n}", task_id=f"t{n}", argv=(str(login_file_only),)))
            statuses.append(await wait_finished(runner, handle))
            await runner.collect(handle)
            procs.append(handle.session.proc)

    assert [(status.returncode, status.error) for status in statuses] == [(0, "")] * 2
    assert not any(isinstance(proc, SandboxedProcess) for proc in procs)
    assert [(tmp_path / f"w{n}" / "login_seen").read_text(encoding="utf-8") for n in range(2)] == ["read\n"] * 2
    warnings = _unconfined_warnings(caplog)
    assert len(warnings) == 1 and warnings[0].endswith(f"({CLAUDE_LOGIN_UNCONFINED})"), warnings
    assert session_isolation_lines() == (TWO_UNCONFINED_LOGINS,)


async def test_the_agent_runner_refuses_a_claude_cli_with_only_its_login_file_where_it_must_be_isolated(
    tmp_path, login_file_only, monkeypatch
):
    monkeypatch.setattr(agent_isolation._STATE, "hosted", True)
    runner = make_runner(tmp_path / "session")

    with pytest.raises(AgentConfigError) as refused:
        await runner.launch(make_spec(workspace := tmp_path / "w", argv=(str(login_file_only),)))

    assert str(refused.value) == CLAUDE_LOGIN_REFUSAL
    assert not (workspace / "login_seen").exists() and session_isolation_lines() == ()


@pytest.mark.parametrize(
    "argv",
    [
        ["codex", "exec", "--json", "--sandbox", "workspace-write", "-"],
        ["codex", "exec", "-s", "read-only", "--full-auto", "-"],
        ["codex", "exec", "--sandbox=workspace-write", "-"],
        ["codex", "exec", CODEX_BYPASS_FLAG, "-"],
    ],
)
def test_codex_in_a_sandbox_runs_with_its_own_sandbox_off(argv):
    out = without_codex_sandbox(argv)

    assert out[:3] == ["codex", "exec", CODEX_BYPASS_FLAG] and out[-1] == "-"
    assert not {"--sandbox", "-s", "--full-auto"} & set(out) and out.count(CODEX_BYPASS_FLAG) == 1
    assert not any(arg.startswith("--sandbox=") or arg in ("workspace-write", "read-only") for arg in out)


def test_a_sandbox_gets_a_copy_of_the_mcp_config_the_app_keeps_private(tmp_path):
    source = tmp_path / "runtime" / "specialist_mcp.json"
    source.parent.mkdir()
    source.write_text('{"mcpServers": {}}', encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cmd = ["claude", "--mcp-config", str(source)]

    stage_mcp_config(_bwrap(), workspace, cmd)

    assert cmd[-1] == str(workspace / "specialist_mcp.json")
    assert Path(cmd[-1]).read_text(encoding="utf-8") == '{"mcpServers": {}}'


class _Exited:
    def __init__(self, code: int | None) -> None:
        self.code = code

    def poll(self) -> int | None:
        return self.code


@pytest.mark.parametrize(
    ("status", "code", "log_text", "failure"),
    [
        (b'{ "child-pid": 9 }\n', 1, "bwrap: Can't find source path /x: No such file\n", "bwrap: Can't find source"),
        (b'{ "child-pid": 9 }\n', 1, "", "bubblewrap exited 1 before the agent started"),
        (b'{ "child-pid": 9 }\n{ "exit-code": 1 }\n', 1, "bwrap: not this\n", ""),
        (b'{ "child-pid": 9 }\n', -9, "", ""),
        (b'{ "child-pid": 9 }\n', None, "", ""),
        (b"", 0, "bwrap: unread status\n", ""),
        (b"{ garbage\n", 1, "", "bubblewrap exited 1 before the agent started"),
    ],
)
def test_a_setup_failure_is_told_from_the_agents_exit_by_bubblewraps_status(tmp_path, status, code, log_text, failure):
    status_pipe = LaunchStatus.pipe()
    os.write(status_pipe.write_fd, status)
    status_pipe.started()
    process_log = tmp_path / "process.log"
    process_log.write_text(log_text, encoding="utf-8")
    try:
        found = SandboxedProcess(_Exited(code), "mark", status_pipe).setup_failure(process_log)
    finally:
        status_pipe.close()
    assert (found.startswith(failure) if failure else found == "") is True


_TRAPPED = """
import pathlib, signal, sys, time
def stop(*_):
    pathlib.Path(sys.argv[1]).write_text("stopped")
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
pathlib.Path(sys.argv[2]).write_text("ready")
time.sleep(60)
"""


def _marked_child(tmp_path: Path, mark: str, code: str = _TRAPPED) -> tuple[subprocess.Popen, Path]:
    stopped, ready = tmp_path / "stopped", tmp_path / "ready"
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(stopped), str(ready)],
        env={**os.environ, REAP_MARK_ENV: mark},
        start_new_session=True,
    )
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    return child, stopped


def test_the_kill_asks_a_sandboxed_agent_to_stop_before_it_ends_bubblewraps_tree(tmp_path, monkeypatch):
    mark = f"b3-graceful-{os.getpid()}"
    child, stopped = _marked_child(tmp_path, mark)
    monkeypatch.setattr(ss, "kill_tree", lambda *args, **kwargs: pytest.fail("the agent stopped on its own"))
    status = LaunchStatus.pipe()
    try:
        assert ss.SpecialistSubprocessDispatcher._kill(SandboxedProcess(child, mark, status)) is True
    finally:
        status.close()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
    assert stopped.read_text(encoding="utf-8") == "stopped" and child.returncode == 0


def test_an_agent_that_ignores_the_term_is_killed_with_bubblewraps_tree(tmp_path, monkeypatch):
    mark = f"b3-stubborn-{os.getpid()}"
    stubborn = _TRAPPED.replace('pathlib.Path(sys.argv[1]).write_text("stopped")\n    sys.exit(0)', "pass")
    child, stopped = _marked_child(tmp_path, mark, stubborn)
    status = LaunchStatus.pipe()
    try:
        process = SandboxedProcess(child, mark, status)
        assert process.stop_agent(grace_sec=0.3) is False
        assert ss.SpecialistSubprocessDispatcher._kill(process) is True
    finally:
        status.close()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
    assert child.returncode == -signal.SIGKILL and not stopped.exists() and not running(child.pid)


def test_a_sandbox_whose_agent_carries_no_mark_is_not_waited_on(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    status = LaunchStatus.pipe()
    try:
        started = time.monotonic()
        assert SandboxedProcess(child, "nobody-carries-this", status).stop_agent(grace_sec=30) is False
        assert time.monotonic() - started < 5
    finally:
        status.close()
        child.kill()
        child.wait(timeout=5)


def test_a_ray_actor_launch_opens_its_status_file_in_a_shell_and_keeps_no_die_with_parent(tmp_path):
    place = _place(tmp_path)
    env = _env()
    prepared = prepare_launch(_bwrap(), place, ["claude"], env)
    try:
        spawn = prepared.actor(["claude"], env)
        status_path = Path(spawn.argv[3])
        assert spawn.argv[:3] == ["/bin/sh", "-c", 'exec 3>>"$0" && exec "$@"'] and spawn.argv[4] == FAKE_BWRAP
        assert spawn.argv[spawn.argv.index("--json-status-fd") + 1] == "3"
        assert "--die-with-parent" not in spawn.argv and spawn.kwargs == {} and REAP_MARK_ENV not in spawn.env
        assert status_path.parent.stat().st_mode & 0o777 == 0o700 and not status_path.exists()
        assert isinstance(prepared.started(object()), SandboxedProcess)
    finally:
        prepared.release()
    assert not status_path.parent.exists()


def test_the_reap_reads_each_tiers_marks_its_own_way(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(launch_env, "marked_pids", lambda key, value: seen.append(f"app:{value}") or [])
    monkeypatch.setattr(launch_env, "marked_pids_as", lambda run_as, value: seen.append(f"as:{value}") or [])

    PreparedLaunch(NoIsolation(("why",))).marked_pids("a")
    PreparedLaunch(_bwrap()).marked_pids("b")
    PreparedLaunch(AccountIsolation(RunAs("u", 1, 1, (1,), "/h"))).marked_pids("c")

    assert seen == ["app:a", "app:b", "as:c"]
