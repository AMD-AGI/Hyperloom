# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session another account of the group can resume: what a leg creates under a strict umask is ``2775`` / ``664``
whichever account made it, so the next leg (on shared storage, often another host account of one group) can lock it,
read and replace its state, open its database and write beside it. A file the operator's credentials may be in is the
one exception and stays owner-only."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import sqlite3
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import hyperloom
from hyperloom.inference_optimizer.session import lock, paths
from hyperloom.inference_optimizer.session.resume_guard import ResumeBlocked, ensure_resume_safe
from hyperloom_common.tests.shared_accounts import other_accounts

_SRC = str(Path(hyperloom.__file__).parents[1])

#: What a first leg creates in a session, with the account's umask at its strictest; one line for each creation point.
_FIRST_LEG = textwrap.dedent(
    """
    import json, os
    from pathlib import Path
    from hyperloom.inference_optimizer.cli.parser import _build_parser
    from hyperloom.inference_optimizer.session import optimization_journal, paths, sbd_v6
    from hyperloom.inference_optimizer.session.lock import SessionLock
    from hyperloom.inference_optimizer.session.manifest import write_manifest
    from hyperloom.orchestrator.bringup.trees import write_trees
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
    from hyperloom.orchestrator.state.shared_state import SharedState

    args = _build_parser().parse_args(["optimize", "--model", "/models/m", "--framework", "sglang"])
    session = paths.make_session_dir("m")
    write_manifest(session, args=args)
    SharedState().save(session)
    with SessionLock(session):
        pass
    SqliteConnection(paths.db_path_for(session)).raw.close()
    sbd_v6.write_timeline_event_at(session, {"type": "phase"})
    sbd_v6.record_write_warning(session, component="test", exc=RuntimeError("x"))
    journal = optimization_journal.Journal.load_or_create(
        session, optimization_journal.Journal(session_id="s", model="m", hardware="h", framework="f")
    )
    journal.finalize(final_throughput=1.0)
    write_trees([], session_dir=session)
    print(session)
    """
)


def _as_this_account(code: str, *, umask: int, workspace: Path, **env: str) -> str:
    """Run ``code`` in a process of its own, as this account, under ``umask`` (this process's is never changed)."""
    preamble = f"import os, sys\nos.umask({umask:#o})\nsys.path.insert(0, {_SRC!r})\n"
    # Nothing of the test process's environment: what an earlier test left there decides what a leg writes.
    clean = {"PATH": os.environ["PATH"], "HOME": str(workspace.parent), "USER_DATA_PATH": str(workspace), **env}
    done = subprocess.run(
        [sys.executable, "-I", "-c", preamble + code], capture_output=True, text=True, timeout=120, env=clean
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _setgid_workspace(path: Path) -> Path:
    path.mkdir()
    path.chmod(0o2775)
    if not path.stat().st_mode & stat.S_ISGID:
        pytest.skip("this filesystem does not keep the setgid bit for the test user")
    return path


def _entries(session: Path) -> tuple[dict[str, int], dict[str, int]]:
    """The mode of each directory and of each regular file below ``session``."""
    directories, files = {}, {}
    for path in sorted(session.rglob("*")):
        (directories if path.is_dir() else files)[path.relative_to(session).as_posix()] = _mode(path)
    return directories, files


@pytest.mark.parametrize("umask", [0o077, 0o022])
def test_a_first_leg_makes_every_directory_2775_and_every_file_664_whatever_the_umask(tmp_path, umask):
    workspace = _setgid_workspace(tmp_path / "claw")
    session = Path(_as_this_account(_FIRST_LEG, umask=umask, workspace=workspace).splitlines()[-1])
    directories, files = _entries(session)
    assert {name: oct(mode) for name, mode in directories.items() if mode != 0o2775} == {}
    assert files == {
        "manifest.json": 0o664,
        "state.json": 0o664,
        "retest_view.json": 0o640,  # its own group-read rule: never group-writable, replaced by rename
        "runtime/optimizer.lock": 0o664,
        "runtime/pod_history.jsonl": 0o664,
        "runtime/breakdown/parts/metadata__coordinator.json": 0o664,
        "storage/coordinator.db": 0o664,
        "reports/sbd_v6/timeline/000001-phase.json": 0o664,
        "reports/sbd_v6/write_warnings.jsonl": 0o664,
        "reports/optimization_journal.json": 0o664,
        "reports/bringup/trees.json": 0o664,
    }
    assert {name: _mode(workspace / name) for name in ("runtime", "runtime/recipe_kb", "logs")} == {
        "runtime": 0o2755,  # the installer's: the group enters it, only its owner changes what is in it
        # below ``runtime/``, which the group cannot write, nothing is shared: the umask decides (ops makes it 2775)
        "runtime/recipe_kb": 0o2000 | (0o775 & ~umask),
        "logs": 0o2775,
    }


def test_the_database_files_sqlite_makes_beside_a_664_database_are_664_too(tmp_path):
    workspace = _setgid_workspace(tmp_path / "claw")
    storage = workspace / "storage"
    code = textwrap.dedent(
        f"""
        import json, os, stat
        from pathlib import Path
        from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
        connection = SqliteConnection(Path({str(storage / "coordinator.db")!r}))
        modes = {{p.name: stat.S_IMODE(p.stat().st_mode) for p in Path({str(storage)!r}).iterdir()}}
        print(json.dumps(modes))
        """
    )
    modes = json.loads(_as_this_account(code, umask=0o077, workspace=workspace).splitlines()[-1])
    assert modes == {"coordinator.db": 0o664, "coordinator.db-shm": 0o664, "coordinator.db-wal": 0o664}


@pytest.mark.parametrize(
    ("pins", "mode"),
    [
        ({}, 0o664),
        ({"operator_extra_env": {"SGLANG_USE_AITER": "1", "MAX_NUM_BATCHED_TOKENS": "8192"}}, 0o664),
        ({"operator_server_args": "--tp 8 --max-total-tokens 120000 --tokenizer-path /models/m"}, 0o664),
        ({"operator_extra_env": {"HF_TOKEN": "hf_live_value_123"}}, 0o600),
        ({"operator_extra_env": {"NOTES": "Authorization: Bearer abcdefghijklmnop"}}, 0o600),
        ({"operator_extra_env": {"HF_ENDPOINT": "https://alice:s3cret-tok3n@hf-mirror"}}, 0o600),
        ({"operator_extra_env": {"HF_ENDPOINT": "https://hf-mirror.example/models"}}, 0o664),
        ({"operator_server_args": "--tp 8 --api-key not-a-pattern"}, 0o600),
        ({"operator_server_args": "--tp 8 --admin-api-key=not-a-pattern"}, 0o600),
    ],
    ids=[
        "no-pins",
        "env-knobs",
        "server-args-knobs",
        "env-token",
        "env-bearer",
        "env-url-userinfo",
        "env-url-without-userinfo",
        "flag-api-key",
        "flag-admin-api-key",
    ],
)
def test_state_json_is_664_unless_the_operators_pins_hold_a_credential(tmp_path, pins, mode):
    workspace = _setgid_workspace(tmp_path / "claw")
    code = textwrap.dedent(
        f"""
        from pathlib import Path
        from hyperloom.orchestrator.state.shared_state import SharedState
        state = SharedState()
        for name, value in {pins!r}.items():
            setattr(state, name, value)
        state.save(Path({str(workspace / "a" / "b")!r}))
        """
    )
    _as_this_account(code, umask=0o077, workspace=workspace)
    assert _mode(workspace / "a" / "b" / "state.json") == mode
    assert (_mode(workspace / "a"), _mode(workspace / "a" / "b")) == (0o2775, 0o2775), "a private file, shared parents"


@pytest.mark.parametrize(
    ("server_args", "mode"),
    [
        (None, 0o664),
        ("--tp 8 --max-total-tokens 120000 --tokenizer-path /models/m", 0o664),
        ("--tp 8 --api-key not-a-pattern", 0o600),
    ],
    ids=["no-args", "knobs", "api-key"],
)
def test_manifest_json_is_664_unless_the_server_args_hold_a_credential(tmp_path, server_args, mode):
    workspace = _setgid_workspace(tmp_path / "claw")
    code = textwrap.dedent(
        f"""
        from pathlib import Path
        from hyperloom.inference_optimizer.cli.parser import _build_parser
        from hyperloom.inference_optimizer.session.manifest import write_manifest
        argv = ["optimize", "--model", "/models/m", "--framework", "sglang"]
        if {server_args!r}:
            argv += ["--server-args", {server_args!r}]
        write_manifest(Path({str(workspace)!r}), args=_build_parser().parse_args(argv))
        """
    )
    _as_this_account(code, umask=0o077, workspace=workspace)
    assert _mode(workspace / "manifest.json") == mode


def test_a_parent_this_account_cannot_create_in_is_named_with_its_owner_and_mode(tmp_path):
    """``<workspace>/<model>`` made by ops without group write: the refusal says which directory and what it needs."""
    if os.geteuid() == 0:
        with other_accounts() as accounts:
            workspace = accounts.shared_dir("claw")
            model = workspace / "DeepSeek-V4.1-Flash"
            model.mkdir()
            os.chown(model, 0, accounts.group)
            model.chmod(0o2755)
            accounts.add_module(Path(paths.__file__), "claw_paths")
            code = (
                f"import os\nos.environ['USER_DATA_PATH'] = {str(workspace)!r}\nimport claw_paths\n"
                "try:\n    claw_paths.make_session_dir('DeepSeek-V4.1-Flash')\n"
                "except PermissionError as exc:\n    print(exc)\n"
            )
            said = accounts.run(accounts.first, code).strip()
            assert said.startswith(f"[Errno 13] cannot create {model}/")
            assert f"the directory {model} (owner 0:{accounts.group} 2755) is not writable by this account" in said
            assert "chmod 2775" in said
            assert sorted(path.name for path in model.iterdir()) == []
        return
    workspace = _setgid_workspace(tmp_path / "claw")
    model = workspace / "DeepSeek-V4.1-Flash"
    model.mkdir()
    model.chmod(0o555)
    os.environ["USER_DATA_PATH"] = str(workspace)
    try:
        with pytest.raises(PermissionError) as raised:
            paths.make_session_dir("DeepSeek-V4.1-Flash")
    finally:
        del os.environ["USER_DATA_PATH"]
    info = model.stat()
    assert f"the directory {model} (owner {info.st_uid}:{info.st_gid} 0555) is not writable by this account" in str(
        raised.value
    )
    assert sorted(path.name for path in model.iterdir()) == []


def test_a_session_made_below_a_parent_with_group_write_is_2775_all_the_way_up(tmp_path):
    workspace = _setgid_workspace(tmp_path / "claw")
    code = "from hyperloom.inference_optimizer.session import paths\nprint(paths.make_session_dir('DeepSeek-V4.1-Flash'))\n"
    session = Path(_as_this_account(code, umask=0o077, workspace=workspace).splitlines()[-1])
    assert (session.parent.parent, session.parent, session) == (workspace, workspace / session.parent.name, session)
    assert {_mode(session.parent), _mode(session), _mode(workspace / "logs")} == {0o2775}
    assert _mode(workspace / "runtime") == 0o2755


def test_the_install_owned_runtime_is_not_group_writable_but_the_session_subtree_is(tmp_path):
    """Two ``runtime/`` directories: the workspace's (the installer writes what a run sources and runs there) and the
    session's (the lock, the ledger and the breakdown spool): only the second one is shared for writing."""
    workspace = _setgid_workspace(tmp_path / "claw")  # as ops sets it up
    code = textwrap.dedent(
        """
        from hyperloom.inference_optimizer.session import paths
        from hyperloom.inference_optimizer.session.lock import SessionLock
        session = paths.make_session_dir("DeepSeek-V4.1-Flash")
        with SessionLock(session):
            pass
        print(session)
        """
    )
    session = Path(_as_this_account(code, umask=0o077, workspace=workspace).splitlines()[-1])
    modes = {
        "workspace root": _mode(workspace),
        "workspace runtime/": _mode(workspace / "runtime"),
        "workspace logs": _mode(workspace / "logs"),
        "model directory": _mode(session.parent),
        "session": _mode(session),
        "session runtime/": _mode(session / "runtime"),
    }
    assert modes == {
        "workspace root": 0o2775,  # ops's, left as it is
        "workspace runtime/": 0o2755,
        "workspace logs": 0o2775,
        "model directory": 0o2775,
        "session": 0o2775,
        "session runtime/": 0o2775,
    }
    assert not modes["workspace runtime/"] & (stat.S_IWGRP | stat.S_IWOTH)


def test_a_workspace_the_app_makes_itself_is_2755_and_shares_nothing_below_it(tmp_path):
    """A workspace root made by ``make_session_dir`` (not set up by ops) is not shared storage: it is ``2755``, and what
    is made below it is as private as the umask leaves it."""
    parent = _setgid_workspace(tmp_path / "shared")
    workspace = parent / "claw"
    code = "from hyperloom.inference_optimizer.session import paths\nprint(paths.make_session_dir('m'))\n"
    session = Path(_as_this_account(code, umask=0o077, workspace=workspace).splitlines()[-1])
    assert _mode(workspace) == 0o2755
    assert {_mode(workspace / "runtime"), _mode(workspace / "logs"), _mode(session.parent), _mode(session)} == {0o2700}
    assert not any(_mode(path) & (stat.S_IWGRP | stat.S_IWOTH) for path in [workspace, *workspace.rglob("*")])


@pytest.mark.parametrize("name", ["manifest.json", "state.json"])
def test_a_resume_by_an_account_that_cannot_read_an_owner_only_file_is_refused_naming_it(tmp_path, monkeypatch, name):
    for each in ("manifest.json", "state.json"):
        (tmp_path / each).write_text("{}")
    target = tmp_path / name
    target.chmod(0o600)
    real_open = Path.open

    def as_another_account(self, *args, **kwargs):
        if self == target:
            raise PermissionError(13, "Permission denied", str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", as_another_account)
    with pytest.raises(ResumeBlocked) as raised:
        ensure_resume_safe(tmp_path, owner_scope="host:1")
    info = target.stat()
    assert str(raised.value) == (
        f"cannot read {target} (owner {info.st_uid}:{info.st_gid} 0600). A session whose --extra-env or --server-args "
        "hold a credential keeps its state and manifest owner-only: resume it as the account that made it, or start "
        "a new session."
    )


#: A first leg whose operator pins hold a credential: its state and manifest stay owner-only.
_CREDENTIAL_LEG = textwrap.dedent(
    """
    from hyperloom.inference_optimizer.cli.parser import _build_parser
    from hyperloom.inference_optimizer.session import paths
    from hyperloom.inference_optimizer.session.manifest import write_manifest
    from hyperloom.orchestrator.state.shared_state import SharedState
    argv = ["optimize", "--model", "/models/m", "--framework", "sglang", "--server-args", "--tp 8 --api-key not-a-pattern"]
    session = paths.make_session_dir("m")
    write_manifest(session, args=_build_parser().parse_args(argv))
    state = SharedState()
    state.operator_extra_env = {"HF_TOKEN": "hf_live_value_123"}
    state.save(session)
    print(session)
    """
)

_READ_AS_THE_NEXT_ACCOUNT = textwrap.dedent(
    """
    import errno, json
    seen = {{}}
    for name in ("manifest.json", "state.json"):
        try:
            seen[name] = len(open({session!r} + "/" + name, "rb").read())
        except PermissionError as exc:
            seen[name] = errno.errorcode[exc.errno]
    print(json.dumps(seen))
    """
)


def test_the_next_account_reads_state_and_manifest_unless_they_hold_a_credential(tmp_path):
    with other_accounts() as accounts:
        for leg, expected in ((_FIRST_LEG, "bytes"), (_CREDENTIAL_LEG, "EACCES")):
            workspace = accounts.shared_dir(f"claw-{expected}")
            session = Path(_as_this_account(leg, umask=0o077, workspace=workspace).splitlines()[-1])
            out = accounts.run(accounts.second, _READ_AS_THE_NEXT_ACCOUNT.format(session=str(session)))
            seen = json.loads(out)
            if expected == "EACCES":
                assert seen == {"manifest.json": "EACCES", "state.json": "EACCES"}
                assert "hf_live_value_123" not in out and "not-a-pattern" not in out
                assert (_mode(session / "manifest.json"), _mode(session / "state.json")) == (0o600, 0o600)
            else:
                assert all(isinstance(size, int) and size > 0 for size in seen.values()), seen
                assert (_mode(session / "manifest.json"), _mode(session / "state.json")) == (0o664, 0o664)


#: A first leg, its coordinator, and a refused launch the bus stores with the operator's argv in it (what
#: ``dispatcher.reap_dispatched_task`` appends as a ``delegated_result``); then every file of the session that the group
#: or others can read, and the ones that hold the key, while the database is still open (its ``-wal`` and ``-shm`` too).
_BUS_LEG = textwrap.dedent(
    """
    import asyncio, json, os, stat
    from pathlib import Path
    from hyperloom.common.env_safety import OPERATOR_EXTRA_ENV_VAR
    from hyperloom.inference_optimizer.cli.parser import _build_parser
    from hyperloom.inference_optimizer.session import paths
    from hyperloom.inference_optimizer.session.manifest import write_manifest
    from hyperloom.orchestrator.bus.message_bus import Message
    from hyperloom.orchestrator.loop.coordinator import Coordinator, default_role_registry
    from hyperloom.orchestrator.state.shared_state import SharedState

    argv = ["optimize", "--model", "/models/m", "--framework", "sglang"]
    server_args, env = {server_args!r}, {env!r}
    if server_args:
        argv += ["--server-args", server_args]
    session = paths.make_session_dir("m")
    write_manifest(session, args=_build_parser().parse_args(argv))
    state = SharedState()
    if {exported!r}:
        # a resume with --reset-state: the state records no pins, the run still exports them
        os.environ["INFERENCE_OPTIMIZER_SERVER_ARGS"] = server_args
        os.environ[OPERATOR_EXTRA_ENV_VAR] = json.dumps(env)
    else:
        state.operator_server_args, state.operator_extra_env = server_args, env
    state.save(session)
    os.environ["INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"] = str(session)
    coordinator = Coordinator(session, backends={{name: object() for name in default_role_registry()}})
    announcement = {{
        "task_id": "t1",
        "kind": "benchmark_run",
        "state": "failed",
        "result": {{"server_argv": server_args.split(), "server_env": env, "enablement_launch_log": "x"}},
    }}
    asyncio.run(coordinator.bus.append_and_seq(Message.new("coordinator", "*", "delegated_result", announcement)))
    shared, holders, database = [], [], {{}}
    for path in sorted(session.rglob("*")):
        if not path.is_file():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.name.startswith("coordinator.db"):
            database[path.name] = mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            shared.append(path.relative_to(session).as_posix())
            if b"SECRET-KEY-VALUE" in path.read_bytes() or b"hf_live_value_123" in path.read_bytes():
                holders.append(path.relative_to(session).as_posix())
    print(json.dumps({{"database": database, "readable": shared, "holders": holders}}))
    """
)


@pytest.mark.parametrize(
    ("server_args", "env", "exported", "database_mode"),
    [
        ("--tp 8 --api-key SECRET-KEY-VALUE", {}, False, 0o600),
        ("--tp 8", {"HF_TOKEN": "hf_live_value_123"}, False, 0o600),
        ("--tp 8", {"HF_ENDPOINT": "https://alice:hf_live_value_123@hf-mirror"}, False, 0o600),
        ("--tp 8 --max-total-tokens 120000", {"SGLANG_USE_AITER": "1"}, False, 0o664),
        ("--tp 8 --api-key SECRET-KEY-VALUE", {}, True, 0o600),
        ("--tp 8", {"HF_TOKEN": "hf_live_value_123"}, True, 0o600),
        ("--tp 8 --max-total-tokens 120000", {"SGLANG_USE_AITER": "1"}, True, 0o664),
    ],
    ids=[
        "server-args-api-key",
        "env-token",
        "env-url-userinfo",
        "no-credential",
        "exported-server-args-api-key",
        "exported-env-token",
        "exported-no-credential",
    ],
)
def test_the_operators_credential_reaches_no_file_the_group_can_read_and_the_database_is_600_then(
    tmp_path, server_args, env, exported, database_mode
):
    workspace = _setgid_workspace(tmp_path / "claw")
    code = _BUS_LEG.format(server_args=server_args, env=env, exported=exported)
    seen = json.loads(_as_this_account(code, umask=0o077, workspace=workspace).splitlines()[-1])
    assert seen["database"] == dict.fromkeys(
        ("coordinator.db", "coordinator.db-wal", "coordinator.db-shm"), database_mode
    )
    assert seen["holders"] == []
    assert ("storage/coordinator.db" in seen["readable"]) == (database_mode == 0o664)


def test_a_database_a_session_made_before_it_held_a_credential_is_narrowed_to_600_by_its_owner(tmp_path):
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection

    database = _setgid_workspace(tmp_path / "claw") / "storage" / "coordinator.db"
    SqliteConnection(database).raw.close()
    assert _mode(database) == 0o664
    SqliteConnection(database, file_mode=0o600).raw.close()
    assert _mode(database) == 0o600
    SqliteConnection(database).raw.close()
    assert _mode(database) == 0o600, "a wider mode asked later never widens it"


_LEFT_BY_A_KILLED_WRITER = textwrap.dedent(
    """
    import os
    from pathlib import Path
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
    connection = SqliteConnection(Path({database!r}))
    connection.raw.execute("CREATE TABLE IF NOT EXISTS leftover (value TEXT)")
    connection.raw.execute("INSERT INTO leftover VALUES ('written by the leg that was killed')")
    os._exit(0)
    """
)


def test_the_wal_and_shm_a_killed_664_writer_left_are_narrowed_with_the_database_and_hold_no_credential_after(tmp_path):
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection

    workspace = _setgid_workspace(tmp_path / "claw")
    database = workspace / "storage" / "coordinator.db"
    _as_this_account(_LEFT_BY_A_KILLED_WRITER.format(database=str(database)), umask=0o022, workspace=workspace)
    sides = [database.with_name(database.name + suffix) for suffix in ("", "-wal", "-shm")]
    assert [_mode(path) for path in sides] == [0o664, 0o664, 0o664], "what the killed writer left"
    connection = SqliteConnection(database, file_mode=0o600)
    try:
        connection.raw.execute("INSERT INTO leftover VALUES ('sk-live-after-the-narrowing')")
        assert [_mode(path) for path in sides] == [0o600, 0o600, 0o600]
        readable = [
            path.name
            for path in database.parent.iterdir()
            if _mode(path) & (stat.S_IRGRP | stat.S_IROTH) and b"sk-live-after" in path.read_bytes()
        ]
        assert readable == []
        assert b"sk-live-after" in (database.with_name(database.name + "-wal")).read_bytes()
    finally:
        connection.raw.close()


def _database_made_664(tmp_path: Path) -> Path:
    """A database a leg without a credential made below shared storage, closed cleanly, with a row in it."""
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection

    database = _setgid_workspace(tmp_path / "claw") / "storage" / "coordinator.db"
    connection = SqliteConnection(database)
    connection.raw.execute("CREATE TABLE leftover (value TEXT)")
    connection.raw.execute("INSERT INTO leftover VALUES ('before')")
    connection.raw.close()
    assert _mode(database) == 0o664
    return database


def _bytes(database: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(database.parent.iterdir()) if path.is_file()}


def _foreign_database(database: Path, monkeypatch) -> Path:
    """The database is 664 and another account's: narrowing it fails ``EPERM``, as it would for this account."""

    def refused(descriptor, mode):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "fchmod", refused)
    return database


def _wal_is_a_link(database: Path, monkeypatch) -> Path:
    victim = database.parent.parent / "somebody-elses-file"
    victim.write_text("kept")
    victim.chmod(0o664)
    database.with_name("coordinator.db-wal").symlink_to(victim)
    return database.with_name("coordinator.db-wal")


def _shm_cannot_be_opened(database: Path, monkeypatch) -> Path:
    side = database.with_name("coordinator.db-shm")
    side.write_bytes(b"left by a killed leg")
    side.chmod(0o664)
    real_open = os.open

    def opened(path, flags, *args, **kwargs):
        if str(path).endswith("-shm"):
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", opened)
    return side


def _database_is_a_link(database: Path, monkeypatch) -> Path:
    real = database.with_name("elsewhere.db")
    database.rename(real)
    database.symlink_to(real)
    return database


_CANNOT_BE_NARROWED = [
    pytest.param(_foreign_database, "not narrowed", id="foreign-owned-fchmod-fails"),
    pytest.param(_wal_is_a_link, "cannot be opened to narrow it", id="side-file-is-a-link"),
    pytest.param(_shm_cannot_be_opened, "cannot be opened to narrow it", id="side-file-unopenable"),
    pytest.param(_database_is_a_link, "cannot be opened", id="database-is-a-link"),
]


@pytest.mark.parametrize(("break_it", "why"), _CANNOT_BE_NARROWED)
def test_a_database_that_must_be_private_and_cannot_be_made_so_is_not_opened_and_nothing_is_written(
    tmp_path, monkeypatch, break_it, why
):
    from hyperloom.orchestrator.bus.storage import connection as bus_connection

    database = _database_made_664(tmp_path)
    named = break_it(database, monkeypatch)
    before = _bytes(database)
    monkeypatch.setattr(
        bus_connection.sqlite3, "connect", lambda *args, **kwargs: pytest.fail("SQLite was opened on a wider file")
    )
    with pytest.raises(bus_connection.DatabaseNotPrivate) as raised:
        bus_connection.SqliteConnection(database, file_mode=0o600)
    assert isinstance(raised.value, PermissionError)
    message = str(raised.value)
    assert message.startswith(f"{named} (owner ") and why in message and "0600" in message, message
    assert "Resume it as the account that owns the file" in message
    assert _bytes(database) == before, "nothing was written, to the database or beside it"


@pytest.mark.parametrize(
    "break_it", [_foreign_database, _wal_is_a_link, _shm_cannot_be_opened], ids=["foreign-owned", "wal-link", "shm"]
)
def test_the_same_failures_only_warn_when_the_database_may_be_664(tmp_path, monkeypatch, caplog, break_it):
    from hyperloom.orchestrator.bus.storage import connection as bus_connection

    database = _database_made_664(tmp_path)
    database.chmod(0o666)  # wider than the shared mode, so that narrowing it is asked
    named = break_it(database, monkeypatch)
    before = _bytes(database)
    with caplog.at_level("WARNING", logger=bus_connection.__name__):
        bus_connection._make_database_file(database, 0o664)
    assert [record.getMessage().split(" (owner ")[0] for record in caplog.records][:1] == [str(named)]
    assert "the session holds a credential" not in caplog.text
    assert _bytes(database) == before


def test_the_lock_a_foreign_account_left_unwritable_is_named_with_its_owner_and_mode(tmp_path, monkeypatch):
    path = lock.session_paths.optimizer_lock_path(tmp_path)
    path.parent.mkdir()
    path.write_text("")
    path.chmod(0o600)
    denied = PermissionError(13, "Permission denied")
    monkeypatch.setattr(lock, "open_shared", lambda *_: (_ for _ in ()).throw(denied))
    with pytest.raises(PermissionError) as raised:
        lock.SessionLock(tmp_path).acquire()
    info = path.stat()
    assert str(raised.value) == (f"[Errno 13] Permission denied: {path} (owner {info.st_uid}:{info.st_gid} 0600)")


def test_an_existing_lock_is_locked_in_place_never_replaced(tmp_path):
    path = lock.session_paths.optimizer_lock_path(tmp_path)
    path.parent.mkdir()
    path.write_text("")
    path.chmod(0o640)
    before = path.stat().st_ino
    with lock.SessionLock(tmp_path):
        assert path.stat().st_ino == before
        with pytest.raises(lock.SessionAlreadyRunning):
            lock.SessionLock(tmp_path).acquire()
    assert (path.stat().st_ino, _mode(path)) == (before, 0o640), "neither replaced nor chmodded"


def test_a_symlink_where_the_lock_belongs_is_refused(tmp_path):
    path = lock.session_paths.optimizer_lock_path(tmp_path)
    path.parent.mkdir()
    path.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(lock.SessionLockPathError, match="must not be a symlink"):
        lock.SessionLock(tmp_path).acquire()
    assert not (tmp_path / "elsewhere").exists()


# -- the next leg is another account -------------------------------------------------------------------------------
_NEXT_LEG = textwrap.dedent(
    """
    import fcntl, json, os, sqlite3
    from pathlib import Path
    session = Path({session!r})
    # lock the session as the optimizer does
    descriptor = os.open(session / "runtime" / "optimizer.lock", os.O_RDWR | os.O_NOFOLLOW)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # the history ledger and a new file in the runtime directory
    with open(session / "runtime" / "pod_history.jsonl", "a", encoding="utf-8") as ledger:
        ledger.write(json.dumps({{"leg": 2}}) + "\\n")
    (session / "runtime" / f"leg-{{os.getuid()}}.txt").write_text("written by the next leg")
    # read the state and the manifest, and replace the state as the application's atomic write does
    from hyperloom_common.io import SHARED, atomic_write_json
    state = json.loads((session / "state.json").read_text())
    manifest = json.loads((session / "manifest.json").read_text())
    atomic_write_json(session / "state.json", state, options=SHARED)
    # write in the database the first leg made
    database = sqlite3.connect(session / "storage" / "coordinator.db", timeout=5)
    database.execute("PRAGMA journal_mode")
    database.execute("CREATE TABLE IF NOT EXISTS leg (n INTEGER)")
    database.execute("INSERT INTO leg VALUES (2)")
    database.commit()
    database.close()
    print(json.dumps({{"schema": manifest["schema_version"]}}))
    """
)


def test_the_next_account_of_the_group_can_lock_read_replace_write_and_open_the_database(tmp_path):
    with other_accounts() as accounts:
        workspace = accounts.shared_dir("claw")
        session = Path(_as_this_account(_FIRST_LEG, umask=0o077, workspace=workspace).splitlines()[-1])
        for account in (accounts.first, accounts.second):
            out = accounts.run(account, _NEXT_LEG.format(session=str(session)), umask=0o077)
            assert json.loads(out) == {"schema": 4}
        assert _mode(session / "state.json") == 0o664
        assert (session / "state.json").stat().st_uid == accounts.second
        for account in (accounts.first, accounts.second):
            assert (session / "runtime" / f"leg-{account}.txt").read_text() == "written by the next leg"
        legs = (session / "runtime" / "pod_history.jsonl").read_text().splitlines()
        assert [json.loads(line).get("leg") for line in legs] == [None, 2, 2]
        rows = sqlite3.connect(session / "storage" / "coordinator.db").execute("SELECT n FROM leg").fetchall()
        assert rows == [(2,), (2,)]


def test_the_lock_still_excludes_across_accounts(tmp_path):
    with other_accounts() as accounts:
        workspace = accounts.shared_dir("claw")
        session = Path(_as_this_account(_FIRST_LEG, umask=0o077, workspace=workspace).splitlines()[-1])
        held = os.open(session / "runtime" / "optimizer.lock", os.O_RDWR)
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            code = (
                "import errno, fcntl, os\n"
                f"fd = os.open({str(session / 'runtime' / 'optimizer.lock')!r}, os.O_RDWR)\n"
                "try:\n    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\nexcept OSError as exc:\n"
                "    print(errno.errorcode[exc.errno])\n"
            )
            assert accounts.run(accounts.second, code).strip() == "EAGAIN"
        finally:
            os.close(held)


def test_without_a_second_account_the_modes_are_what_let_one_in(tmp_path):
    """Where no second account can be had, what lets it in is the permission bits, asserted on the result of a leg."""
    workspace = _setgid_workspace(tmp_path / "claw")
    session = Path(_as_this_account(_FIRST_LEG, umask=0o077, workspace=workspace).splitlines()[-1])
    group_can_write = {
        path.relative_to(session).as_posix() for path in [session, *session.rglob("*")] if _mode(path) & stat.S_IWGRP
    }
    for needed in (
        "runtime",
        "runtime/optimizer.lock",
        "runtime/pod_history.jsonl",
        "storage",
        "storage/coordinator.db",
    ):
        assert needed in group_can_write
