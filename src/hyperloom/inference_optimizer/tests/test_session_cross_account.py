# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session another account of the group can resume: what a leg creates under a strict umask is ``2775`` / ``664``
whichever account made it, so the next leg (on shared storage, often another host account of one group) can lock it,
read and replace its state, open its database and write beside it. A file the operator's credentials may be in is the
one exception and stays owner-only."""

from __future__ import annotations

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
    done = subprocess.run(
        [sys.executable, "-I", "-c", preamble + code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "USER_DATA_PATH": str(workspace), **env},
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
    assert set(directories.values()) == {0o2775}, directories
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
    assert {_mode(workspace / name) for name in ("runtime", "runtime/recipe_kb", "logs")} == {0o2775}


def test_the_database_files_sqlite_makes_beside_a_664_database_are_664_too(tmp_path):
    workspace = _setgid_workspace(tmp_path / "claw")
    storage = tmp_path / "storage"
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
        ({"operator_server_args": "--tp 8 --api-key not-a-pattern"}, 0o600),
        ({"operator_server_args": "--tp 8 --admin-api-key=not-a-pattern"}, 0o600),
    ],
    ids=["no-pins", "env-knobs", "server-args-knobs", "env-token", "env-bearer", "flag-api-key", "flag-admin-api-key"],
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
        state.save(Path({str(workspace)!r}))
        """
    )
    _as_this_account(code, umask=0o077, workspace=workspace)
    assert _mode(workspace / "state.json") == mode


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
    assert {_mode(session.parent), _mode(session), _mode(workspace / "runtime"), _mode(workspace / "logs")} == {0o2775}


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
