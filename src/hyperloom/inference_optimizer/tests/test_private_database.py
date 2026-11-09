# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A coordinator database that holds the operator's credential is kept where no other account of the group can reach it.

The session directory is one the group can write, and SQLite reaches a database, and makes its journal files, by path:
any directory on that path in which another account can rename entries lets it put a database of its own there while
one is opened. So the database of a session that holds a credential lies outside the session directory, in a directory
of this account's below directories only root or this account can write in (a sticky one included); one that is not
such a place is refused before anything is made."""

from __future__ import annotations

import errno
import json
import os
import shutil
import stat
import tempfile
import textwrap
import threading
from pathlib import Path

import pytest

import hyperloom
from hyperloom.inference_optimizer.session.paths import (
    ENV_LEDGER_DIR,
    db_path_for,
    private_db_exists,
    private_db_path_for,
    shared_db_path_for,
)
from hyperloom.inference_optimizer.session.resume_guard import ensure_resume_safe
from hyperloom.inference_optimizer.tools import event_counts
from hyperloom.orchestrator.bus.storage import connection as bus_connection
from hyperloom.orchestrator.bus.storage.connection import DatabaseNotPrivate, SqliteConnection
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.state_file import database_target
from hyperloom_common.tests.shared_accounts import Accounts, other_accounts, theirs

_SECRET = "sk-live-secret-value-1234"
_SRC = str(Path(hyperloom.__file__).parents[1])
_PRIVATE = 0o600


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _session(tmp_path: Path) -> Path:
    """A session in a workspace set up to be shared (setgid, group-writable)."""
    workspace = tmp_path / "claw"
    workspace.mkdir()
    workspace.chmod(0o2775)
    if not workspace.stat().st_mode & stat.S_ISGID:
        pytest.skip("this filesystem does not keep the setgid bit for the test user")
    return workspace / "s"


@pytest.fixture
def own_tmp(tmp_path, monkeypatch) -> Path:
    """This account's temporary directory for the test: where a private database is made."""
    path = tmp_path / "tmp"
    path.mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(path))
    return path


def _write(connection: SqliteConnection, value: str) -> None:
    connection.raw.execute("CREATE TABLE IF NOT EXISTS leftover (value TEXT)")
    connection.raw.execute("INSERT INTO leftover VALUES (?)", (value,))


def _tree(root: Path) -> dict[str, tuple[int, bytes | None]]:
    """Every entry below ``root``: its mode, and its bytes when it is a file."""
    entries = {}
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        data = path.read_bytes() if stat.S_ISREG(info.st_mode) else None
        entries[path.relative_to(root).as_posix()] = (stat.S_IMODE(info.st_mode), data)
    return entries


def _readable_with(root: Path, needle: str) -> list[str]:
    """The files below ``root`` that the group or others can read and that hold ``needle``."""
    return [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file() and _mode(path) & (stat.S_IRGRP | stat.S_IROTH) and needle.encode() in path.read_bytes()
    ]


def _no_sqlite(monkeypatch) -> None:
    monkeypatch.setattr(
        bus_connection.sqlite3, "connect", lambda *args, **kwargs: pytest.fail("SQLite was opened on a refused path")
    )


# -- where the database lives ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("journal_mode", ["WAL", "DELETE"])
def test_a_credentials_database_and_all_beside_it_are_in_a_directory_of_this_account_outside_the_session(
    tmp_path, own_tmp, journal_mode
):
    session = _session(tmp_path)
    where = private_db_path_for(session)
    connection = SqliteConnection(where, journal_mode=journal_mode, file_mode=_PRIVATE)
    try:
        connection.raw.execute("BEGIN IMMEDIATE")  # a rollback journal exists for as long as this is open
        _write(connection, _SECRET)
        directory = where.parent
        assert own_tmp in directory.parents, "below this account's temporary directory"
        assert not session.exists(), "nothing was made in the session directory"
        assert os.stat(directory).st_uid == os.geteuid() and _mode(directory) == 0o700
        names = sorted(path.name for path in directory.iterdir())
        assert names[0] == "coordinator.db" and all(name.startswith("coordinator.db") for name in names)
        assert {_mode(path) for path in directory.iterdir()} == {_PRIVATE}
        assert _readable_with(own_tmp, _SECRET) == [] and _readable_with(tmp_path / "claw", _SECRET) == []
        assert db_path_for(session) == where
        connection.raw.execute("COMMIT")
    finally:
        connection.raw.close()


def test_a_session_without_a_credential_keeps_the_database_in_the_session_for_the_group(tmp_path, own_tmp):
    session = _session(tmp_path)
    where, mode = database_target(session, SharedState())
    assert (where, mode) == (shared_db_path_for(session), 0o664)
    SqliteConnection(where, file_mode=mode).raw.close()
    assert db_path_for(session) == where and list(own_tmp.iterdir()) == []


def test_a_session_that_holds_a_credential_is_given_the_private_path_and_keeps_using_it_without_one(tmp_path, own_tmp):
    session = _session(tmp_path)
    state = SharedState()
    state.operator_extra_env = {"HF_TOKEN": "hf_live_value_123"}
    assert database_target(session, state) == (private_db_path_for(session), _PRIVATE)
    where, mode = database_target(session, state)
    SqliteConnection(where, file_mode=mode).raw.close()
    assert database_target(session, SharedState()) == (private_db_path_for(session), _PRIVATE), "once made, kept"
    assert db_path_for(session) == private_db_path_for(session)


def test_files_planted_in_the_session_beside_where_the_database_used_to_be_are_never_used(tmp_path, own_tmp):
    session = _session(tmp_path)
    storage = session / "storage"
    storage.mkdir(parents=True)
    storage.chmod(0o2775)
    planted = [storage / f"coordinator.db{suffix}" for suffix in ("-journal", "-wal", "-shm")]
    for path in planted:
        path.write_bytes(b"")
        path.chmod(0o666)  # a file another account of the group made and left open to it
    connection = SqliteConnection(private_db_path_for(session), journal_mode="DELETE", file_mode=_PRIVATE)
    try:
        connection.raw.execute("BEGIN IMMEDIATE")
        _write(connection, _SECRET)
        assert [path.stat().st_size for path in planted] == [0, 0, 0]
        connection.raw.execute("COMMIT")
    finally:
        connection.raw.close()
    assert _readable_with(tmp_path / "claw", _SECRET) == []


def test_no_directory_on_the_path_of_a_private_database_can_have_an_entry_renamed_by_another_account(tmp_path, own_tmp):
    where = private_db_path_for(_session(tmp_path))
    SqliteConnection(where, file_mode=_PRIVATE).raw.close()
    for directory in (where.parent, *where.parent.parents):
        info = os.stat(directory)
        assert info.st_uid in (0, os.geteuid()), directory
        assert not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH) or info.st_mode & stat.S_ISVTX, directory


def test_the_session_events_tool_reads_the_private_database(tmp_path, own_tmp, capsys):
    session = _session(tmp_path)
    SqliteConnection(private_db_path_for(session), file_mode=_PRIVATE).raw.close()
    assert event_counts.main([str(session)]) == 0
    assert json.loads(capsys.readouterr().out) == {}


def _a_planted_link(where: Path) -> None:
    target = where.parent.parent / "somebody-elses-file"
    target.write_text("kept")
    where.symlink_to(target)


def _a_planted_directory(where: Path) -> None:
    where.mkdir()


def _a_file_of_another_account(where: Path) -> None:
    if os.geteuid() != 0:
        pytest.skip("a file of another account needs root to make")
    where.write_text("not ours")
    os.chown(where, 60999, 60999)


@pytest.mark.parametrize("plant", [_a_planted_link, _a_planted_directory, _a_file_of_another_account])
def test_an_entry_planted_where_the_private_database_would_be_is_not_taken_for_it(tmp_path, own_tmp, plant):
    """A link, a directory or a file of another account there does not move even a credential-free session onto the
    private path, where the open would refuse it and the read-only tools would read it."""
    session = _session(tmp_path)
    where = private_db_path_for(session)
    where.parent.mkdir(parents=True, mode=0o700)
    plant(where)
    assert not private_db_exists(session)
    assert db_path_for(session) == shared_db_path_for(session)
    assert database_target(session, SharedState()) == (shared_db_path_for(session), 0o664)


def test_a_ledger_directory_that_is_not_absolute_is_refused_by_name(tmp_path, own_tmp, monkeypatch):
    monkeypatch.setenv(ENV_LEDGER_DIR, "relative/ledgers")
    session = _session(tmp_path)
    for call in (private_db_path_for, db_path_for):
        with pytest.raises(ValueError, match="HYPERLOOM_LEDGER_DIR must be an absolute path"):
            call(session)


# -- a place that is not safe is refused before anything is made --------------------------------------------------------
def _group_writable_tmp(own_tmp: Path, monkeypatch) -> Path:
    own_tmp.chmod(0o2775)
    return own_tmp


def _link_to_a_group_writable_directory(own_tmp: Path, monkeypatch) -> Path:
    elsewhere = own_tmp.parent / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    elsewhere.chmod(0o2775)
    (own_tmp / f"hyperloom-ledger-{os.geteuid()}").symlink_to(elsewhere)
    return elsewhere


def _the_directory_is_wider(own_tmp: Path, monkeypatch) -> Path:
    directory = private_db_path_for(own_tmp.parent / "claw" / "s").parent
    directory.mkdir(parents=True, mode=0o700)
    directory.chmod(0o755)
    return directory


@pytest.mark.parametrize(
    ("make", "why"),
    [
        pytest.param(_group_writable_tmp, "can be written by its group or others", id="group-writable-tmp"),
        pytest.param(
            _link_to_a_group_writable_directory,
            "can be written by its group or others",
            id="link-to-a-shared-directory",
        ),
        pytest.param(
            _the_directory_is_wider, "is not a directory of this account's that no other", id="wider-directory"
        ),
    ],
)
def test_a_place_another_account_could_swap_a_directory_in_is_refused_and_nothing_is_made(
    tmp_path, own_tmp, monkeypatch, make, why
):
    session = _session(tmp_path)
    named = make(own_tmp, monkeypatch)
    before = _tree(tmp_path)
    _no_sqlite(monkeypatch)
    monkeypatch.setattr(os, "mkdir", lambda *args, **kwargs: pytest.fail("a directory was made before the refusal"))
    with pytest.raises(DatabaseNotPrivate) as raised:
        SqliteConnection(private_db_path_for(session), file_mode=_PRIVATE)
    assert isinstance(raised.value, PermissionError)
    assert why in str(raised.value) and "Point TMPDIR at one" in str(raised.value), str(raised.value)
    assert str(raised.value).startswith(str(named.resolve())) or str(named.resolve()) in str(raised.value)
    assert _tree(tmp_path) == before, "no directory or file was made"


def test_a_sticky_group_writable_temporary_directory_is_accepted_as_a_shared_tmp_is(tmp_path, own_tmp):
    own_tmp.chmod(0o3775)
    where = private_db_path_for(_session(tmp_path))
    SqliteConnection(where, file_mode=_PRIVATE).raw.close()
    assert where.exists()


def test_a_link_planted_as_the_ledger_root_after_the_path_was_resolved_is_refused_not_judged_by_its_target(
    tmp_path, own_tmp, monkeypatch
):
    """When the ledger root does not exist yet, another account can make it a link of its own between the resolving of the
    path and the checks: a link to a directory that would pass (here one of this account's, ``0700``) must not."""
    session = _session(tmp_path)
    where = private_db_path_for(session)
    safe = tmp_path / "safe"
    safe.mkdir(mode=0o700)
    real_realpath, planted = os.path.realpath, []

    def resolving_then_planting(path, *args, **kwargs):
        resolved = real_realpath(path, *args, **kwargs)
        if Path(path) == where.parent and not planted:
            where.parent.parent.symlink_to(safe)
            planted.append(where.parent.parent)
        return resolved

    monkeypatch.setattr(os.path, "realpath", resolving_then_planting)
    _no_sqlite(monkeypatch)
    monkeypatch.setattr(os, "mkdir", lambda *args, **kwargs: pytest.fail("a directory was made through the link"))
    with pytest.raises(DatabaseNotPrivate) as raised:
        SqliteConnection(where, file_mode=_PRIVATE)
    assert planted and str(raised.value).startswith(f"{planted[0]} (owner ") and "is a link" in str(raised.value)
    assert list(safe.iterdir()) == [], "nothing was written through the link"


def test_a_link_made_where_a_directory_was_about_to_be_made_is_refused_before_anything_is_made_below_it(
    tmp_path, own_tmp, monkeypatch
):
    """The ledger root is made by another account as a link between the looking for it and the making of it: the
    making finds it there, and must not go on to make the directory below it, through the link."""
    session = _session(tmp_path)
    where = private_db_path_for(session)
    safe = tmp_path / "safe"
    safe.mkdir(mode=0o700)
    real_mkdir, attempted = os.mkdir, []

    def planting(path, *args, **kwargs):
        attempted.append(Path(path))
        if Path(path) == where.parent.parent and not where.parent.parent.is_symlink():
            where.parent.parent.symlink_to(safe)
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", planting)
    _no_sqlite(monkeypatch)
    with pytest.raises(DatabaseNotPrivate) as raised:
        SqliteConnection(where, file_mode=_PRIVATE)
    assert "is a link" in str(raised.value)
    assert attempted == [where.parent.parent], "no directory was made below the link, even to be removed again"
    assert list(safe.iterdir()) == [], "nothing was made below the link"


# -- a file in the private directory that cannot be had ---------------------------------------------------------------
def _private_database(tmp_path: Path) -> Path:
    session = _session(tmp_path)
    where = private_db_path_for(session)
    connection = SqliteConnection(where, file_mode=_PRIVATE)
    _write(connection, "before")
    connection.raw.close()
    return where


def _fchmod_fails(where: Path, monkeypatch) -> Path:
    where.chmod(0o644)

    def refused(descriptor, mode):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "fchmod", refused)
    return where


def _wal_is_a_link(where: Path, monkeypatch) -> Path:
    victim = where.parent.parent / "somebody-elses-file"
    victim.write_text("kept")
    victim.chmod(0o664)
    where.with_name("coordinator.db-wal").symlink_to(victim)
    return where.with_name("coordinator.db-wal")


def _shm_is_not_a_file(where: Path, monkeypatch) -> Path:
    os.mkfifo(where.with_name("coordinator.db-shm"))
    return where.with_name("coordinator.db-shm")


def _database_is_theirs(where: Path, monkeypatch) -> Path:
    return theirs(where, monkeypatch)


def _database_is_a_link(where: Path, monkeypatch) -> Path:
    where.rename(where.parent.parent / "elsewhere.db")
    where.symlink_to(where.parent.parent / "elsewhere.db")
    return where


@pytest.mark.parametrize(
    ("break_it", "why"),
    [
        pytest.param(_database_is_theirs, "is not a regular file of this account's", id="database-is-another-accounts"),
        pytest.param(_shm_is_not_a_file, "is not a regular file of this account's", id="side-file-is-a-fifo"),
        pytest.param(_fchmod_fails, "cannot be made owner-only", id="fchmod-fails"),
        pytest.param(_wal_is_a_link, "cannot be opened", id="side-file-is-a-link"),
        pytest.param(_database_is_a_link, "cannot be opened", id="database-is-a-link"),
    ],
)
def test_a_file_in_the_private_directory_that_cannot_be_had_refuses_the_database_and_nothing_is_written(
    tmp_path, own_tmp, monkeypatch, break_it, why
):
    where = _private_database(tmp_path)
    named = break_it(where, monkeypatch)
    before = _tree(tmp_path)
    _no_sqlite(monkeypatch)
    with pytest.raises(DatabaseNotPrivate) as raised:
        SqliteConnection(where, file_mode=_PRIVATE)
    assert str(raised.value).startswith(f"{named} (owner ") and why in str(raised.value), str(raised.value)
    assert _tree(tmp_path) == before


def test_a_refusal_after_the_directories_and_the_database_were_made_removes_them_again(tmp_path, own_tmp, monkeypatch):
    session = _session(tmp_path)
    before = _tree(tmp_path)

    def refused(descriptor, mode):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "fchmod", refused)
    _no_sqlite(monkeypatch)
    with pytest.raises(DatabaseNotPrivate):
        SqliteConnection(private_db_path_for(session), file_mode=_PRIVATE)
    assert _tree(tmp_path) == before and not (own_tmp / f"hyperloom-ledger-{os.geteuid()}").exists()


# -- two accounts of one group -----------------------------------------------------------------------------------------
_LEG = textwrap.dedent(
    """
    import json, os, sys
    sys.dont_write_bytecode = True
    sys.path.insert(0, {src!r})
    os.environ["TMPDIR"] = {tmp!r}
    from pathlib import Path
    from hyperloom.inference_optimizer.session.paths import db_path_for, private_db_path_for, shared_db_path_for
    from hyperloom.orchestrator.bus.storage.connection import DatabaseNotPrivate, SqliteConnection
    session = Path({session!r})
    """
)


def _leg(accounts: Accounts, uid: int, tmp: Path, session: Path, code: str) -> str:
    """Run ``code`` as account ``uid`` with ``tmp`` as its temporary directory, the session and the helpers in scope."""
    head = _LEG.format(src=_SRC, tmp=str(tmp), session=str(session))
    return accounts.run(uid, head + textwrap.dedent(code), umask=0o077)


def _own_tmp_of(accounts: Accounts, uid: int) -> Path:
    """A temporary directory of account ``uid``: its own, 0700, in a directory only root can write in."""
    path = accounts.root / f"tmp-{uid}"
    path.mkdir(mode=0o700)
    os.chown(path, uid, uid)
    return path


def _sticky_tmp(accounts: Accounts) -> Path:
    """A temporary directory shared by the accounts as ``/tmp`` is: world-writable and sticky, root's."""
    path = accounts.shared_dir("tmp")
    path.chmod(0o1777)
    return path


def _reaches_the_checkout(accounts: Accounts) -> bool:
    probe = f"import sys\nsys.dont_write_bytecode = True\nsys.path.insert(0, {_SRC!r})\nimport hyperloom.orchestrator.bus.storage.connection, hyperloom.inference_optimizer.session.paths\n"
    try:
        accounts.run(accounts.second, probe)
    except AssertionError:
        return False
    return True


def test_a_second_account_resumes_with_a_database_of_its_own_and_the_first_is_never_locked_out(own_tmp):
    """Account 2 resumes account 1's credential-free session with a credential: it refuses to touch the database account
    1 made in the session, keeps its own where account 1 cannot read it, and makes nothing in the session, so account 1
    resumes its own session as before. What the earlier leg's database recorded is not carried over."""
    with other_accounts() as accounts:
        if not _reaches_the_checkout(accounts):
            pytest.skip("the other accounts cannot read this checkout or its interpreter's packages")
        first, second = accounts.first, accounts.second
        tmp = {uid: _own_tmp_of(accounts, uid) for uid in (first, second)}
        session = accounts.shared_dir("claw") / "s"
        accounts.run(
            first,
            _LEG.format(src=_SRC, tmp=str(tmp[first]), session=str(session))
            + textwrap.dedent(
                """
                connection = SqliteConnection(shared_db_path_for(session), file_mode=0o664)
                connection.raw.execute("CREATE TABLE leg (name TEXT)")
                connection.raw.execute("INSERT INTO leg VALUES ('first')")
                connection.raw.close()
                """
            ),
            umask=0o077,
        )
        shared_before = sorted((p.relative_to(session).as_posix(), _mode(p)) for p in session.rglob("*"))
        assert ("storage/coordinator.db", 0o664) in shared_before

        out = _leg(
            accounts,
            second,
            tmp[second],
            session,
            f"""
            where = private_db_path_for(session)
            connection = SqliteConnection(where, file_mode=0o600)
            connection.raw.execute("CREATE TABLE leg (name TEXT)")
            connection.raw.execute("INSERT INTO leg VALUES ({_SECRET!r})")
            connection.raw.close()
            print(json.dumps({{"where": str(where)}}))
            """,
        )
        second_database = Path(json.loads(out)["where"])
        assert tmp[second] in second_database.parents and second_database.exists()
        assert sorted((p.relative_to(session).as_posix(), _mode(p)) for p in session.rglob("*")) == shared_before, (
            "nothing was made in the session"
        )
        assert _readable_with(session, _SECRET) == []
        seen = _leg(
            accounts,
            first,
            tmp[first],
            session,
            f"""
            import errno
            kept = db_path_for(session)
            connection = SqliteConnection(kept, file_mode=0o664)
            rows = connection.raw.execute("SELECT name FROM leg").fetchall()
            connection.raw.close()
            try:
                open({str(second_database)!r}, "rb")
                other = "readable"
            except PermissionError as exc:
                other = errno.errorcode[exc.errno]
            print(json.dumps({{"kept": str(kept), "rows": [r[0] for r in rows], "other": other}}))
            """,
        )
        assert json.loads(seen) == {"kept": str(shared_db_path_for(session)), "rows": ["first"], "other": "EACCES"}
        ensure_resume_safe(session, owner_scope="host:1")  # the guard's own view of it, in this process


# renameat2(old, new, RENAME_EXCHANGE): the atomic swap of two directories a member of the group can make wherever it
# can write in both of their parents
_SWAPPER = textwrap.dedent(
    """
    import ctypes, errno, json, os, time
    libc = ctypes.CDLL(None, use_errno=True)
    AT_FDCWD, RENAME_EXCHANGE = -100, 2
    stop, own, marker, root = {stop!r}, {own!r}, {marker!r}, {root!r}
    os.mkdir(own)
    attempts = swapped = 0
    refused = {{}}

    def refuse(number):
        name = errno.errorcode.get(number, str(number))
        refused[name] = refused.get(name, 0) + 1

    deadline = time.monotonic() + 60
    while not os.path.exists(stop) and time.monotonic() < deadline:
        targets = [root]
        if os.path.exists(marker):  # where the database is, as its account reports it
            below = os.path.dirname(open(marker).read())
            for part in os.path.relpath(below, root).split(os.sep):
                targets.append(os.path.join(targets[-1], part))
        for target in targets:
            if not os.path.isdir(target):
                continue
            sibling = os.path.join(os.path.dirname(target), ".swap-" + str(os.getpid()))
            # a directory of its own elsewhere, and a sibling of the target made where it can write
            for old in (own, sibling):
                attempts += 1
                if old == sibling:
                    try:
                        os.mkdir(sibling)
                    except OSError as exc:
                        refuse(exc.errno)
                        continue
                if libc.renameat2(AT_FDCWD, old.encode(), AT_FDCWD, target.encode(), RENAME_EXCHANGE) == 0:
                    swapped += 1
                    libc.renameat2(AT_FDCWD, old.encode(), AT_FDCWD, target.encode(), RENAME_EXCHANGE)
                else:
                    refuse(ctypes.get_errno())
                if old == sibling:
                    os.rmdir(sibling)
        time.sleep(0.002)
    print(json.dumps({{"attempts": attempts, "swapped": swapped, "refused": refused}}))
    """
)

_OPENER = """
rows = 0
for _ in range(40):
    where = private_db_path_for(session)
    connection = SqliteConnection(where, journal_mode="DELETE", file_mode=0o600)
    connection.raw.execute("CREATE TABLE IF NOT EXISTS leg (value TEXT)")
    connection.raw.execute("INSERT INTO leg VALUES (?)", ({secret!r},))
    rows = connection.raw.execute("SELECT count(*) FROM leg").fetchone()[0]
    connection.raw.close()
    if not os.path.exists({marker!r}):
        Path({marker!r}).write_text(str(where))
        os.chmod({marker!r}, 0o644)
print(json.dumps({{"where": str(where), "rows": rows}}))
"""


def test_an_account_exchanging_the_directories_on_the_path_of_a_private_database_gets_nothing(own_tmp):
    """Account 2 loops ``renameat2(RENAME_EXCHANGE)`` of a directory of its own with each directory from the shared
    root down to the one that holds account 1's database (wherever account 1 reports it), while account 1 opens the
    database and writes the credential forty times: no exchange succeeds (a failed one is covered too), and the
    credential is in no file of account 2."""
    with other_accounts() as accounts:
        if not _reaches_the_checkout(accounts):
            pytest.skip("the other accounts cannot read this checkout or its interpreter's packages")
        first, second = accounts.first, accounts.second
        sticky = _sticky_tmp(accounts)
        session = accounts.shared_dir("claw") / "s"
        theirs_dir = Path(tempfile.mkdtemp(prefix="hl-swapper-"))  # a member's own place, outside the shared root
        try:
            os.chown(theirs_dir, 0, accounts.group)
            theirs_dir.chmod(0o2775)
            stop, own, marker = accounts.root / "stop", theirs_dir / "evil", sticky / "where"
            results: dict[str, str] = {}

            def swap() -> None:
                code = _SWAPPER.format(stop=str(stop), own=str(own), marker=str(marker), root=str(accounts.root))
                results["b"] = accounts.run(second, code, umask=0o022)

            _leg(
                accounts,
                first,
                sticky,
                session,
                "from hyperloom_common.shared_storage import make_shared_dirs\nmake_shared_dirs(session / 'storage')",
            )
            swapper = threading.Thread(target=swap)
            swapper.start()
            failure = None
            try:
                opened = _leg(accounts, first, sticky, session, _OPENER.format(secret=_SECRET, marker=str(marker)))
            except AssertionError as exc:  # a database that was swapped away from under it
                opened, failure = "{}", str(exc)
            finally:
                stop.write_text("")
                swapper.join(timeout=90)
            swap_result = json.loads(results["b"])
            assert swap_result["attempts"] > 0 and swap_result["swapped"] == 0, swap_result
            assert set(swap_result["refused"]) <= {"EACCES", "EPERM"}, swap_result
            assert failure is None, failure
            assert json.loads(opened)["rows"] == 40
            assert [p for p in theirs_dir.rglob("*") if p.is_file() and _SECRET.encode() in p.read_bytes()] == []
        finally:
            shutil.rmtree(theirs_dir, ignore_errors=True)


def test_a_member_of_the_group_squatting_the_private_directory_is_refused_and_nothing_is_written_there(own_tmp):
    with other_accounts() as accounts:
        if not _reaches_the_checkout(accounts):
            pytest.skip("the other accounts cannot read this checkout or its interpreter's packages")
        first, second = accounts.first, accounts.second
        sticky = _sticky_tmp(accounts)
        session = accounts.shared_dir("claw") / "s"
        accounts.run(
            second,
            f"import os\nos.mkdir({str(sticky / f'hyperloom-ledger-{first}')!r}, 0o777)\n"
            f"os.chmod({str(sticky / f'hyperloom-ledger-{first}')!r}, 0o777)\n",
        )
        squatted = sticky / f"hyperloom-ledger-{first}"
        said = _leg(
            accounts,
            first,
            sticky,
            session,
            """
            try:
                SqliteConnection(private_db_path_for(session), file_mode=0o600)
                print("opened")
            except DatabaseNotPrivate as exc:
                print(str(exc))
            """,
        )
        assert said.startswith(f"{squatted} (owner {second}:") and "is another account's" in said, said
        assert list(squatted.iterdir()) == [] and not session.exists()


def test_a_temporary_directory_the_group_can_write_in_without_a_sticky_bit_is_refused_and_stays_empty(own_tmp):
    with other_accounts() as accounts:
        if not _reaches_the_checkout(accounts):
            pytest.skip("the other accounts cannot read this checkout or its interpreter's packages")
        loose = accounts.shared_dir("loose")  # root:group 2775, no sticky bit: a member can rename what is in it
        session = accounts.shared_dir("claw") / "s"
        said = _leg(
            accounts,
            accounts.first,
            loose,
            session,
            """
            try:
                SqliteConnection(private_db_path_for(session), file_mode=0o600)
                print("opened")
            except DatabaseNotPrivate as exc:
                print(str(exc))
            """,
        )
        assert said.startswith(f"{loose} (owner 0:{accounts.group} 2775) can be written by its group or others"), said
        assert list(loose.iterdir()) == []
