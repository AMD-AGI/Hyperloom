# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A coordinator database that holds the operator's credential lives where only its owner can enter.

The session directory is one the group can write, and SQLite makes its journal files by name, so a database kept there
is read through a file another account put beside it first. Below ``storage/`` a directory only this account can enter
holds the database and everything SQLite keeps beside it; one that is there already must be this account's and closed
to everyone else, or the database is not opened."""

from __future__ import annotations

import errno
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
from hyperloom.inference_optimizer.session.paths import PRIVATE_STORAGE_DIRNAME, db_path_for
from hyperloom.inference_optimizer.tools import event_counts
from hyperloom.orchestrator.bus.storage import connection as bus_connection
from hyperloom.orchestrator.bus.storage.connection import DatabaseNotPrivate, SqliteConnection
from hyperloom_common.tests.shared_accounts import other_accounts, theirs

_SECRET = "sk-live-secret-value-1234"
_SRC = str(Path(hyperloom.__file__).parents[1])


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


def _write(connection: SqliteConnection, value: str) -> None:
    connection.raw.execute("CREATE TABLE IF NOT EXISTS leftover (value TEXT)")
    connection.raw.execute("INSERT INTO leftover VALUES (?)", (value,))


def _files(session: Path) -> dict[str, bytes]:
    return {
        path.relative_to(session).as_posix(): path.read_bytes() for path in sorted(session.rglob("*")) if path.is_file()
    }


def _readable_with(session: Path, needle: str) -> list[str]:
    """The files of the session the group or others can read that hold ``needle``."""
    return [
        path.relative_to(session).as_posix()
        for path in sorted(session.rglob("*"))
        if path.is_file() and _mode(path) & (stat.S_IRGRP | stat.S_IROTH) and needle.encode() in path.read_bytes()
    ]


def _no_sqlite(monkeypatch) -> None:
    monkeypatch.setattr(
        bus_connection.sqlite3, "connect", lambda *args, **kwargs: pytest.fail("SQLite was opened on a wider file")
    )


# -- where the database lives ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("journal_mode", ["WAL", "DELETE"])
def test_a_private_database_and_all_beside_it_are_in_a_directory_only_its_owner_can_enter(tmp_path, journal_mode):
    session = _session(tmp_path)
    database = session / "storage" / "coordinator.db"
    connection = SqliteConnection(database, journal_mode=journal_mode, file_mode=0o600)
    try:
        connection.raw.execute("BEGIN IMMEDIATE")  # a rollback journal exists for as long as this is open
        _write(connection, _SECRET)
        private = session / "storage" / PRIVATE_STORAGE_DIRNAME
        assert os.stat(private).st_uid == os.geteuid() and not _mode(private) & 0o077
        assert sorted(path.name for path in (session / "storage").iterdir()) == [PRIVATE_STORAGE_DIRNAME]
        names = sorted(path.name for path in private.iterdir())
        assert names[0] == "coordinator.db" and all(name.startswith("coordinator.db") for name in names)
        assert {_mode(path) for path in private.iterdir()} == {0o600}
        assert _readable_with(session, _SECRET) == []
        assert db_path_for(session) == private / "coordinator.db"
        connection.raw.execute("COMMIT")
    finally:
        connection.raw.close()


def test_a_session_that_never_held_a_credential_keeps_its_database_where_it_was(tmp_path):
    session = _session(tmp_path)
    SqliteConnection(session / "storage" / "coordinator.db").raw.close()
    assert db_path_for(session) == session / "storage" / "coordinator.db"
    assert not (session / "storage" / PRIVATE_STORAGE_DIRNAME).exists()


def test_a_later_leg_without_a_credential_keeps_using_the_private_database(tmp_path):
    session = _session(tmp_path)
    SqliteConnection(session / "storage" / "coordinator.db", file_mode=0o600).raw.close()
    later = SqliteConnection(db_path_for(session))  # the shared mode asked: the database stays where it is, private
    later.raw.close()
    assert (db_path_for(session).parent.name, _mode(db_path_for(session))) == (PRIVATE_STORAGE_DIRNAME, 0o600)
    assert not (session / "storage" / "coordinator.db").exists()


# -- what another account puts beside it ----------------------------------------------------------------------------
@pytest.mark.parametrize("journal_mode", ["WAL", "DELETE"])
def test_files_planted_beside_where_the_database_was_are_never_written(tmp_path, journal_mode):
    session = _session(tmp_path)
    storage = session / "storage"
    storage.mkdir(parents=True)
    storage.chmod(0o2775)
    planted = [storage / f"coordinator.db{suffix}" for suffix in ("-journal", "-wal", "-shm")]
    for path in planted:
        path.write_bytes(b"")
        path.chmod(0o666)  # a file another account of the group made and left open to it
    connection = SqliteConnection(storage / "coordinator.db", journal_mode=journal_mode, file_mode=0o600)
    try:
        connection.raw.execute("BEGIN IMMEDIATE")
        _write(connection, _SECRET)
        assert [path.stat().st_size for path in planted] == [0, 0, 0]
        assert _readable_with(session, _SECRET) == []
        connection.raw.execute("COMMIT")
    finally:
        connection.raw.close()
    assert [path.read_bytes() for path in planted] == [b"", b"", b""]


def test_another_account_of_the_group_cannot_enter_the_directory_to_plant_a_journal_or_read_the_database():
    with other_accounts() as accounts:
        session = accounts.shared_dir("claw") / "s"
        connection = SqliteConnection(session / "storage" / "coordinator.db", journal_mode="DELETE", file_mode=0o600)
        try:
            connection.raw.execute("BEGIN IMMEDIATE")
            _write(connection, _SECRET)
            private = session / "storage" / PRIVATE_STORAGE_DIRNAME
            code = textwrap.dedent(
                f"""
                import errno, json, os
                seen = {{}}
                for what, action in {{
                    "list": lambda: os.listdir({str(private)!r}),
                    "plant": lambda: open({str(private / "coordinator.db-journal")!r}, "wb"),
                    "read": lambda: open({str(private / "coordinator.db")!r}, "rb").read(),
                    "plant-in-storage": lambda: open({str(session / "storage" / "k.db-journal")!r}, "wb").close(),
                }}.items():
                    try:
                        action()
                        seen[what] = "allowed"
                    except OSError as exc:
                        seen[what] = errno.errorcode[exc.errno]
                print(json.dumps(seen))
                """
            )
            seen = json.loads(accounts.run(accounts.second, code))
            assert seen == {"list": "EACCES", "plant": "EACCES", "read": "EACCES", "plant-in-storage": "allowed"}
            connection.raw.execute("COMMIT")
        finally:
            connection.raw.close()
        assert _readable_with(session, _SECRET) == []


# -- a directory that is not the owner's ----------------------------------------------------------------------------
def _wider(private: Path, monkeypatch) -> None:
    private.mkdir(mode=0o700)
    private.chmod(0o777)


def _another_accounts(private: Path, monkeypatch) -> None:
    private.mkdir(mode=0o700)
    monkeypatch.setattr(bus_connection, "_this_account", lambda: os.geteuid() + 1)


def _a_link(private: Path, monkeypatch) -> None:
    (private.parent.parent / "elsewhere").mkdir()
    private.symlink_to(private.parent.parent / "elsewhere")


@pytest.mark.parametrize(
    ("make", "why"),
    [
        pytest.param(_wider, "no other account can enter", id="wider-than-owner-only"),
        pytest.param(_another_accounts, "no other account can enter", id="another-accounts"),
        pytest.param(_a_link, "without following a link", id="a-link"),
    ],
)
def test_an_owner_only_directory_that_is_not_this_accounts_or_is_wider_or_is_a_link_is_refused(
    tmp_path, monkeypatch, make, why
):
    session = _session(tmp_path)
    storage = session / "storage"
    storage.mkdir(parents=True)
    storage.chmod(0o2775)
    private = storage / PRIVATE_STORAGE_DIRNAME
    make(private, monkeypatch)
    before = _files(session)
    _no_sqlite(monkeypatch)
    with pytest.raises(DatabaseNotPrivate) as raised:
        SqliteConnection(storage / "coordinator.db", file_mode=0o600)
    assert isinstance(raised.value, PermissionError)
    assert str(raised.value).startswith(f"{private} (owner ") and why in str(raised.value), str(raised.value)
    assert "Resume it as the account that owns that directory" in str(raised.value)
    assert _files(session) == before, "nothing was written, and no database was made"


# -- a database made before the session held a credential ------------------------------------------------------------
_KILLED_WRITER = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path
    sys.path.insert(0, {src!r})
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
    connection = SqliteConnection(Path({database!r}))
    connection.raw.execute("CREATE TABLE IF NOT EXISTS leftover (value TEXT)")
    connection.raw.execute("INSERT INTO leftover VALUES ('written by the leg that was killed')")
    os._exit(0)
    """
)


def test_a_database_and_the_wal_a_killed_leg_left_are_moved_in_whole_and_what_is_written_next_is_private(tmp_path):
    session = _session(tmp_path)
    database = session / "storage" / "coordinator.db"
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import os\nos.umask(0o022)\n" + _KILLED_WRITER.format(src=_SRC, database=str(database)),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    left = sorted(path.name for path in database.parent.iterdir())
    assert left == ["coordinator.db", "coordinator.db-shm", "coordinator.db-wal"]
    assert {_mode(database.parent / name) for name in left} == {0o664}, "what the killed leg left"

    connection = SqliteConnection(database, file_mode=0o600)
    try:
        _write(connection, _SECRET)
        private = session / "storage" / PRIVATE_STORAGE_DIRNAME
        assert sorted(path.name for path in (session / "storage").iterdir()) == [PRIVATE_STORAGE_DIRNAME]
        assert sorted(path.name for path in private.iterdir()) == left
        assert {_mode(private / name) for name in left} == {0o600}
        assert connection.raw.execute("SELECT value FROM leftover ORDER BY rowid").fetchall()[0][0] == (
            "written by the leg that was killed"
        )
        assert _readable_with(session, _SECRET) == []
    finally:
        connection.raw.close()


def _earlier_database(tmp_path: Path) -> tuple[Path, Path]:
    """A session whose database was made before it held a credential: (the database, the session)."""
    session = _session(tmp_path)
    database = session / "storage" / "coordinator.db"
    first = SqliteConnection(database)
    _write(first, "before")
    first.raw.close()
    return database, session


@pytest.mark.parametrize("foreign", ["coordinator.db", "coordinator.db-journal"])
def test_a_file_beside_an_earlier_database_that_is_not_this_accounts_is_not_moved_and_nothing_is_opened(
    tmp_path, monkeypatch, foreign
):
    """Another account's journal beside the database would be rolled into it when SQLite opened it; its database is not
    this account's to make private."""
    database, session = _earlier_database(tmp_path)
    journal = database.with_name("coordinator.db-journal")
    journal.write_bytes(b"left by another account")
    theirs(database.with_name(foreign), monkeypatch)
    before = _files(session)
    _no_sqlite(monkeypatch)
    with pytest.raises(DatabaseNotPrivate) as raised:
        SqliteConnection(database, file_mode=0o600)
    named = database.with_name(foreign)
    assert str(raised.value).startswith(f"{named} (owner ") and "is not a regular file of this account's" in str(
        raised.value
    )
    assert _files(session) == before, "nothing was moved, narrowed or written"
    assert list((session / "storage" / PRIVATE_STORAGE_DIRNAME).iterdir()) == []


# -- a file in the owner-only directory that cannot be had -----------------------------------------------------------
def _private_database(tmp_path: Path) -> Path:
    session = _session(tmp_path)
    connection = SqliteConnection(session / "storage" / "coordinator.db", file_mode=0o600)
    _write(connection, "before")
    connection.raw.close()
    return session / "storage" / PRIVATE_STORAGE_DIRNAME


def _fchmod_fails(private: Path, monkeypatch) -> Path:
    (private / "coordinator.db").chmod(0o644)

    def refused(descriptor, mode):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "fchmod", refused)
    return private / "coordinator.db"


def _wal_is_a_link(private: Path, monkeypatch) -> Path:
    victim = private.parent.parent / "somebody-elses-file"
    victim.write_text("kept")
    victim.chmod(0o664)
    (private / "coordinator.db-wal").symlink_to(victim)
    return private / "coordinator.db-wal"


def _shm_cannot_be_opened(private: Path, monkeypatch) -> Path:
    (private / "coordinator.db-shm").write_bytes(b"left by a killed leg")
    real_open = os.open

    def opened(path, flags, *args, **kwargs):
        if str(path).endswith("-shm"):
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", opened)
    return private / "coordinator.db-shm"


def _database_is_a_link(private: Path, monkeypatch) -> Path:
    (private / "coordinator.db").rename(private.parent.parent / "elsewhere.db")
    (private / "coordinator.db").symlink_to(private.parent.parent / "elsewhere.db")
    return private / "coordinator.db"


def _database_is_theirs(private: Path, monkeypatch) -> Path:
    return theirs(private / "coordinator.db", monkeypatch)


def _shm_is_not_a_file(private: Path, monkeypatch) -> Path:
    os.mkfifo(private / "coordinator.db-shm")
    return private / "coordinator.db-shm"


@pytest.mark.parametrize(
    ("break_it", "why"),
    [
        pytest.param(_database_is_theirs, "is not a regular file of this account's", id="database-is-another-accounts"),
        pytest.param(_shm_is_not_a_file, "is not a regular file of this account's", id="side-file-is-a-fifo"),
        pytest.param(_fchmod_fails, "cannot be made owner-only", id="fchmod-fails"),
        pytest.param(_wal_is_a_link, "cannot be opened", id="side-file-is-a-link"),
        pytest.param(_shm_cannot_be_opened, "cannot be opened", id="side-file-unopenable"),
        pytest.param(_database_is_a_link, "cannot be opened", id="database-is-a-link"),
    ],
)
def test_a_file_in_the_owner_only_directory_that_cannot_be_had_refuses_the_database_and_nothing_is_written(
    tmp_path, monkeypatch, break_it, why
):
    private = _private_database(tmp_path)
    named = break_it(private, monkeypatch)
    before = _files(private.parent.parent)
    _no_sqlite(monkeypatch)
    with pytest.raises(DatabaseNotPrivate) as raised:
        SqliteConnection(private.parent / "coordinator.db", file_mode=0o600)
    assert str(raised.value).startswith(f"{named} (owner ") and why in str(raised.value), str(raised.value)
    assert _files(private.parent.parent) == before


# -- the directory replaced while the database is opened -------------------------------------------------------------
def test_a_directory_swapped_in_for_the_owner_only_one_before_SQLite_opens_it_is_refused(tmp_path, monkeypatch):
    session = _session(tmp_path)
    storage = session / "storage"
    storage.mkdir(parents=True)
    storage.chmod(0o2775)
    private = storage / PRIVATE_STORAGE_DIRNAME
    real_connect = sqlite3.connect

    def swapped_first(path, *args, **kwargs):
        private.rename(storage / "moved")
        private.mkdir(mode=0o777)
        private.chmod(0o777)
        return real_connect(path, *args, **kwargs)  # makes a database in the directory that was swapped in

    monkeypatch.setattr(bus_connection.sqlite3, "connect", swapped_first)
    with pytest.raises(DatabaseNotPrivate) as raised:
        SqliteConnection(storage / "coordinator.db", file_mode=0o600)
    assert "was replaced while the database was opened" in str(raised.value)


def test_a_directory_swapped_in_after_the_database_is_open_is_never_written_to(tmp_path):
    session = _session(tmp_path)
    storage = session / "storage"
    connection = SqliteConnection(storage / "coordinator.db", journal_mode="DELETE", file_mode=0o600)
    try:
        _write(connection, "before")
        private = storage / PRIVATE_STORAGE_DIRNAME
        private.rename(storage / "moved")
        private.mkdir(mode=0o777)
        private.chmod(0o777)
        with pytest.raises(sqlite3.OperationalError):  # SQLite refuses to write a database that has been moved
            connection.raw.execute("BEGIN IMMEDIATE")
            _write(connection, _SECRET)
        assert list(private.iterdir()) == [], "no journal was made in the directory that was swapped in"
    finally:
        connection.raw.close()


# -- who finds the private database ----------------------------------------------------------------------------------
def test_db_path_for_names_the_private_database_when_this_account_cannot_enter_its_directory(tmp_path, monkeypatch):
    session = tmp_path / "s"
    private = session / "storage" / PRIVATE_STORAGE_DIRNAME / "coordinator.db"
    real_stat = os.stat

    def another_accounts(path, *args, **kwargs):
        if Path(path) == private:
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", another_accounts)
    assert db_path_for(session) == private, "the caller is refused when it opens it; nothing is made beside it"


def test_the_session_events_tool_reads_the_private_database(tmp_path, capsys):
    session = _session(tmp_path)
    SqliteConnection(session / "storage" / "coordinator.db", file_mode=0o600).raw.close()
    assert event_counts.main([str(session)]) == 0
    assert json.loads(capsys.readouterr().out) == {}
