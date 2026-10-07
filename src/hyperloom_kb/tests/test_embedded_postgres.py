# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A home's embedded PostgreSQL starts once, is reused while it runs, keeps its data, and never widens a directory.

A home it cannot run from, for its permissions or for its file system, is refused with the reason.
"""

from __future__ import annotations

import os
import pwd
import shutil
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from hyperloom_kb.embedded_postgres import (
    DB_USER,
    PGDATA_DIR,
    EmbeddedPostgresError,
    root_run_problem,
    start_embedded_postgres,
)
from hyperloom_kb.tests.postgres_fixtures import requires_embedded_postgres

pytestmark = requires_embedded_postgres
as_root = pytest.mark.skipif(os.geteuid() != 0, reason="only a root caller runs the server as another user")


@pytest.fixture
def reachable_dir() -> Iterator[Path]:
    """A directory every user may traverse, as the database user of a root run needs; pytest's own admit only us."""

    directory = Path(tempfile.mkdtemp(prefix="hyperloom-kb-pg-test-"))
    directory.chmod(0o755)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_a_home_database_starts_once_is_reused_and_keeps_its_data(reachable_dir: Path) -> None:
    home = reachable_dir / "home"
    server = start_embedded_postgres(home)
    try:
        reused = start_embedded_postgres(home)
        with psycopg.connect(server.conninfo, autocommit=True) as connection:
            connection.execute("CREATE TABLE kept (value TEXT)")
            connection.execute("INSERT INTO kept VALUES ('across a restart')")
    finally:
        server.stop()
    restarted = start_embedded_postgres(home)
    try:
        with psycopg.connect(restarted.conninfo) as connection:
            kept = connection.execute("SELECT value FROM kept").fetchall()
    finally:
        restarted.stop()

    assert (server.started, reused.started, restarted.started) == (True, False, True)
    assert reused.conninfo == server.conninfo
    assert kept == [("across a restart",)]


def test_a_home_too_deep_for_a_socket_path_serves_from_a_short_one(reachable_dir: Path) -> None:
    home = reachable_dir / ("d" * 100) / "home"
    server = start_embedded_postgres(home)
    try:
        with psycopg.connect(server.conninfo) as connection:
            connection.execute("SELECT 1")
    finally:
        server.stop()

    assert f"host={home}" not in server.conninfo
    assert "host=/tmp/hyperloom-kb-" in server.conninfo


def test_a_file_system_that_cannot_keep_the_data_directory_private_is_refused(
    reachable_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = reachable_dir / "home"
    (home / PGDATA_DIR).mkdir(parents=True)
    (home / PGDATA_DIR).chmod(0o777)
    monkeypatch.setattr(Path, "chmod", lambda *_args, **_kwargs: None)

    with pytest.raises(EmbeddedPostgresError, match="drwxrwxrwx, but PostgreSQL needs its data directory private"):
        start_embedded_postgres(home)

    assert not (home / PGDATA_DIR / "PG_VERSION").exists()


def test_a_home_other_users_cannot_traverse_is_named_before_a_root_run_tries_it(reachable_dir: Path) -> None:
    private = reachable_dir / "private"
    private.mkdir(mode=0o700)

    assert root_run_problem(private / "kb") == (
        f"the KB home {private / 'kb'} is under {private} (drwx------), which other users may not traverse"
    )
    assert "the KB home" not in root_run_problem(reachable_dir / "kb")
    assert stat.S_IMODE(private.stat().st_mode) == 0o700


@as_root
def test_a_root_caller_runs_the_server_as_the_database_user(reachable_dir: Path) -> None:
    server = start_embedded_postgres(reachable_dir / "home")
    try:
        with psycopg.connect(server.conninfo) as connection:
            connection.execute("SELECT 1")
    finally:
        server.stop()

    assert (reachable_dir / "home" / PGDATA_DIR).stat().st_uid == pwd.getpwnam(DB_USER).pw_uid


@as_root
def test_a_home_the_database_user_cannot_reach_is_refused_and_left_as_it_is(reachable_dir: Path) -> None:
    private = reachable_dir / "private"
    private.mkdir(mode=0o700)

    with pytest.raises(EmbeddedPostgresError, match=f"through {private}"):
        start_embedded_postgres(private / "home")

    assert stat.S_IMODE(private.stat().st_mode) == 0o700


@as_root
def test_a_file_system_that_refuses_to_change_owners_is_named(
    reachable_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def root_squashed(*_args: object) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "chown", root_squashed)

    with pytest.raises(EmbeddedPostgresError, match="refuses ownership changes, as NFS exported with root_squash"):
        start_embedded_postgres(reachable_dir / "home")
