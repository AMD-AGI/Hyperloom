# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixtures for tests that serve an Experience KB: one PostgreSQL server per test session, and a fresh database for
every KB a test serves.

The server is the one ``HYPERLOOM_KB_TEST_DATABASE_URL`` names, or else an embedded one. A test that runs an
embedded server itself, or starts a service that does, is marked ``requires_embedded_postgres``.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from hyperloom_kb.database import Database
from hyperloom_kb.embedded_postgres import start_embedded_postgres, unavailable_reason

TEST_DATABASE_URL_ENV = "HYPERLOOM_KB_TEST_DATABASE_URL"

requires_embedded_postgres = pytest.mark.skipif(
    bool(unavailable_reason()), reason=unavailable_reason() or "the embedded database runs here"
)


@pytest.fixture(scope="session")
def postgres_conninfo() -> Iterator[str]:
    url = os.environ.get(TEST_DATABASE_URL_ENV, "").strip()
    if url:
        yield url
        return
    reason = unavailable_reason()
    if reason:
        pytest.skip(f"{reason}; set {TEST_DATABASE_URL_ENV} to a PostgreSQL server to run this test")
    # A root run serves the database as a dedicated user, which must reach the data directory; pytest's own
    # temporary directories admit only their owner.
    home = Path(tempfile.mkdtemp(prefix="hyperloom-kb-tests-"))
    home.chmod(0o755)
    server = start_embedded_postgres(home)
    try:
        yield server.conninfo
    finally:
        if server.started:
            server.stop()
        shutil.rmtree(home, ignore_errors=True)


@pytest.fixture
def new_database(request: pytest.FixtureRequest) -> Iterator[Callable[[], Database]]:
    """Make an empty database, migrated and open; every one is closed and dropped when the test ends.

    The server is reached only when a test makes its first database, so a test that makes none needs no server.
    """

    created: list[tuple[str, Database]] = []

    def make() -> Database:
        conninfo = request.getfixturevalue("postgres_conninfo")
        name = f"kb_{uuid.uuid4().hex}"
        with psycopg.connect(conninfo, autocommit=True) as admin:
            admin.execute(f'CREATE DATABASE "{name}"')
        database = Database(make_conninfo(conninfo, dbname=name))
        created.append((name, database))
        return database

    yield make
    if not created:
        return
    with psycopg.connect(request.getfixturevalue("postgres_conninfo"), autocommit=True) as admin:
        for name, database in created:
            database.close()
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def database_url(new_database: Callable[[], Database]) -> str:
    """The connection string of a fresh database, for a test that starts a service process of its own."""

    database = new_database()
    url = database.pool.conninfo
    database.close()
    return url


@pytest.fixture
def reachable_tmp_path(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``tmp_path``, traversable by the database user a root run starts a home's embedded database as.

    Only pytest's own directories are opened, and only to traversal, so a service spawned under ``tmp_path`` can
    run its database there.
    """

    if os.geteuid() == 0:
        top = tmp_path_factory.getbasetemp().parent
        for directory in (tmp_path, *tmp_path.parents):
            directory.chmod(directory.stat().st_mode | stat.S_IXOTH)
            if directory == top:
                break
    return tmp_path
