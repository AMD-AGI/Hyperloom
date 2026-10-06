# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixtures for tests that serve an Experience KB: one embedded PostgreSQL server per test session, and a fresh
database for every KB a test serves."""

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
from hyperloom_kb.embedded_postgres import start_embedded_postgres


@pytest.fixture(scope="session")
def postgres_conninfo() -> Iterator[str]:
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
def new_database(postgres_conninfo: str) -> Iterator[Callable[[], Database]]:
    """Make an empty database, migrated and open; every one is closed and dropped when the test ends."""

    created: list[tuple[str, Database]] = []

    def make() -> Database:
        name = f"kb_{uuid.uuid4().hex}"
        with psycopg.connect(postgres_conninfo, autocommit=True) as admin:
            admin.execute(f'CREATE DATABASE "{name}"')
        database = Database(make_conninfo(postgres_conninfo, dbname=name))
        created.append((name, database))
        return database

    yield make
    with psycopg.connect(postgres_conninfo, autocommit=True) as admin:
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
