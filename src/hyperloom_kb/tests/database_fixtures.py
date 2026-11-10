# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixtures for tests that serve an Experience KB: a fresh database for every KB a test serves.

A test's databases are SQLite files, as a workspace's service keeps, or, when ``HYPERLOOM_KB_TEST_DATABASE_URL`` names
a PostgreSQL server, databases on that server, so the suite runs on either. A test only PostgreSQL can serve, such as
one of service processes sharing a database, is marked ``requires_postgres``.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator

import pytest

from hyperloom_kb.database import SQLITE_FILE, Database, PostgresDatabase, SqliteDatabase

TEST_DATABASE_URL_ENV = "HYPERLOOM_KB_TEST_DATABASE_URL"

requires_postgres = pytest.mark.skipif(
    not os.environ.get(TEST_DATABASE_URL_ENV, "").strip(),
    reason=f"{TEST_DATABASE_URL_ENV} names no PostgreSQL server",
)


@pytest.fixture(scope="session")
def postgres_conninfo() -> str:
    url = os.environ.get(TEST_DATABASE_URL_ENV, "").strip()
    if not url:
        pytest.skip(f"{TEST_DATABASE_URL_ENV} names no PostgreSQL server")
    return url


@pytest.fixture
def new_database(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Callable[[], Database]]:
    """Make an empty database, migrated and open; every one is closed, and dropped on PostgreSQL, when the test ends."""

    created: list[Database] = []
    dropped: list[str] = []

    def make() -> Database:
        if not os.environ.get(TEST_DATABASE_URL_ENV, "").strip():
            database: Database = SqliteDatabase(tmp_path_factory.mktemp("kb-database") / SQLITE_FILE)
        else:
            import psycopg
            from psycopg.conninfo import make_conninfo

            conninfo = request.getfixturevalue("postgres_conninfo")
            name = f"kb_{uuid.uuid4().hex}"
            with psycopg.connect(conninfo, autocommit=True) as admin:
                admin.execute(f'CREATE DATABASE "{name}"')
            dropped.append(name)
            database = PostgresDatabase(make_conninfo(conninfo, dbname=name))
        created.append(database)
        return database

    yield make
    for database in created:
        database.close()
    if dropped:
        import psycopg

        with psycopg.connect(request.getfixturevalue("postgres_conninfo"), autocommit=True) as admin:
            for name in dropped:
                admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def database_url(new_database: Callable[[], Database]) -> str:
    """The connection string of a fresh PostgreSQL database, for a test that starts service processes of its own."""

    database = new_database()
    assert isinstance(database, PostgresDatabase), "database_url serves tests marked requires_postgres"
    url = database.conninfo
    database.close()
    return url


__all__ = ["TEST_DATABASE_URL_ENV", "database_url", "new_database", "postgres_conninfo", "requires_postgres"]
