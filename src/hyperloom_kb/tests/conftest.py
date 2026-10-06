# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every test here may call ``fresh_database()`` for a new, migrated database it does not have to clean up."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from hyperloom_kb.database import Database
from hyperloom_kb.tests.postgres_fixtures import database_url, new_database, postgres_conninfo, reachable_tmp_path

_factory: list[Callable[[], Database]] = []


def fresh_database() -> Database:
    """A new database for the running test, closed and dropped when it ends."""

    if not _factory:
        raise RuntimeError("fresh_database() serves only a running test of hyperloom_kb")
    return _factory[-1]()


@pytest.fixture(autouse=True)
def _fresh_databases(new_database: Callable[[], Database]) -> Iterator[None]:
    _factory.append(new_database)
    try:
        yield
    finally:
        _factory.pop()


__all__ = ["database_url", "fresh_database", "new_database", "postgres_conninfo", "reachable_tmp_path"]
