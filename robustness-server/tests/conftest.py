"""Shared pytest fixtures.

Tests fall into two layers:

* Pure unit tests construct their own collaborators (fake repos, fake
  ``SafeListFn``, ``httpx.MockTransport`` etc.) and don't need any
  fixture beyond the ``settings`` builder.
* HTTP smoke tests boot the FastAPI app with all background services
  disabled and a stub ``Database`` / ``Repository`` set installed via
  ``monkeypatch``. The ``api_client`` fixture handles that wiring so
  individual tests stay short.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import robustness_server.app as app_module
from robustness_server.config import Settings, reset_settings_for_test
from robustness_server.store import (
    AssignmentsRepository,
    EventsRepository,
    SessionsRepository,
    set_database_for_test,
)
from robustness_server.store.pool import Database


@pytest.fixture
def settings() -> Iterator[Settings]:
    reset_settings_for_test()
    s = Settings(
        database_url="postgresql://test:test@localhost:5432/test",
        apply_migrations_on_start=False,
        nats_servers=["nats://test:4222"],
        enable_nats_consumer=False,
        enable_kv_watcher=False,
        enable_workload_reconciler=False,
    )
    yield s
    reset_settings_for_test()


class FakeAcquireContext:
    """Minimal async context manager mimicking ``asyncpg.Pool.acquire()``."""

    def __init__(self, conn: object) -> None:
        self._conn = conn

    async def __aenter__(self) -> object:
        return self._conn

    async def __aexit__(self, *_: object) -> None:
        return None


class FakePool:
    """Pool stub the repositories can hold without touching real PG."""

    def __init__(self, conn: object) -> None:
        self._conn = conn

    def acquire(self) -> FakeAcquireContext:
        return FakeAcquireContext(self._conn)


class FakeDatabase:
    """Stand-in for ``Database`` used by ``set_database_for_test``."""

    def __init__(self, pool: FakePool) -> None:
        self.pool = pool


@pytest.fixture
def api_client(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> Iterator[TestClient]:
    """FastAPI client backed by stub services.

    Every test that touches the HTTP surface but not Postgres uses
    this fixture. The lifespan is replaced with a no-op (so the real
    asyncpg pool is never opened) and ``set_database_for_test`` wires
    a sentinel database for endpoints that build repositories on
    demand.
    """

    @asynccontextmanager
    async def _noop_lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    monkeypatch.setattr(app_module, "_lifespan", _noop_lifespan)
    set_database_for_test(FakeDatabase(FakePool(object())))
    app = app_module.create_app(settings=settings)
    with TestClient(app) as client:
        yield client
    set_database_for_test(None)


__all__ = [
    "AssignmentsRepository",
    "EventsRepository",
    "FakeAcquireContext",
    "FakeDatabase",
    "FakePool",
    "SessionsRepository",
    "api_client",
    "settings",
]


# Keep these symbols imported so editors/linters don't flag them as
# unused — they are public surface for the rest of the test suite.
_ = Database
_ = AssignmentsRepository
_ = EventsRepository
_ = SessionsRepository
