"""asyncpg connection pool wrapper + migration runner."""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

import asyncpg

from ..config import Settings

logger = logging.getLogger(__name__)


class Database:
    """Lifecycle wrapper around an asyncpg pool.

    The pool is opened on ``connect()`` and closed on ``disconnect()``;
    repository helpers obtain connections via the ``acquire`` async
    context manager. Migrations from
    ``robustness_server.store.migrations`` run inside a transaction
    during ``connect()`` when ``settings.apply_migrations_on_start`` is
    set.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database is not connected")
        return self._pool

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=self._settings.database_url,
            min_size=self._settings.database_pool_min,
            max_size=self._settings.database_pool_max,
            server_settings={"search_path": self._settings.database_schema},
            init=self._init_connection,
        )
        if self._settings.apply_migrations_on_start:
            await self._apply_migrations()

    async def disconnect(self) -> None:
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        # search_path must be set per-connection because asyncpg pools
        # may rotate connections across requests; server_settings only
        # applies to new connections, this guards against stale pools.
        await conn.execute(
            f'SET search_path TO "{self._settings.database_schema}", public'
        )

    async def _apply_migrations(self) -> None:
        files = sorted(self._migration_files())
        if not files:
            logger.info("No migrations found in store/migrations")
            return
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for path in files:
                    sql = path.read_text(encoding="utf-8")
                    logger.info("Applying migration: %s", path.name)
                    await conn.execute(sql)

    @staticmethod
    def _migration_files() -> list[Path]:
        package = resources.files("robustness_server.store.migrations")
        out: list[Path] = []
        for entry in package.iterdir():
            # files() yields Traversable; convert to Path when possible
            # so callers can read text uniformly.
            if entry.name.endswith(".sql"):
                with resources.as_file(entry) as p:
                    out.append(p)
        return out


_database: Database | None = None


def get_database() -> Database:
    if _database is None:
        raise RuntimeError("Database has not been initialised; call set_database first")
    return _database


def set_database_for_test(db: Database | None) -> None:
    """Override the singleton; tests use this with a stub or pool."""

    global _database
    _database = db


def install_database(db: Database) -> None:
    """Wire the singleton at startup."""

    global _database
    _database = db
