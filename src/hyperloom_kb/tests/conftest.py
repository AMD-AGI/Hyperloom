# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every test here may call ``fresh_database()`` for a new, migrated database it does not have to clean up."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from hyperloom_kb.database import Database
from hyperloom_kb.tests.database_fixtures import database_url, new_database, postgres_conninfo

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


@pytest.fixture(autouse=True)
def _hold_standard_raw_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete record names the shared test log. Give a service or client those bytes when it asks."""

    from hyperloom_kb.http_service import ExperienceHTTPService
    from hyperloom_kb.remote import RemoteClient
    from hyperloom_kb.tests.logs import LOG, LOG_PATH

    def _names_log(experience: object) -> bool:
        return any(ref.sha256 == LOG.sha256 for ref in getattr(experience, "raw_logs", ()))

    original_write = ExperienceHTTPService.write

    def write(self: ExperienceHTTPService, experience: object, declaration: object = None, **kwargs: object) -> object:
        if _names_log(experience) and not self._files.holds(LOG.sha256, LOG.bytes):
            with LOG_PATH.open("rb") as stream:
                self.put_file(LOG.sha256, LOG.bytes, stream)
        return original_write(self, experience, declaration, **kwargs)  # type: ignore[arg-type]

    original_send = RemoteClient.send_files

    def send_files(self: RemoteClient, experience: object, files: dict[str, object]) -> None:
        merged = dict(files)
        if _names_log(experience):
            merged.setdefault(LOG.sha256, LOG_PATH)
        original_send(self, experience, merged)  # type: ignore[arg-type]

    original_spool = RemoteClient._spool

    def spool(self: RemoteClient, experience: object, body: object, files: dict[str, object]) -> None:
        merged = dict(files)
        if _names_log(experience):
            merged.setdefault(LOG.sha256, LOG_PATH)
        original_spool(self, experience, body, merged)  # type: ignore[arg-type]

    monkeypatch.setattr(ExperienceHTTPService, "write", write)
    monkeypatch.setattr(RemoteClient, "send_files", send_files)
    monkeypatch.setattr(RemoteClient, "_spool", spool)


__all__ = ["database_url", "fresh_database", "new_database", "postgres_conninfo"]
