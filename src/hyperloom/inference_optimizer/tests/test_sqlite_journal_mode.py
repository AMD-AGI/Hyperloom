# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session database's journal mode is chosen deliberately, then checked."""

from __future__ import annotations

import sqlite3

import pytest

from hyperloom.orchestrator.bus.storage import connection as conn_mod
from hyperloom.orchestrator.bus.storage.connection import (
    JOURNAL_MODE_ENV,
    JournalModeError,
    SqliteConnection,
    resolve_journal_mode,
)


def test_the_mode_is_resolved_and_rejected_before_any_file_is_opened(monkeypatch):
    """An unrecognised mode never reaches SQLite, which would ignore it."""
    monkeypatch.setenv(JOURNAL_MODE_ENV, "wal")
    assert resolve_journal_mode() == "WAL"

    monkeypatch.setenv(JOURNAL_MODE_ENV, "  delete ")
    assert resolve_journal_mode() == "DELETE"

    monkeypatch.delenv(JOURNAL_MODE_ENV, raising=False)
    assert resolve_journal_mode() == "WAL"

    with pytest.raises(JournalModeError):
        resolve_journal_mode("WALL")


def test_the_requested_mode_is_verified_after_the_database_is_open(tmp_path):
    """Verification reads the mode back rather than trusting the pragma."""
    db = SqliteConnection(tmp_path / "delete.db", journal_mode="DELETE")
    try:
        assert db.journal_mode == "DELETE"
        assert str(db.fetchone_sync("PRAGMA journal_mode")[0]).upper() == "DELETE"
    finally:
        db.close()


def test_a_database_that_declined_the_mode_fails_loudly_instead_of_running_on(tmp_path):
    """An in-memory database cannot enter WAL and does not say so by failing."""
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(JournalModeError):
            conn_mod._apply_pragmas(conn, "WAL")
    finally:
        conn.close()
