# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The database contract the service relies on, the same whether a SQLite file or a PostgreSQL server keeps it."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from hyperloom_kb.database import SCHEMA_VERSION, SQLITE_FILE, DatabaseError, SqliteDatabase, open_database
from hyperloom_kb.tests.conftest import fresh_database


def test_a_transaction_that_raises_leaves_nothing_behind() -> None:
    database = fresh_database()
    kb_id = database.resolve_kb()

    with pytest.raises(RuntimeError, match="abandoned"):
        with database.transaction() as connection:
            connection.execute("UPDATE kbs SET last_sequence = %(next)s WHERE kb_id = %(kb)s", {"next": 7, "kb": kb_id})
            raise RuntimeError("abandoned")
    with database.transaction() as connection:
        row = connection.execute("SELECT last_sequence FROM kbs WHERE kb_id = %s", (kb_id,)).fetchone()

    assert row["last_sequence"] == 0


def test_a_database_keeps_its_one_kb_across_every_open() -> None:
    database = fresh_database()

    assert database.resolve_kb() == database.resolve_kb()
    assert database.resolve_kb().startswith("kb-")
    assert database.answers(2.0)


def test_a_workspace_home_keeps_its_database_in_sqlite(tmp_path: Path) -> None:
    first = open_database(tmp_path / "home")
    kb_id = first.resolve_kb()
    reopened = open_database(tmp_path / "home")

    assert isinstance(first, SqliteDatabase)
    assert first.path == (tmp_path / "home" / SQLITE_FILE).resolve()
    assert reopened.resolve_kb() == kb_id
    assert reopened.size_bytes() > 0


def test_a_home_from_main_starts_a_new_kb_and_keeps_main_s_database_unread(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    with closing(sqlite3.connect(home / "kb.sqlite3")) as main_index:
        main_index.execute(
            """
            CREATE TABLE experiences (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                experience_id TEXT NOT NULL UNIQUE,
                schema_ref TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            )
            """
        )
        main_index.execute(
            "INSERT INTO experiences(experience_id, schema_ref, indexed_at) VALUES ('exp-main', 's', 't')"
        )
        main_index.commit()

    kb_id = open_database(home).resolve_kb()

    with closing(sqlite3.connect(home / "kb.sqlite3")) as main_index:
        kept = main_index.execute("SELECT experience_id FROM experiences").fetchall()
    assert kb_id.startswith("kb-")
    assert kept == [("exp-main",)]


def test_a_database_newer_than_this_code_is_refused(tmp_path: Path) -> None:
    path = tmp_path / SQLITE_FILE
    SqliteDatabase(path)
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO kb_migrations(version, applied_at) VALUES (?, 'later')", (SCHEMA_VERSION + 1,))

    with pytest.raises(DatabaseError, match=f"newer than this code's {SCHEMA_VERSION}"):
        SqliteDatabase(path)


def test_an_unreachable_sqlite_home_is_a_database_error(tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("not a directory", encoding="utf-8")

    with pytest.raises(DatabaseError, match="cannot open the Experience KB database"):
        SqliteDatabase(blocked / SQLITE_FILE)
