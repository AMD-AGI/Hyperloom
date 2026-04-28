"""Tests for storage/schema.py + storage/connection.py."""

from __future__ import annotations

import sqlite3

from inference_optimizer.storage.connection import open_connection
from inference_optimizer.storage.schema import (
    SCHEMA_VERSION,
    ensure_schema,
    reset_schema,
)


def test_schema_creates_all_four_tables(db_path):
    conn = open_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "ORDER BY name"
        )
        tables = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

    expected = {"leases", "events", "cursors", "tasks", "schema_version",
                "sqlite_sequence"}
    assert expected.issubset(tables), f"missing tables: {expected - tables}"


def test_wal_pragma_active(db_path):
    conn = open_connection(db_path)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"


def test_ensure_schema_is_idempotent(db_path):
    conn = open_connection(db_path)
    try:
        v1 = ensure_schema(conn)
        v2 = ensure_schema(conn)
        assert v1 == v2 == SCHEMA_VERSION
    finally:
        conn.close()


def test_reset_schema_wipes_data(db_path):
    conn = open_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO cursors(agent, last_processed_seq, "
            "last_processed_msg_id, processed_at) VALUES (?,?,?,?)",
            ("executor", 5, "abc", "2026-04-27T00:00:00Z"),
        )
        conn.commit()
        cur = conn.execute("SELECT COUNT(*) FROM cursors")
        assert cur.fetchone()[0] == 1
        reset_schema(conn)
        cur = conn.execute("SELECT COUNT(*) FROM cursors")
        assert cur.fetchone()[0] == 0
    finally:
        conn.close()


def test_event_seq_is_autoincrementing(db_path):
    conn = open_connection(db_path)
    try:
        rows = []
        for i in range(5):
            conn.execute(
                "INSERT INTO events(msg_id, from_agent, to_agent, topic, "
                "in_reply_to, payload, priority, ts) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (f"msg{i}", "exec", "*", "event", None, "{}", 1, "now"),
            )
            rows.append(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()
        # Strict monotonic
        assert rows == sorted(rows)
        assert rows[0] == 1
        assert rows[-1] == 5
    finally:
        conn.close()


def test_cursors_pk_uniqueness(db_path):
    conn = open_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO cursors(agent, last_processed_seq, "
            "last_processed_msg_id, processed_at) VALUES (?,?,?,?)",
            ("executor", 1, "a", "now"),
        )
        try:
            conn.execute(
                "INSERT INTO cursors(agent, last_processed_seq, "
                "last_processed_msg_id, processed_at) VALUES (?,?,?,?)",
                ("executor", 2, "b", "now"),
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("expected IntegrityError on duplicate PK")
    finally:
        conn.close()
