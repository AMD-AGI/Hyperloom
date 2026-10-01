# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which stored Experiences this service's reads see, and the labels that name those states so they can be restored.

A schema's state is its members, the stored Experiences in it, and its exclusions; reads see the members that are
not excluded. Only what a restore set outside the state is recorded, so a home with no record holds every stored
Experience in its state and excludes none.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hyperloom_kb.schema import JsonValue

LABEL_MANUAL = "manual"
LABEL_BEFORE_RESTORE = "before_restore"
LABEL_BEFORE_PULL = "before_pull"
_AUTOMATIC_NAMES = {LABEL_BEFORE_RESTORE: "before restore", LABEL_BEFORE_PULL: "before pull"}


class UnknownStateItem(LookupError):
    """Raised for a label or an Experience this service does not hold."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(members: Collection[str], excluded: Collection[str]) -> str:
    encoded = json.dumps({"members": sorted(members), "excluded": sorted(excluded)}, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class Label:
    """One saved state of one schema; ``label_id`` identifies it, ``name`` is only for people."""

    label_id: str
    schema_ref: str
    name: str
    reason: str
    created_at: str
    member_count: int
    excluded_count: int

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "label_id": self.label_id,
            "schema_ref": self.schema_ref,
            "name": self.name,
            "reason": self.reason,
            "created_at": self.created_at,
            "member_count": self.member_count,
            "excluded_count": self.excluded_count,
        }


class LocalState:
    """Durable per-schema state, exclusions with their history, and labels."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS outside (
                    schema_ref TEXT NOT NULL,
                    experience_id TEXT NOT NULL,
                    PRIMARY KEY (schema_ref, experience_id)
                );
                CREATE TABLE IF NOT EXISTS exclusions (
                    schema_ref TEXT NOT NULL,
                    experience_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    excluded_at TEXT NOT NULL,
                    PRIMARY KEY (schema_ref, experience_id)
                );
                CREATE TABLE IF NOT EXISTS exclusion_history (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    schema_ref TEXT NOT NULL,
                    experience_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS labels (
                    label_id TEXT PRIMARY KEY,
                    schema_ref TEXT NOT NULL,
                    name TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    digest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS label_members (
                    label_id TEXT NOT NULL,
                    experience_id TEXT NOT NULL,
                    PRIMARY KEY (label_id, experience_id)
                );
                CREATE TABLE IF NOT EXISTS label_exclusions (
                    label_id TEXT NOT NULL,
                    experience_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    excluded_at TEXT NOT NULL,
                    PRIMARY KEY (label_id, experience_id)
                );
                CREATE TABLE IF NOT EXISTS current_label (
                    schema_ref TEXT PRIMARY KEY,
                    label_id TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = sqlite3.connect(self.path, timeout=30)
            connection.row_factory = sqlite3.Row
            try:
                with connection:
                    yield connection
            finally:
                connection.close()

    @staticmethod
    def _ids(connection: sqlite3.Connection, query: str, *args: str) -> frozenset[str]:
        return frozenset(str(row[0]) for row in connection.execute(query, args))

    def members(self, schema_ref: str, stored: Collection[str]) -> frozenset[str]:
        with self._connection() as connection:
            outside = self._ids(connection, "SELECT experience_id FROM outside WHERE schema_ref = ?", schema_ref)
        return frozenset(stored) - outside

    def excluded(self, schema_ref: str) -> frozenset[str]:
        with self._connection() as connection:
            return self._ids(connection, "SELECT experience_id FROM exclusions WHERE schema_ref = ?", schema_ref)

    def visible(self, schema_ref: str, stored: Collection[str]) -> frozenset[str]:
        return self.members(schema_ref, stored) - self.excluded(schema_ref)

    def exclude(self, schema_ref: str, experience_id: str, reason: str) -> None:
        at = _now()
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO exclusions(schema_ref, experience_id, reason, excluded_at) VALUES (?, ?, ?, ?)",
                (schema_ref, experience_id, reason, at),
            )
            self._log(connection, schema_ref, experience_id, "exclude", reason, at)

    def include(self, schema_ref: str, experience_id: str) -> bool:
        """Lift an exclusion and release what it withheld; ``False`` when the Experience was neither excluded nor
        withheld."""

        with self._connection() as connection:
            lifted = connection.execute(
                "DELETE FROM exclusions WHERE schema_ref = ? AND experience_id = ?", (schema_ref, experience_id)
            ).rowcount
            released = lifted or experience_id in self._withheld(connection, schema_ref)
            if released:
                self._log(connection, schema_ref, experience_id, "include", "", _now())
        return bool(released)

    @staticmethod
    def _withheld(connection: sqlite3.Connection, schema_ref: str) -> frozenset[str]:
        return LocalState._ids(
            connection,
            """
            SELECT experience_id FROM exclusion_history AS entry
            WHERE schema_ref = ? AND action = 'exclude' AND sequence = (
                SELECT MAX(sequence) FROM exclusion_history
                WHERE schema_ref = entry.schema_ref AND experience_id = entry.experience_id
            )
            """,
            schema_ref,
        )

    def withheld(self, schema_ref: str) -> frozenset[str]:
        """The Experiences last excluded and not included since; a restore that lifts an exclusion releases none."""

        with self._connection() as connection:
            return self._withheld(connection, schema_ref)

    def hidden_digest(self, schema_ref: str) -> str:
        """Changes whenever an exclusion or a restore changes which stored Experiences reads do not see."""

        with self._connection() as connection:
            outside = self._ids(connection, "SELECT experience_id FROM outside WHERE schema_ref = ?", schema_ref)
        return _digest(outside, self.excluded(schema_ref))

    @staticmethod
    def _log(
        connection: sqlite3.Connection, schema_ref: str, experience_id: str, action: str, reason: str, at: str
    ) -> None:
        connection.execute(
            "INSERT INTO exclusion_history(schema_ref, experience_id, action, reason, at) VALUES (?, ?, ?, ?, ?)",
            (schema_ref, experience_id, action, reason, at),
        )

    def exclusions(self, schema_ref: str) -> list[dict[str, JsonValue]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT experience_id, reason, excluded_at FROM exclusions WHERE schema_ref = ? ORDER BY excluded_at",
                (schema_ref,),
            ).fetchall()
        return [dict(row) for row in rows]

    def exclusion_history(self, schema_ref: str) -> list[dict[str, JsonValue]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT experience_id, action, reason, at FROM exclusion_history
                WHERE schema_ref = ? ORDER BY sequence
                """,
                (schema_ref,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _current_digest(self, schema_ref: str, stored: Collection[str]) -> str:
        return _digest(self.members(schema_ref, stored), self.excluded(schema_ref))

    def current(self, schema_ref: str, stored: Collection[str]) -> tuple[Label | None, bool]:
        """The label the state was last saved as or restored to, and whether it changed since."""

        members, excluded = self.members(schema_ref, stored), self.excluded(schema_ref)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT labels.* FROM current_label JOIN labels USING (label_id)
                WHERE current_label.schema_ref = ?
                """,
                (schema_ref,),
            ).fetchone()
            label = None if row is None else self._label(connection, row)
        if row is None:
            return None, bool(members or excluded)
        return label, row["digest"] != _digest(members, excluded)

    def label(self, schema_ref: str, stored: Collection[str], *, name: str = "", reason: str = LABEL_MANUAL) -> Label:
        """Save the current state of ``schema_ref`` under a new label, which becomes its current label."""

        members, excluded = self.members(schema_ref, stored), self.excluded(schema_ref)
        label_id = f"label-{uuid.uuid4().hex}"
        created_at = _now()
        if not name and reason in _AUTOMATIC_NAMES:
            name = f"{_AUTOMATIC_NAMES[reason]} {created_at}"
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO labels(label_id, schema_ref, name, reason, created_at, digest) VALUES (?, ?, ?, ?, ?, ?)",
                (label_id, schema_ref, name, reason, created_at, _digest(members, excluded)),
            )
            connection.executemany(
                "INSERT INTO label_members(label_id, experience_id) VALUES (?, ?)",
                ((label_id, experience_id) for experience_id in sorted(members)),
            )
            connection.execute(
                """
                INSERT INTO label_exclusions(label_id, experience_id, reason, excluded_at)
                SELECT ?, experience_id, reason, excluded_at FROM exclusions WHERE schema_ref = ?
                """,
                (label_id, schema_ref),
            )
            connection.execute(
                "INSERT OR REPLACE INTO current_label(schema_ref, label_id) VALUES (?, ?)", (schema_ref, label_id)
            )
        return self.get_label(label_id)

    def save_if_modified(self, schema_ref: str, stored: Collection[str], reason: str) -> Label | None:
        """Label the current state when it differs from its current label, so the change it is about to undergo
        can be undone."""

        _, modified = self.current(schema_ref, stored)
        return self.label(schema_ref, stored, reason=reason) if modified else None

    @staticmethod
    def _label(connection: sqlite3.Connection, row: sqlite3.Row) -> Label:
        label_id = str(row["label_id"])

        def count(table: str) -> int:
            return int(
                connection.execute(f"SELECT COUNT(*) FROM {table} WHERE label_id = ?", (label_id,)).fetchone()[0]
            )

        return Label(
            label_id=label_id,
            schema_ref=str(row["schema_ref"]),
            name=str(row["name"]),
            reason=str(row["reason"]),
            created_at=str(row["created_at"]),
            member_count=count("label_members"),
            excluded_count=count("label_exclusions"),
        )

    def get_label(self, label_id: str) -> Label:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM labels WHERE label_id = ?", (label_id,)).fetchone()
            if row is None:
                raise UnknownStateItem(f"label {label_id} does not exist")
            return self._label(connection, row)

    def labels(self, schema_ref: str) -> list[Label]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM labels WHERE schema_ref = ? ORDER BY created_at DESC, label_id", (schema_ref,)
            ).fetchall()
            return [self._label(connection, row) for row in rows]

    def delete_label(self, label_id: str) -> None:
        self.get_label(label_id)
        with self._connection() as connection:
            for table in ("label_members", "label_exclusions", "current_label", "labels"):
                connection.execute(f"DELETE FROM {table} WHERE label_id = ?", (label_id,))

    def restore(self, label: Label, stored: Collection[str]) -> Label | None:
        """Make ``label``'s state the current one, first labelling a current state no label holds.

        Returns the label the replaced state was saved under, when it had to be saved.
        """

        saved = self.save_if_modified(label.schema_ref, stored, LABEL_BEFORE_RESTORE)
        with self._connection() as connection:
            members = self._ids(
                connection, "SELECT experience_id FROM label_members WHERE label_id = ?", label.label_id
            )
            connection.execute("DELETE FROM outside WHERE schema_ref = ?", (label.schema_ref,))
            connection.executemany(
                "INSERT INTO outside(schema_ref, experience_id) VALUES (?, ?)",
                ((label.schema_ref, experience_id) for experience_id in sorted(frozenset(stored) - members)),
            )
            connection.execute("DELETE FROM exclusions WHERE schema_ref = ?", (label.schema_ref,))
            connection.execute(
                """
                INSERT INTO exclusions(schema_ref, experience_id, reason, excluded_at)
                SELECT ?, experience_id, reason, excluded_at FROM label_exclusions WHERE label_id = ?
                """,
                (label.schema_ref, label.label_id),
            )
            connection.execute(
                "INSERT OR REPLACE INTO current_label(schema_ref, label_id) VALUES (?, ?)",
                (label.schema_ref, label.label_id),
            )
        return saved

    def bring_in(self, schema_ref: str, experience_ids: Collection[str]) -> None:
        """Put stored Experiences a restore set outside back into the state; their exclusions stand."""

        with self._connection() as connection:
            connection.executemany(
                "DELETE FROM outside WHERE schema_ref = ? AND experience_id = ?",
                ((schema_ref, experience_id) for experience_id in experience_ids),
            )


__all__ = [
    "LABEL_BEFORE_PULL",
    "LABEL_BEFORE_RESTORE",
    "LABEL_MANUAL",
    "Label",
    "LocalState",
    "UnknownStateItem",
]
