# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which stored Experiences a KB's reads see, and the labels that name those states so they can be restored.

A schema's state is its members, the stored Experiences in it, and its exclusions; reads see the members that are
not excluded. Only what a restore set outside the state is recorded, so a KB with no record holds every stored
Experience in its state and excludes none. Every method runs in the caller's transaction, so a caller that holds a
schema's lock changes its state atomically.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import psycopg

from hyperloom_kb.database import utc_now
from hyperloom_kb.schema import JsonValue

LABEL_MANUAL = "manual"
LABEL_BEFORE_RESTORE = "before_restore"
LABEL_BEFORE_PULL = "before_pull"
_AUTOMATIC_NAMES = {LABEL_BEFORE_RESTORE: "before restore", LABEL_BEFORE_PULL: "before pull"}

Connection = psycopg.Connection[dict[str, Any]]


class UnknownStateItem(LookupError):
    """Raised for a label or an Experience this KB does not hold."""


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
    """One KB's per-schema state, exclusions with their history, and labels."""

    def __init__(self, kb_id: str) -> None:
        self.kb_id = kb_id

    def _ids(self, connection: Connection, query: str, *args: str) -> frozenset[str]:
        return frozenset(str(row["experience_id"]) for row in connection.execute(query, (self.kb_id, *args)))

    def stored(self, connection: Connection, schema_ref: str) -> frozenset[str]:
        return self._ids(
            connection, "SELECT experience_id FROM experiences WHERE kb_id = %s AND schema_ref = %s", schema_ref
        )

    def _outside(self, connection: Connection, schema_ref: str) -> frozenset[str]:
        return self._ids(
            connection, "SELECT experience_id FROM outside WHERE kb_id = %s AND schema_ref = %s", schema_ref
        )

    def members(self, connection: Connection, schema_ref: str) -> frozenset[str]:
        return self.stored(connection, schema_ref) - self._outside(connection, schema_ref)

    def excluded(self, connection: Connection, schema_ref: str) -> frozenset[str]:
        return self._ids(
            connection, "SELECT experience_id FROM exclusions WHERE kb_id = %s AND schema_ref = %s", schema_ref
        )

    def visible(self, connection: Connection, schema_ref: str) -> frozenset[str]:
        return self.members(connection, schema_ref) - self.excluded(connection, schema_ref)

    def exclude(self, connection: Connection, schema_ref: str, experience_id: str, reason: str) -> None:
        at = utc_now()
        connection.execute(
            """
            INSERT INTO exclusions(kb_id, schema_ref, experience_id, reason, excluded_at) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (kb_id, schema_ref, experience_id) DO UPDATE SET reason = excluded.reason,
                excluded_at = excluded.excluded_at
            """,
            (self.kb_id, schema_ref, experience_id, reason, at),
        )
        self._log(connection, schema_ref, experience_id, "exclude", reason, at)

    def include(self, connection: Connection, schema_ref: str, experience_id: str) -> bool:
        """Lift an exclusion and release what it withheld; ``False`` when the Experience was neither excluded nor
        withheld."""

        lifted = connection.execute(
            "DELETE FROM exclusions WHERE kb_id = %s AND schema_ref = %s AND experience_id = %s",
            (self.kb_id, schema_ref, experience_id),
        ).rowcount
        released = bool(lifted) or experience_id in self.withheld(connection, schema_ref)
        if released:
            self._log(connection, schema_ref, experience_id, "include", "", utc_now())
        return released

    def withheld(self, connection: Connection, schema_ref: str) -> frozenset[str]:
        """The Experiences last excluded and not included since; a restore that lifts an exclusion releases none."""

        return self._ids(
            connection,
            """
            SELECT experience_id FROM (
                SELECT DISTINCT ON (experience_id) experience_id, action FROM exclusion_history
                WHERE kb_id = %s AND schema_ref = %s ORDER BY experience_id, entry_id DESC
            ) AS latest WHERE action = 'exclude'
            """,
            schema_ref,
        )

    def hidden_digest(self, connection: Connection, schema_ref: str) -> str:
        """Changes whenever an exclusion or a restore changes which stored Experiences reads do not see."""

        return _digest(self._outside(connection, schema_ref), self.excluded(connection, schema_ref))

    def _log(
        self, connection: Connection, schema_ref: str, experience_id: str, action: str, reason: str, at: str
    ) -> None:
        connection.execute(
            """
            INSERT INTO exclusion_history(kb_id, schema_ref, experience_id, action, reason, at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (self.kb_id, schema_ref, experience_id, action, reason, at),
        )

    def exclusions(self, connection: Connection, schema_ref: str) -> list[dict[str, JsonValue]]:
        rows = connection.execute(
            """
            SELECT experience_id, reason, excluded_at FROM exclusions WHERE kb_id = %s AND schema_ref = %s
            ORDER BY excluded_at, experience_id
            """,
            (self.kb_id, schema_ref),
        ).fetchall()
        return [dict(row) for row in rows]

    def exclusion_history(self, connection: Connection, schema_ref: str) -> list[dict[str, JsonValue]]:
        rows = connection.execute(
            """
            SELECT experience_id, action, reason, at FROM exclusion_history WHERE kb_id = %s AND schema_ref = %s
            ORDER BY entry_id
            """,
            (self.kb_id, schema_ref),
        ).fetchall()
        return [dict(row) for row in rows]

    def current(self, connection: Connection, schema_ref: str) -> tuple[Label | None, bool]:
        """The label the state was last saved as or restored to, and whether it changed since."""

        members, excluded = self.members(connection, schema_ref), self.excluded(connection, schema_ref)
        row = connection.execute(
            """
            SELECT labels.* FROM current_labels JOIN labels USING (kb_id, label_id)
            WHERE current_labels.kb_id = %s AND current_labels.schema_ref = %s
            """,
            (self.kb_id, schema_ref),
        ).fetchone()
        if row is None:
            return None, bool(members or excluded)
        return self._label(connection, row), row["digest"] != _digest(members, excluded)

    def label(self, connection: Connection, schema_ref: str, *, name: str = "", reason: str = LABEL_MANUAL) -> Label:
        """Save the current state of ``schema_ref`` under a new label, which becomes its current label."""

        members, excluded = self.members(connection, schema_ref), self.excluded(connection, schema_ref)
        label_id = f"label-{uuid.uuid4().hex}"
        created_at = utc_now()
        if not name and reason in _AUTOMATIC_NAMES:
            name = f"{_AUTOMATIC_NAMES[reason]} {created_at}"
        connection.execute(
            """
            INSERT INTO labels(kb_id, label_id, schema_ref, name, reason, created_at, digest)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (self.kb_id, label_id, schema_ref, name, reason, created_at, _digest(members, excluded)),
        )
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO label_members(kb_id, label_id, experience_id) VALUES (%s, %s, %s)",
                [(self.kb_id, label_id, experience_id) for experience_id in sorted(members)],
            )
        connection.execute(
            """
            INSERT INTO label_exclusions(kb_id, label_id, experience_id, reason, excluded_at)
            SELECT kb_id, %s, experience_id, reason, excluded_at FROM exclusions WHERE kb_id = %s AND schema_ref = %s
            """,
            (label_id, self.kb_id, schema_ref),
        )
        self._make_current(connection, schema_ref, label_id)
        return self.get_label(connection, label_id)

    def _make_current(self, connection: Connection, schema_ref: str, label_id: str) -> None:
        connection.execute(
            """
            INSERT INTO current_labels(kb_id, schema_ref, label_id) VALUES (%s, %s, %s)
            ON CONFLICT (kb_id, schema_ref) DO UPDATE SET label_id = excluded.label_id
            """,
            (self.kb_id, schema_ref, label_id),
        )

    def save_if_modified(self, connection: Connection, schema_ref: str, reason: str) -> Label | None:
        """Label the current state when it differs from its current label, so the change it is about to undergo
        can be undone."""

        _, modified = self.current(connection, schema_ref)
        return self.label(connection, schema_ref, reason=reason) if modified else None

    def _label(self, connection: Connection, row: dict[str, Any]) -> Label:
        label_id = str(row["label_id"])

        def count(table: str) -> int:
            counted = connection.execute(
                f"SELECT COUNT(*) AS n FROM {table} WHERE kb_id = %s AND label_id = %s", (self.kb_id, label_id)
            ).fetchone()
            return int(counted["n"]) if counted else 0

        return Label(
            label_id=label_id,
            schema_ref=str(row["schema_ref"]),
            name=str(row["name"]),
            reason=str(row["reason"]),
            created_at=str(row["created_at"]),
            member_count=count("label_members"),
            excluded_count=count("label_exclusions"),
        )

    def get_label(self, connection: Connection, label_id: str) -> Label:
        row = connection.execute(
            "SELECT * FROM labels WHERE kb_id = %s AND label_id = %s", (self.kb_id, label_id)
        ).fetchone()
        if row is None:
            raise UnknownStateItem(f"label {label_id} does not exist")
        return self._label(connection, row)

    def labels(self, connection: Connection, schema_ref: str) -> list[Label]:
        rows = connection.execute(
            "SELECT * FROM labels WHERE kb_id = %s AND schema_ref = %s ORDER BY created_at DESC, label_id",
            (self.kb_id, schema_ref),
        ).fetchall()
        return [self._label(connection, row) for row in rows]

    def delete_label(self, connection: Connection, label_id: str) -> None:
        self.get_label(connection, label_id)
        for table in ("label_members", "label_exclusions", "current_labels", "labels"):
            connection.execute(f"DELETE FROM {table} WHERE kb_id = %s AND label_id = %s", (self.kb_id, label_id))

    def restore(self, connection: Connection, label: Label) -> Label | None:
        """Make ``label``'s state the current one, first labelling a current state no label holds.

        Returns the label the replaced state was saved under, when it had to be saved.
        """

        saved = self.save_if_modified(connection, label.schema_ref, LABEL_BEFORE_RESTORE)
        members = self._ids(
            connection, "SELECT experience_id FROM label_members WHERE kb_id = %s AND label_id = %s", label.label_id
        )
        connection.execute("DELETE FROM outside WHERE kb_id = %s AND schema_ref = %s", (self.kb_id, label.schema_ref))
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO outside(kb_id, schema_ref, experience_id) VALUES (%s, %s, %s)",
                [
                    (self.kb_id, label.schema_ref, experience_id)
                    for experience_id in sorted(self.stored(connection, label.schema_ref) - members)
                ],
            )
        connection.execute(
            "DELETE FROM exclusions WHERE kb_id = %s AND schema_ref = %s", (self.kb_id, label.schema_ref)
        )
        connection.execute(
            """
            INSERT INTO exclusions(kb_id, schema_ref, experience_id, reason, excluded_at)
            SELECT kb_id, %s, experience_id, reason, excluded_at FROM label_exclusions
            WHERE kb_id = %s AND label_id = %s
            """,
            (label.schema_ref, self.kb_id, label.label_id),
        )
        self._make_current(connection, label.schema_ref, label.label_id)
        return saved

    def bring_in(self, connection: Connection, schema_ref: str, experience_ids: Collection[str]) -> None:
        """Put stored Experiences a restore set outside back into the state; their exclusions stand."""

        with connection.cursor() as cursor:
            cursor.executemany(
                "DELETE FROM outside WHERE kb_id = %s AND schema_ref = %s AND experience_id = %s",
                [(self.kb_id, schema_ref, experience_id) for experience_id in experience_ids],
            )


__all__ = [
    "LABEL_BEFORE_PULL",
    "LABEL_BEFORE_RESTORE",
    "LABEL_MANUAL",
    "Label",
    "LocalState",
    "UnknownStateItem",
]
