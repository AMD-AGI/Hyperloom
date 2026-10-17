# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where one KB keeps its Distills: the forms each schema's collection writes claims in, every entry, every version.

The database decides which forms, entries, and versions exist, and holds for each version what a listing shows of it.
A version's content is its record file under ``<root>/<kb_id>/distill``, only ever read against the content hash the
database holds for it. Which version an entry shows, and whether it is deleted, is the schema's state (``LocalState``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hyperloom_kb.database import Connection, utc_now
from hyperloom_kb.distill.model import DistillEntry, FormDeclaration, Scope, require_entry_id
from hyperloom_kb.local_state import LocalState, UnknownStateItem
from hyperloom_kb.schema import JsonValue, SchemaValidationError
from hyperloom_kb.storage import StorageContractError, _atomic_write

DISTILL_DIR = "distill"
CREATED, UNCHANGED = "created", "unchanged"


class UnknownDistill(UnknownStateItem):
    """Raised for a Distill, a version of one, or a form this KB does not hold."""


class DistillConflict(StorageContractError):
    """Raised when a version an entry already has arrives with other content, or an entry under another schema."""


def _json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class DistillFiles:
    def __init__(self, root: Path, kb_id: str) -> None:
        self.directory = root / kb_id / DISTILL_DIR

    def _path(self, entry_id: str, version: int) -> Path:
        return self.directory / require_entry_id(entry_id) / f"{int(version)}.json"

    def write(self, entry_id: str, version: int, data: bytes) -> None:
        _atomic_write(self._path(entry_id, version), data)

    def holds(self, entry_id: str, version: int, size: int) -> bool:
        path = self._path(entry_id, version)
        return path.is_file() and path.stat().st_size == size

    def read(self, entry_id: str, version: int, content_hash: str) -> DistillEntry:
        path = self._path(entry_id, version)
        try:
            entry = DistillEntry.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise StorageContractError(f"cannot read version {version} of {entry_id}: {exc}") from exc
        if entry.content_hash() != content_hash:
            raise StorageContractError(
                f"the record file of version {version} of {entry_id} does not hold its stored content"
            )
        return entry


@dataclass(frozen=True)
class CatalogRow:
    """What a listing shows of an entry's current version, without reading its file."""

    entry_id: str
    version: int
    content_hash: str
    schema_ref: str
    form_ref: str
    form_name: str
    summary: str
    tags: tuple[str, ...]
    scope: Scope
    supports: int
    contradicts: int
    irrelevant: int
    runs: int
    stored_at: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "entry_id": self.entry_id,
            "version": self.version,
            "content_hash": self.content_hash,
            "schema_ref": self.schema_ref,
            "form_ref": self.form_ref,
            "form_name": self.form_name,
            "summary": self.summary,
            "tags": list(self.tags),
            "scope": self.scope.to_dict(),
            "supports": self.supports,
            "contradicts": self.contradicts,
            "irrelevant": self.irrelevant,
            "runs": self.runs,
            "stored_at": self.stored_at,
        }


class DistillStore:
    """The forms, entries, and versions of one KB's Distills; every method runs in the caller's transaction."""

    def __init__(self, root: Path, kb_id: str, state: LocalState) -> None:
        self.kb_id = kb_id
        self._files = DistillFiles(root, kb_id)
        self._state = state

    def register_form(self, connection: Connection, schema_ref: str, form: FormDeclaration) -> bool:
        """Register ``form`` for the collection of ``schema_ref``; ``False`` when it already was."""

        return bool(
            connection.execute(
                """
                INSERT INTO distill_forms(kb_id, schema_ref, form_ref, name, declaration, registered_at)
                VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
                """,
                (self.kb_id, schema_ref, form.form_ref, form.name, _json(form.to_dict()), utc_now()),
            ).rowcount
        )

    def form(self, connection: Connection, schema_ref: str, form_ref: str) -> FormDeclaration:
        row = connection.execute(
            "SELECT declaration FROM distill_forms WHERE kb_id = %s AND schema_ref = %s AND form_ref = %s",
            (self.kb_id, schema_ref, form_ref),
        ).fetchone()
        if row is None:
            raise UnknownDistill(f"form {form_ref} is not registered for {schema_ref}")
        return FormDeclaration.from_dict(json.loads(row["declaration"]))

    def forms(self, connection: Connection, schema_ref: str) -> list[FormDeclaration]:
        """The forms of ``schema_ref``'s collection, by name and then by when each version was registered."""

        rows = connection.execute(
            """
            SELECT declaration FROM distill_forms WHERE kb_id = %s AND schema_ref = %s
            ORDER BY name, registered_at, form_ref
            """,
            (self.kb_id, schema_ref),
        ).fetchall()
        return [FormDeclaration.from_dict(json.loads(row["declaration"])) for row in rows]

    def schema_of(self, connection: Connection, entry_id: str) -> str:
        row = connection.execute(
            "SELECT schema_ref FROM distill_entries WHERE kb_id = %s AND entry_id = %s", (self.kb_id, entry_id)
        ).fetchone()
        if row is None:
            raise UnknownDistill(f"Distill {entry_id} is not held here")
        return str(row["schema_ref"])

    def latest_version(self, connection: Connection, entry_id: str) -> int:
        """The highest version the KB stores of ``entry_id``, 0 for an entry it does not hold."""

        row = connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM distill_versions WHERE kb_id = %s AND entry_id = %s",
            (self.kb_id, entry_id),
        ).fetchone()
        return int(row["version"]) if row else 0

    def store(self, connection: Connection, entry: DistillEntry, form: FormDeclaration) -> str:
        """Store a version the caller validated against ``form`` and its schema, and make it the one the entry shows.

        A version is appended after every version the KB holds of the entry; the same version again is ``unchanged``,
        and one of other content, or of an entry the KB holds under another schema, is a conflict.
        """

        if form.form_ref != entry.form_ref:
            raise SchemaValidationError(f"{entry.id} is written in {entry.form_ref}, not {form.form_ref}")
        data = entry.canonical_bytes()
        content_hash = entry.content_hash()
        existing = connection.execute(
            "SELECT content_hash FROM distill_versions WHERE kb_id = %s AND entry_id = %s AND version = %s",
            (self.kb_id, entry.id, entry.version),
        ).fetchone()
        if existing is not None:
            if existing["content_hash"] != content_hash:
                raise DistillConflict(f"version {entry.version} of {entry.id} already exists with other content")
            return UNCHANGED
        held = connection.execute(
            "SELECT schema_ref FROM distill_entries WHERE kb_id = %s AND entry_id = %s", (self.kb_id, entry.id)
        ).fetchone()
        if held is None:
            connection.execute(
                "INSERT INTO distill_entries(kb_id, entry_id, schema_ref, created_at) VALUES (%s, %s, %s, %s)",
                (self.kb_id, entry.id, entry.schema_ref, utc_now()),
            )
        elif held["schema_ref"] != entry.schema_ref:
            raise DistillConflict(f"{entry.id} is a Distill of {held['schema_ref']}, not {entry.schema_ref}")
        latest = self.latest_version(connection, entry.id)
        if entry.version <= latest:
            raise SchemaValidationError(f"version {entry.version} of {entry.id} is not after its latest, {latest}")
        # A version brings the form it is written in, as a pulled one does from the KB that wrote it.
        self.register_form(connection, entry.schema_ref, form)
        summary_field = form.summary_field
        summary = entry.claim.get(summary_field.name, "") if summary_field is not None else ""
        self._files.write(entry.id, entry.version, data)
        stats = entry.evidence.stats
        connection.execute(
            """
            INSERT INTO distill_versions(kb_id, entry_id, version, schema_ref, form_ref, content_hash, bytes, stored_at,
                form_name, summary, tags, scope, claim, supports, contradicts, irrelevant, runs, representatives)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                self.kb_id,
                entry.id,
                entry.version,
                entry.schema_ref,
                entry.form_ref,
                content_hash,
                len(data),
                utc_now(),
                form.name,
                summary if isinstance(summary, str) else "",
                _json(list(entry.tags)),
                _json(entry.scope.to_dict()),
                _json(entry.to_dict()["claim"]),
                stats.supports,
                stats.contradicts,
                stats.irrelevant,
                stats.runs,
                _json(list(entry.evidence.representatives)),
            ),
        )
        self._state.set_current(connection, entry.id, entry.version)
        return CREATED

    def read(self, connection: Connection, entry_id: str, version: int | None = None) -> DistillEntry:
        """Version ``version`` of ``entry_id``, or the one it shows when ``version`` is ``None``."""

        if version is None:
            row = connection.execute(
                """
                SELECT distill_versions.version, distill_versions.content_hash FROM distill_current
                JOIN distill_versions USING (kb_id, entry_id, version)
                WHERE distill_current.kb_id = %s AND distill_current.entry_id = %s
                """,
                (self.kb_id, entry_id),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT version, content_hash FROM distill_versions WHERE kb_id = %s AND entry_id = %s AND version = %s",
                (self.kb_id, entry_id, version),
            ).fetchone()
        if row is None:
            what = f"Distill {entry_id}" if version is None else f"version {version} of {entry_id}"
            raise UnknownDistill(f"{what} is not held here")
        return self._files.read(entry_id, int(row["version"]), str(row["content_hash"]))

    def versions(self, connection: Connection, entry_id: str) -> list[int]:
        rows = connection.execute(
            "SELECT version FROM distill_versions WHERE kb_id = %s AND entry_id = %s ORDER BY version",
            (self.kb_id, entry_id),
        ).fetchall()
        return [int(row["version"]) for row in rows]

    def catalog(self, connection: Connection, schema_ref: str) -> list[CatalogRow]:
        """What a listing shows of each Distill of ``schema_ref``, at the version it shows, newest entry first."""

        rows = connection.execute(
            """
            SELECT versions.* FROM distill_current
            JOIN distill_versions AS versions USING (kb_id, entry_id, version)
            JOIN distill_entries AS entries USING (kb_id, entry_id)
            WHERE distill_current.kb_id = %s AND entries.schema_ref = %s
            ORDER BY entries.created_at DESC, versions.entry_id
            """,
            (self.kb_id, schema_ref),
        ).fetchall()
        return [
            CatalogRow(
                entry_id=str(row["entry_id"]),
                version=int(row["version"]),
                content_hash=str(row["content_hash"]),
                schema_ref=str(row["schema_ref"]),
                form_ref=str(row["form_ref"]),
                form_name=str(row["form_name"]),
                summary=str(row["summary"]),
                tags=tuple(json.loads(row["tags"])),
                scope=Scope.from_dict(json.loads(row["scope"])),
                supports=int(row["supports"]),
                contradicts=int(row["contradicts"]),
                irrelevant=int(row["irrelevant"]),
                runs=int(row["runs"]),
                stored_at=str(row["stored_at"]),
            )
            for row in rows
        ]

    def check_records(self, connection: Connection) -> tuple[int, int]:
        """How many versions the database holds, and how many of their files are missing or of another size."""

        rows = connection.execute(
            "SELECT entry_id, version, bytes FROM distill_versions WHERE kb_id = %s", (self.kb_id,)
        ).fetchall()
        missing = sum(
            1 for row in rows if not self._files.holds(str(row["entry_id"]), int(row["version"]), int(row["bytes"]))
        )
        return len(rows), missing


__all__ = [
    "CREATED",
    "DISTILL_DIR",
    "UNCHANGED",
    "CatalogRow",
    "DistillConflict",
    "DistillFiles",
    "DistillStore",
    "UnknownDistill",
]
