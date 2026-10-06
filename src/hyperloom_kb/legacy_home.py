# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A home served before the Experience KB kept its index and state in a database, read once to adopt it.

Such a home holds its schemas and records under ``canonical/``, its identity in ``identity.json``, and the
Experiences it pulled in ``sync.sqlite3``. Adopting it keeps all three, so its Experiences, its ``kb_id``, and the
rule that pulled Experiences are never pushed back survive the move.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from hyperloom_kb.schema import Experience, ExperienceDeclaration
from hyperloom_kb.storage import LocalExperienceStore, LocalSchemaRegistry, StoredExperience

CANONICAL_DIR = "canonical"
IDENTITY_FILE = "identity.json"
SYNC_LEDGER = "sync.sqlite3"


def _completion_order(record: StoredExperience) -> tuple[str, str]:
    experience = record.experience
    return ((experience.completed_at or experience.created_at).isoformat(), experience.id)


@dataclass(frozen=True)
class LegacyHome:
    home: Path
    kb_id: str

    @classmethod
    def find(cls, home: Path) -> LegacyHome | None:
        """The legacy contents of ``home``, or ``None`` when it holds none."""

        identity = home / IDENTITY_FILE
        if not identity.is_file() and not (home / CANONICAL_DIR).is_dir():
            return None
        kb_id = str(json.loads(identity.read_text(encoding="utf-8"))["kb_id"]) if identity.is_file() else ""
        return cls(home, kb_id)

    def schemas(self) -> tuple[ExperienceDeclaration, ...]:
        return LocalSchemaRegistry(self.home / CANONICAL_DIR).list_schemas()

    def experiences(self) -> list[Experience]:
        """Every stored Experience, in the order they were completed, which is the order a fresh index writes."""

        store = LocalExperienceStore(self.home / CANONICAL_DIR)
        records = [record for schema in self.schemas() for record in store.list_experiences(schema.schema_ref)]
        return [record.experience for record in sorted(records, key=_completion_order)]

    def pulled(self) -> frozenset[str]:
        path = self.home / SYNC_LEDGER
        if not path.is_file():
            return frozenset()
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = connection.execute("SELECT experience_id FROM pulled").fetchall()
        except sqlite3.OperationalError:
            return frozenset()
        finally:
            connection.close()
        return frozenset(str(row[0]) for row in rows)


__all__ = ["LegacyHome"]
