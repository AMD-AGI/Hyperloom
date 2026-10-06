# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A home an older service kept on disk is adopted once, keeping its Experiences, its identity, and what it pulled."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from hyperloom_kb import (
    Change,
    Experience,
    ExperienceDeclaration,
    ExperienceHTTPService,
    ExperienceStatus,
    FieldDeclaration,
    HTTPServiceConfig,
    LocalExperienceStore,
    LocalSchemaRegistry,
    ObjectiveDeclaration,
    ObjectiveDirection,
    Outcome,
    Provenance,
    derive_experience_id,
)
from hyperloom_kb.tests.conftest import fresh_database

TOKEN = "legacy-token"
NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)
LEGACY_KB = "kb-0123456789abcdef0123456789abcdef"
SCHEMA = ExperienceDeclaration(
    identity=(FieldDeclaration("model", "Model."),),
    baseline_identity=(FieldDeclaration("config", "Baseline."),),
    change_identity=(FieldDeclaration("knob", "Knob."),),
    objectives=(ObjectiveDeclaration("throughput@v1", ObjectiveDirection.HIGHER_IS_BETTER, "Throughput."),),
    decisions=("keep", "revert"),
)


def _experience(seq: int) -> Experience:
    return Experience(
        id=derive_experience_id("legacy-test", "run", seq),
        run_id="run",
        seq=seq,
        created_at=NOW,
        completed_at=NOW.replace(hour=seq),
        identity={"model": "qwen3"},
        objective="throughput@v1",
        baseline_identity={"config": "default"},
        baseline_value=100.0,
        provenance=Provenance("legacy-test", "1"),
        schema_ref=SCHEMA.schema_ref,
        status=ExperienceStatus.COMPLETE,
        reasoning=f"Try knob {seq}.",
        change=Change({"knob": f"knob_{seq}"}, f"Set knob {seq}.", kind="config"),
        outcome=Outcome("keep", 110.0 + seq),
        reflection="Measured.",
    )


def _legacy_home(home: Path) -> None:
    """Lay out ``home`` the way a service did before it kept its index and state in a database."""

    canonical = home / "canonical"
    LocalSchemaRegistry(canonical).register_schema(SCHEMA)
    store = LocalExperienceStore(canonical)
    for seq in (2, 0, 1):
        store.insert_complete(_experience(seq))
    (home / "identity.json").write_text(json.dumps({"kb_id": LEGACY_KB}), encoding="utf-8")
    ledger = sqlite3.connect(home / "sync.sqlite3")
    with ledger:
        ledger.execute("CREATE TABLE pulled (experience_id TEXT PRIMARY KEY)")
        ledger.execute("INSERT INTO pulled VALUES (?)", (_experience(2).id,))
    ledger.close()


def _service(home: Path, database) -> ExperienceHTTPService:
    return ExperienceHTTPService(HTTPServiceConfig(home, TOKEN), SCHEMA, None, database=database)


def _writes(database) -> int:
    with database.transaction() as connection:
        row = connection.execute("SELECT COUNT(*) AS n FROM writes").fetchone()
    return int(row["n"])


def test_a_legacy_home_is_adopted_once_with_its_identity_and_what_it_pulled(tmp_path: Path) -> None:
    home, database = tmp_path / "home", fresh_database()
    _legacy_home(home)

    adopted = _service(home, database)
    listed = [item["experience_id"] for item in adopted.list_experiences()["items"]]
    health = adopted.health()
    writes = _writes(database)
    again = _service(home, database)

    assert adopted.kb_id == LEGACY_KB
    assert health["schemas"] == {SCHEMA.schema_ref: 3}
    # Listing names what this KB wrote; the Experience it pulled stays out of it, and so out of every push.
    assert listed == [_experience(0).id, _experience(1).id]
    assert (again.kb_id, _writes(database)) == (LEGACY_KB, writes)


def test_a_new_home_is_a_new_kb(tmp_path: Path) -> None:
    service = _service(tmp_path / "home", fresh_database())

    assert service.kb_id.startswith("kb-") and service.kb_id != LEGACY_KB
    assert service.health()["experience_count"] == 0
