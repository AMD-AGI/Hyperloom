# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A Distill is a versioned entry of its schema's collection: forms check its claim, its scope names declared fields,
every version stays readable, and the schema's state decides which version it shows and whether it is deleted."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hyperloom_kb import (
    ExperienceDeclaration,
    ExperienceHTTPService,
    FieldDeclaration,
    FieldKind,
    FieldRole,
    HTTPServiceConfig,
    ObjectiveDeclaration,
    Provenance,
    SchemaValidationError,
)
from hyperloom_kb.database import _DDL_V1, Database, SqliteDatabase
from hyperloom_kb.distill import (
    CREATED,
    UNCHANGED,
    Condition,
    DistillConflict,
    DistillEntry,
    DistillStore,
    Evidence,
    EvidenceClass,
    EvidenceMember,
    EvidenceStats,
    FormDeclaration,
    Relation,
    RelationKind,
    Relations,
    Scope,
    UnknownDistill,
    new_entry_id,
)
from hyperloom_kb.local_state import LocalState
from hyperloom_kb.storage import StorageContractError
from hyperloom_kb.tests.conftest import fresh_database

TOKEN = "distill-token"
NOW = datetime(2026, 10, 13, tzinfo=timezone.utc)
SCHEMA = ExperienceDeclaration(
    objectives=(ObjectiveDeclaration("throughput@v1", "Maximize throughput."),),
    identity=(FieldDeclaration("model", "Model."), FieldDeclaration("tp", "Tensor parallel.", FieldKind.NUMBER)),
    change=(
        FieldDeclaration("knob", "Knob.", group=True),
        FieldDeclaration("summary", "What changed.", role=FieldRole.SUMMARY),
    ),
    outcome=(
        FieldDeclaration("decision", "Decision.", role=FieldRole.DECISION, values=("keep", "revert")),
        FieldDeclaration("value", "Throughput.", kind=FieldKind.NUMBER, role=FieldRole.MEASUREMENT),
    ),
    reflection=(FieldDeclaration("text", "Reflection.", kind=FieldKind.TEXT),),
)
RULE = FormDeclaration(
    "rule",
    "What a change does under the scope.",
    (
        FieldDeclaration("statement", "What holds.", FieldKind.TEXT, required=True, role=FieldRole.SUMMARY),
        FieldDeclaration("effect", "Its direction.", values=("helps", "hurts", "neutral")),
    ),
)


def _evidence(members: tuple[str, ...] = ("exp-a", "exp-b"), contradicting: tuple[str, ...] = ()) -> Evidence:
    listed = tuple(EvidenceMember(m, EvidenceClass.SUPPORT) for m in members) + tuple(
        EvidenceMember(m, EvidenceClass.CONTRADICT) for m in contradicting
    )
    return Evidence(
        members=listed,
        stats=EvidenceStats(len(members), len(contradicting), 0, 1),
        check=Scope((Condition("outcome.decision", ("revert",)),)),
        representatives=members[:1],
    )


def _entry(
    entry_id: str, version: int = 1, *, statement: str = "Switching attention alone does not help."
) -> DistillEntry:
    return DistillEntry(
        id=entry_id,
        version=version,
        schema_ref=SCHEMA.schema_ref,
        form_ref=RULE.form_ref,
        created_at=NOW,
        provenance=Provenance("distill-test", "1"),
        scope=Scope((Condition("identity.model", ("qwen3",)),)),
        claim={"statement": statement, "effect": "neutral"},
        evidence=_evidence(),
        author="zili",
        tags=("attention",),
    )


@pytest.fixture
def kb(tmp_path: Path) -> tuple[Database, DistillStore, LocalState, ExperienceHTTPService]:
    database = fresh_database()
    app = ExperienceHTTPService(HTTPServiceConfig(tmp_path / "home", TOKEN), SCHEMA, None, database=database)
    state = LocalState(app.kb_id)
    return database, DistillStore(tmp_path / "home", app.kb_id, state), state, app


def _store(kb: tuple[Database, DistillStore, LocalState, ExperienceHTTPService], entry: DistillEntry) -> str:
    database, store, _, _ = kb
    with database.transaction() as connection:
        return store.store(connection, entry, RULE)


# Forms and scope: what a claim may hold and which fields a scope may name.


def test_a_form_is_content_addressed_and_declares_its_claim_within_the_kbs_rules() -> None:
    same = FormDeclaration.from_dict(RULE.to_dict())
    changed = replace(RULE, description="Another wording.")

    assert same.form_ref == RULE.form_ref and changed.form_ref != RULE.form_ref
    assert RULE.summary_field is not None and RULE.summary_field.name == "statement"
    with pytest.raises(SchemaValidationError, match="at least one claim field"):
        FormDeclaration("empty", "No fields.", ())
    with pytest.raises(SchemaValidationError, match="cannot join a group"):
        FormDeclaration("grouped", "Grouped.", (FieldDeclaration("x", "X.", group=True),))
    with pytest.raises(SchemaValidationError, match="only the summary role"):
        FormDeclaration(
            "measured", "Measured.", (FieldDeclaration("x", "X.", FieldKind.NUMBER, role=FieldRole.MEASUREMENT),)
        )
    with pytest.raises(SchemaValidationError, match="required field 'statement'"):
        RULE.validate_claim({"effect": "helps"})
    with pytest.raises(SchemaValidationError, match="does not declare"):
        RULE.validate_claim({"statement": "x", "mechanism": "y"})


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        (Condition("identity.gpu", ("mi300x",)), "not a declared field"),
        (Condition("reflection.text", ("measured",)), "does not hold one scalar"),
        (Condition("identity.tp", ("two",)), "must be a number"),
        (Condition("outcome.decision", ("maybe",)), "must be one of"),
        (Condition("objective", ("latency@v1",)), "is not declared"),
    ],
)
def test_a_scope_names_only_declared_single_scalar_fields_with_values_of_their_kind(
    condition: Condition, message: str
) -> None:
    with pytest.raises(SchemaValidationError, match=message):
        Scope((condition,)).validate(SCHEMA)


def test_a_scope_holds_when_every_condition_holds_and_tells_a_missing_value_apart() -> None:
    scope = Scope((Condition("identity.model", ("qwen3", "llama")), Condition("identity.tp", (2,))))
    scope.validate(SCHEMA)

    def experience(**identity: object) -> object:
        return type("E", (), {"objective": "throughput@v1", "identity": identity})()

    assert scope.holds(experience(model="qwen3", tp=2.0)) is True
    assert scope.holds(experience(model="mistral", tp=2)) is False
    assert scope.holds(experience(model="qwen3")) is None
    assert Scope((Condition("identity.tp", (1,)),)).holds(experience(tp=True)) is False
    assert Scope.from_dict(scope.to_dict()) == scope


# Versions: immutable, appended, each readable against its content.


def test_a_version_round_trips_and_its_evidence_must_agree_with_itself() -> None:
    entry = _entry(new_entry_id())

    assert DistillEntry.from_dict(entry.to_dict()) == entry
    assert set(entry.knowledge()) == {"scope", "claim", "evidence", "relations", "tags", "notes"}
    with pytest.raises(SchemaValidationError, match="do not count"):
        replace(entry.evidence, stats=EvidenceStats(1, 0, 0, 1))
    with pytest.raises(SchemaValidationError, match="representatives must be members"):
        replace(entry.evidence, representatives=("exp-z",))
    with pytest.raises(SchemaValidationError, match="still a member"):
        replace(entry.evidence, removed=("exp-a",))
    with pytest.raises(SchemaValidationError, match="at least one Experience"):
        Evidence(members=(), stats=EvidenceStats(0, 0, 0, 0))
    with pytest.raises(SchemaValidationError, match="relate to itself"):
        replace(entry, relations=Relations((Relation(RelationKind.REFINES, entry.id),)))
    entry.validate(RULE, SCHEMA)
    with pytest.raises(SchemaValidationError, match="form_ref"):
        entry.validate(replace(RULE, name="other"), SCHEMA)


def test_versions_append_under_one_id_and_every_one_stays_readable(kb) -> None:
    database, store, _, _ = kb
    entry_id = new_entry_id()
    first, second = _entry(entry_id), _entry(entry_id, 2, statement="Only on vLLM 0.29.")

    assert _store(kb, first) == CREATED
    assert _store(kb, first) == UNCHANGED
    with pytest.raises(DistillConflict, match="other content"):
        _store(kb, replace(first, tags=("other",)))
    assert _store(kb, second) == CREATED
    with database.transaction() as connection:
        current = store.read(connection, entry_id)
        oldest = store.read(connection, entry_id, 1)
        versions = store.versions(connection, entry_id)
        [row] = store.catalog(connection, SCHEMA.schema_ref)
        forms = store.forms(connection, SCHEMA.schema_ref)
        with pytest.raises(UnknownDistill):
            store.read(connection, entry_id, 3)
        with pytest.raises(UnknownDistill):
            store.schema_of(connection, new_entry_id())

    assert (current, oldest, versions) == (second, first, [1, 2])
    assert (row.entry_id, row.version, row.form_name, row.summary, row.tags) == (
        entry_id,
        2,
        "rule",
        "Only on vLLM 0.29.",
        ("attention",),
    )
    assert (row.supports, row.contradicts, row.runs) == (2, 0, 1)
    assert forms == [RULE]


def test_a_version_may_skip_numbers_as_a_pulled_latest_one_does_but_never_comes_before_the_latest(kb) -> None:
    database, store, _, _ = kb
    entry_id = new_entry_id()

    assert _store(kb, _entry(entry_id, 4)) == CREATED
    with pytest.raises(SchemaValidationError, match="not after its latest, 4"):
        _store(kb, _entry(entry_id, 3))
    with database.transaction() as connection:
        assert store.versions(connection, entry_id) == [4]


def test_a_version_file_is_read_only_against_the_content_the_database_holds(kb, tmp_path: Path) -> None:
    database, store, _, app = kb
    entry_id = new_entry_id()
    _store(kb, _entry(entry_id))
    path = tmp_path / "home" / app.kb_id / "distill" / entry_id / "1.json"
    path.write_text(path.read_text(encoding="utf-8").replace("does not help", "helps"), encoding="utf-8")

    with database.transaction() as connection, pytest.raises(StorageContractError, match="does not hold its stored"):
        store.read(connection, entry_id)
    path.unlink()
    with database.transaction() as connection:
        assert store.check_records(connection) == (1, 1)


# State: deleting is excluding, and a label keeps the version each Distill showed.


def test_deleting_a_distill_hides_it_until_it_is_included_again(kb) -> None:
    database, _, state, _ = kb
    entry_id = new_entry_id()
    _store(kb, _entry(entry_id))

    with database.transaction() as connection:
        state.exclude(connection, SCHEMA.schema_ref, entry_id, "superseded by a narrower scope")
        hidden = state.visible_entries(connection, SCHEMA.schema_ref)
        state.include(connection, SCHEMA.schema_ref, entry_id)
        shown = state.visible_entries(connection, SCHEMA.schema_ref)

    assert (hidden, shown) == (frozenset(), {entry_id})


def test_restoring_a_label_brings_back_the_version_each_distill_showed_and_sets_newer_ones_outside(kb) -> None:
    database, _, state, _ = kb
    kept, later = new_entry_id(), new_entry_id()
    _store(kb, _entry(kept))
    with database.transaction() as connection:
        label = state.label(connection, SCHEMA.schema_ref, name="one rule")
    _store(kb, _entry(kept, 2, statement="Revised."))
    _store(kb, _entry(later))

    with database.transaction() as connection:
        _, modified = state.current(connection, SCHEMA.schema_ref)
        saved = state.restore(connection, label)
        restored = (
            state.current_versions(connection, SCHEMA.schema_ref)[kept],
            state.entry_members(connection, SCHEMA.schema_ref),
        )
    assert saved is not None
    with database.transaction() as connection:
        state.restore(connection, saved)
        undone = (
            state.current_versions(connection, SCHEMA.schema_ref)[kept],
            state.entry_members(connection, SCHEMA.schema_ref),
        )

    assert (label.distill_count, modified, saved.distill_count) == (1, True, 2)
    assert restored == (1, {kept})
    assert undone == (2, {kept, later})


def test_distills_never_change_the_experience_state_a_pull_compares(kb) -> None:
    database, _, state, _ = kb
    entry_id = new_entry_id()
    with database.transaction() as connection:
        before = state.hidden_digest(connection, SCHEMA.schema_ref)
        label = state.label(connection, SCHEMA.schema_ref)
    _store(kb, _entry(entry_id))
    with database.transaction() as connection:
        state.exclude(connection, SCHEMA.schema_ref, entry_id, "wrong")
        state.restore(connection, label)
        after = state.hidden_digest(connection, SCHEMA.schema_ref)

    assert before == after


# Upgrade and observability.


def test_a_database_from_before_distills_upgrades_in_place(tmp_path: Path) -> None:
    path = tmp_path / "database.sqlite3"
    with sqlite3.connect(path) as raw:
        raw.executescript(_DDL_V1.format(serial_key="INTEGER PRIMARY KEY AUTOINCREMENT") + ";")
        raw.execute("CREATE TABLE kb_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
        raw.execute("INSERT INTO kb_migrations VALUES (1, '2026-10-01T00:00:00Z')")
        raw.execute("INSERT INTO kbs(kb_id, created_at) VALUES ('kb-old', '2026-10-01T00:00:00Z')")

    database = SqliteDatabase(path)

    assert database.resolve_kb() == "kb-old"
    with database.transaction() as connection:
        versions = [row["version"] for row in connection.execute("SELECT version FROM kb_migrations ORDER BY version")]
        assert connection.execute("SELECT COUNT(*) AS n FROM distill_versions").fetchone()["n"] == 0
    assert versions == [1, 2]


def test_metrics_count_distills_by_state_and_a_missing_version_file_makes_the_kb_unready(kb, tmp_path: Path) -> None:
    database, _, state, app = kb
    shown, deleted = new_entry_id(), new_entry_id()
    _store(kb, _entry(shown))
    _store(kb, _entry(deleted))
    with database.transaction() as connection:
        state.exclude(connection, SCHEMA.schema_ref, deleted, "wrong")
    gauges = {
        (name, labels.get("state")): value
        for name, labels, value in app.metric_gauges()
        if name == "hyperloom_kb_distill_entries"
    }
    (tmp_path / "home" / app.kb_id / "distill" / shown / "1.json").unlink()
    restarted = ExperienceHTTPService(HTTPServiceConfig(tmp_path / "home", TOKEN), SCHEMA, None, database=database)
    missing = {
        name: value for name, labels, value in restarted.metric_gauges() if name == "hyperloom_kb_records_missing"
    }

    assert gauges == {
        ("hyperloom_kb_distill_entries", "visible"): 1,
        ("hyperloom_kb_distill_entries", "deleted"): 1,
        ("hyperloom_kb_distill_entries", "outside"): 0,
    }
    assert missing == {"hyperloom_kb_records_missing": 1}
