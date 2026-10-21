from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from hyperloom_kb import (
    Experience,
    ExperienceDeclaration,
    ExperienceStatus,
    FieldDeclaration,
    FieldKind,
    FieldRole,
    ImmutableExperienceConflict,
    InMemoryExperienceStore,
    InsertStatus,
    ObjectiveDeclaration,
    Provenance,
    StorageContractError,
    StoredExperience,
    derive_experience_id,
    experience_content_hash,
)


def declaration(*, objective: str = "throughput@v1") -> ExperienceDeclaration:
    return ExperienceDeclaration(
        objectives=(ObjectiveDeclaration(objective, "Maximize throughput."),),
        identity=(FieldDeclaration("model", "Model."),),
        baseline=(FieldDeclaration("config", "Baseline configuration.", group=True),),
        change=(
            FieldDeclaration("knob", "Changed knob.", group=True),
            FieldDeclaration("summary", "What changed.", role=FieldRole.SUMMARY),
        ),
        outcome=(
            FieldDeclaration("decision", "Decision.", role=FieldRole.DECISION, values=("keep", "revert")),
            FieldDeclaration("value", "Throughput.", kind=FieldKind.NUMBER, role=FieldRole.MEASUREMENT),
        ),
        reflection=(FieldDeclaration("text", "Reflection.", kind=FieldKind.TEXT),),
    )


def complete_experience(
    schema: ExperienceDeclaration | None = None,
    *,
    seq: int = 0,
) -> Experience:
    resolved_schema = schema or declaration()
    now = datetime(2026, 9, 17, tzinfo=timezone.utc)
    return Experience(
        id=derive_experience_id("test", "run-1", seq),
        run_id="run-1",
        seq=seq,
        created_at=now,
        completed_at=now,
        identity={"model": "qwen3"},
        objective="throughput@v1",
        baseline={"config": "default"},
        provenance=Provenance("test", "1"),
        schema_ref=resolved_schema.schema_ref,
        status=ExperienceStatus.COMPLETE,
        rationale={"reasoning": "Test one knob."},
        change={"knob": "page_size", "summary": "Use page size 64."},
        outcome={"decision": "keep", "value": 1.1},
        reflection={"text": "Throughput improved."},
    )


def test_stored_experience_hash_is_backend_neutral() -> None:
    experience = complete_experience()
    digest = experience_content_hash(experience)

    assert StoredExperience(experience, digest).content_hash == digest
    with pytest.raises(StorageContractError, match="content hash"):
        StoredExperience(experience, "0" * 64)


def test_store_conformance_submit_replay_conflict_and_list() -> None:
    store = InMemoryExperienceStore()
    schema = declaration()
    other_schema = declaration(objective="latency@v1")
    experience = complete_experience(schema)

    created = store.insert_complete(experience)
    replay = store.insert_complete(experience)

    assert created.status is InsertStatus.CREATED
    assert replay.status is InsertStatus.UNCHANGED
    assert store.get_experience(experience.id) == created.record
    assert store.list_experiences(schema.schema_ref) == (created.record,)
    assert store.list_experiences(other_schema.schema_ref) == ()

    with pytest.raises(ImmutableExperienceConflict):
        store.insert_complete(replace(experience, reflection={"text": "Different immutable content."}))


def test_store_conformance_rejects_incomplete() -> None:
    experience = complete_experience()

    with pytest.raises(StorageContractError):
        InMemoryExperienceStore().insert_complete(
            replace(
                experience,
                status=ExperienceStatus.IN_PROGRESS,
                completed_at=None,
                rationale={},
                change={},
                outcome={},
                reflection={},
            )
        )


def test_store_conformance_is_atomic_for_equal_threaded_replays() -> None:
    store = InMemoryExperienceStore()
    experience = complete_experience()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(lambda _: store.insert_complete(experience), range(16)))

    assert sum(item.status is InsertStatus.CREATED for item in results) == 1
    assert sum(item.status is InsertStatus.UNCHANGED for item in results) == 15
