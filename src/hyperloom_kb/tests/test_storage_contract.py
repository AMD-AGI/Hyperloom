from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import pytest

from hyperloom_kb import (
    CompleteExperienceRequired,
    Experience,
    ExperienceDeclaration,
    ExperienceService,
    ExperienceStatus,
    FieldDeclaration,
    FieldKind,
    FieldRole,
    ImmutableExperienceConflict,
    InMemoryExperienceStore,
    InMemorySchemaRegistry,
    InsertStatus,
    LocalExperienceStore,
    LocalSchemaRegistry,
    ObjectiveDeclaration,
    Provenance,
    StorageContractError,
    StoredExperience,
    UnknownSchemaRef,
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


def _insert_local_process(root: str, experience: Experience, queue: Any) -> None:
    service = ExperienceService(
        LocalSchemaRegistry(root),
        LocalExperienceStore(root),
    )
    try:
        queue.put(service.submit_complete(experience).status.value)
    except (ImmutableExperienceConflict, StorageContractError, OSError) as exc:  # pragma: no cover - parent asserts
        queue.put(f"error:{type(exc).__name__}:{exc}")


def test_stored_experience_hash_is_backend_neutral() -> None:
    experience = complete_experience()
    digest = experience_content_hash(experience)

    assert StoredExperience(experience, digest).content_hash == digest
    with pytest.raises(StorageContractError, match="content hash"):
        StoredExperience(experience, "0" * 64)


@pytest.fixture(params=("memory", "local"))
def service(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> ExperienceService:
    factories: dict[str, Callable[[], ExperienceService]] = {
        "memory": lambda: ExperienceService(
            InMemorySchemaRegistry(),
            InMemoryExperienceStore(),
        ),
        "local": lambda: ExperienceService(
            LocalSchemaRegistry(tmp_path),
            LocalExperienceStore(tmp_path),
        ),
    }
    return factories[str(request.param)]()


def test_store_conformance_register_submit_replay_conflict_and_list(
    service: ExperienceService,
) -> None:
    schema = declaration()
    other_schema = declaration(objective="latency@v1")
    assert service.register_schema(schema) == schema.schema_ref
    assert service.register_schema(schema) == schema.schema_ref
    service.register_schema(other_schema)
    experience = complete_experience(schema)

    created = service.submit_complete(experience)
    replay = service.submit_complete(experience)

    assert created.status is InsertStatus.CREATED
    assert replay.status is InsertStatus.UNCHANGED
    assert service.get_experience(experience.id) == created.record
    assert service.list_experiences(schema.schema_ref) == (created.record,)
    assert service.list_experiences(other_schema.schema_ref) == ()

    with pytest.raises(ImmutableExperienceConflict):
        service.submit_complete(replace(experience, reflection={"text": "Different immutable content."}))


def test_store_conformance_rejects_unknown_schema_and_incomplete(
    service: ExperienceService,
) -> None:
    schema = declaration()
    experience = complete_experience(schema)

    with pytest.raises(UnknownSchemaRef):
        service.submit_complete(experience)

    service.register_schema(schema)
    with pytest.raises(CompleteExperienceRequired):
        service.submit_complete(
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


def test_store_conformance_is_atomic_for_equal_threaded_replays(
    service: ExperienceService,
) -> None:
    schema = declaration()
    service.register_schema(schema)
    experience = complete_experience(schema)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(lambda _: service.submit_complete(experience), range(16)))

    assert sum(item.status is InsertStatus.CREATED for item in results) == 1
    assert sum(item.status is InsertStatus.UNCHANGED for item in results) == 15


def test_local_store_is_atomic_across_processes(tmp_path: Path) -> None:
    schema = declaration()
    ExperienceService(
        LocalSchemaRegistry(tmp_path),
        LocalExperienceStore(tmp_path),
    ).register_schema(schema)
    experience = complete_experience(schema)
    context = get_context("fork")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_insert_local_process,
            args=(str(tmp_path), experience, queue),
        )
        for _ in range(6)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    statuses = [queue.get(timeout=1) for _ in processes]
    assert statuses.count(InsertStatus.CREATED.value) == 1
    assert statuses.count(InsertStatus.UNCHANGED.value) == 5


def test_local_store_never_overwrites_conflicting_process_write(tmp_path: Path) -> None:
    schema = declaration()
    service = ExperienceService(
        LocalSchemaRegistry(tmp_path),
        LocalExperienceStore(tmp_path),
    )
    service.register_schema(schema)
    first = complete_experience(schema)
    conflicting = replace(first, reflection={"text": "Conflicting immutable content."})
    context = get_context("fork")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_insert_local_process,
            args=(str(tmp_path), experience, queue),
        )
        for experience in (first, conflicting)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    statuses = [queue.get(timeout=1) for _ in processes]
    assert statuses.count(InsertStatus.CREATED.value) == 1
    assert sum(item.startswith("error:ImmutableExperienceConflict:") for item in statuses) == 1
    stored = service.get_experience(first.id)
    assert stored is not None
    assert stored.experience in (first, conflicting)
