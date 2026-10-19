from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from hyperloom_kb import (
    Experience,
    ExperienceDeclaration,
    ExperienceService,
    ExperienceStatus,
    FieldDeclaration,
    FieldKind,
    FieldRole,
    InMemoryExperienceStore,
    InMemoryQueryViewStore,
    InMemorySchemaRegistry,
    LocalQueryViewStore,
    ObjectiveDeclaration,
    Provenance,
    QueryViewBuilder,
    QueryViewMaintainer,
    derive_experience_id,
    experience_content_hash,
    repeat_group_key,
)
from hyperloom_kb.storage import StoredExperience

NOW = datetime(2026, 9, 17, tzinfo=timezone.utc)


def declaration(*, objective: str = "throughput@v1") -> ExperienceDeclaration:
    return ExperienceDeclaration(
        objectives=(ObjectiveDeclaration(objective, "Maximize throughput."),),
        identity=(
            FieldDeclaration("model", "Model."),
            FieldDeclaration("gpu", "GPU."),
        ),
        baseline=(
            FieldDeclaration("config", "Baseline configuration.", group=True),
            FieldDeclaration("value", "Baseline throughput.", kind=FieldKind.NUMBER, role=FieldRole.MEASUREMENT),
        ),
        change=(
            FieldDeclaration("knob", "Changed knob.", group=True),
            FieldDeclaration("summary", "What changed.", role=FieldRole.SUMMARY),
            FieldDeclaration("content", "The change.", kind=FieldKind.TEXT),
        ),
        outcome=(
            FieldDeclaration("decision", "Decision.", role=FieldRole.DECISION, values=("keep", "revert", "failed")),
            FieldDeclaration("value", "Throughput.", kind=FieldKind.NUMBER, role=FieldRole.MEASUREMENT),
            FieldDeclaration("accuracy_passed", "Whether the accuracy gate passed.", kind=FieldKind.BOOLEAN),
            FieldDeclaration("gates", "Gates checked.", many=True),
        ),
        reflection=(FieldDeclaration("text", "Reflection.", kind=FieldKind.TEXT),),
    )


def experience(
    schema: ExperienceDeclaration,
    *,
    run_id: str,
    seq: int,
    knob: str = "page_size",
    baseline: str = "default",
    decision: str = "keep",
    value: float = 110.0,
    accuracy_passed: bool = True,
) -> Experience:
    return Experience(
        id=derive_experience_id("test", run_id, seq),
        run_id=run_id,
        seq=seq,
        created_at=NOW,
        completed_at=NOW,
        identity={"model": "qwen3", "gpu": "mi355x"},
        objective=schema.objectives[0].id,
        baseline={"config": baseline, "value": 100.0},
        provenance=Provenance("test", "1"),
        schema_ref=schema.schema_ref,
        status=ExperienceStatus.COMPLETE,
        rationale={"reasoning": f"Test {knob} against the measured bottleneck."},
        change={"knob": knob, "summary": f"Change {knob}.", "content": f"--{knob} 32"},
        outcome={"decision": decision, "value": value, "accuracy_passed": accuracy_passed, "gates": ("accuracy",)},
        reflection={"text": f"{decision} after measurement."},
    )


def stored(*items: Experience) -> tuple[StoredExperience, ...]:
    return tuple(StoredExperience(item, experience_content_hash(item)) for item in items)


def test_query_view_builds_stable_groups_annotations_and_lookup() -> None:
    schema = declaration()
    first = experience(schema, run_id="run-1", seq=0, decision="keep", value=110)
    repeated = experience(schema, run_id="run-2", seq=0, decision="revert", value=95, accuracy_passed=False)
    other = experience(schema, run_id="run-3", seq=0, knob="chunk", value=115)

    view = QueryViewBuilder().build(schema, stored(other, repeated, first))
    replay = QueryViewBuilder().build(schema, stored(first, other, repeated))

    assert view == replay
    assert view.ref.local_page_sequence == 3
    assert len(view.groups) == 2
    group = view.groups[repeat_group_key(schema, first)]
    assert group.member_ids == tuple(sorted((first.id, repeated.id)))
    assert group.annotations.member_count == 2
    assert group.annotations.distinct_run_count == 2
    assert group.annotations.decision_counts == {"keep": 1, "revert": 1}
    assert group.annotations.measurement_median == 102.5
    assert group.annotations.measurement_variance == 56.25
    assert group.annotations.flag_counts == {"accuracy_passed": {"true": 1, "false": 1}}
    model_hits = view.field_lookup["identity.model"]['"qwen3"']
    assert model_hits == tuple(sorted((first.id, repeated.id, other.id)))
    assert view.field_lookup["change.knob"]['"chunk"'] == (other.id,)
    assert view.field_lookup["outcome.decision"]['"revert"'] == (repeated.id,)


def test_exact_lookup_indexes_single_scalars_and_every_identity_key() -> None:
    schema = declaration()
    tagged = replace(
        experience(schema, run_id="run-1", seq=0),
        identity={"model": "qwen3", "gpu": "mi355x", "tenant": "lab-a"},
    )

    view = QueryViewBuilder().build(schema, stored(tagged))

    assert view.field_lookup["identity.tenant"]['"lab-a"'] == (tagged.id,)
    for unindexed in ("change.content", "outcome.gates", "rationale.reasoning", "reflection.text"):
        assert unindexed not in view.field_lookup


def test_repeat_group_key_is_the_identity_objective_and_declared_group_fields() -> None:
    schema = declaration()
    first = experience(schema, run_id="run-1", seq=0)

    assert repeat_group_key(schema, first) != repeat_group_key(
        schema, experience(schema, run_id="run-2", seq=0, baseline="tuned")
    )
    assert repeat_group_key(schema, first) != repeat_group_key(
        schema, experience(schema, run_id="run-2", seq=0, knob="chunk")
    )
    assert repeat_group_key(schema, first) == repeat_group_key(
        schema, replace(experience(schema, run_id="run-2", seq=0), change={**first.change, "summary": "Other words."})
    )


def test_a_schema_with_no_group_fields_groups_by_identity_and_objective() -> None:
    base = declaration()
    ungrouped = replace(
        base,
        baseline=tuple(replace(item, group=False) for item in base.baseline),
        change=tuple(replace(item, group=False) for item in base.change),
    )
    first = replace(experience(ungrouped, run_id="run-1", seq=0, knob="page_size"))
    second = replace(experience(ungrouped, run_id="run-2", seq=0, knob="chunk"))

    view = QueryViewBuilder().build(ungrouped, stored(first, second))

    assert [group.member_ids for group in view.groups.values()] == [tuple(sorted((first.id, second.id)))]


def test_notes_neither_split_a_repeat_group_nor_enter_exact_lookup() -> None:
    schema = declaration()
    plain = experience(schema, run_id="run-1", seq=0)
    noted = replace(experience(schema, run_id="run-2", seq=0), notes={"interconnect": "XGMI saturated."})

    view = QueryViewBuilder().build(schema, stored(plain, noted))

    assert view.groups[repeat_group_key(schema, plain)].member_ids == tuple(sorted((plain.id, noted.id)))
    assert not [field for field in view.field_lookup if field.startswith("notes")]


def test_local_view_store_keeps_old_generation_and_moves_current(tmp_path: Path) -> None:
    schema = declaration()
    schemas = InMemorySchemaRegistry()
    experiences = InMemoryExperienceStore()
    service = ExperienceService(schemas, experiences)
    service.register_schema(schema)
    first = experience(schema, run_id="run-1", seq=0)
    service.submit_complete(first)
    views = LocalQueryViewStore(tmp_path)
    maintainer = QueryViewMaintainer(schemas, experiences, views)

    old_ref = maintainer.rebuild(schema.schema_ref)
    second = experience(schema, run_id="run-2", seq=0)
    service.submit_complete(second)
    current_ref = maintainer.rebuild(schema.schema_ref)

    assert old_ref != current_ref
    assert views.get_view(old_ref.view_id).visible_experience_ids == (first.id,)
    assert views.current_view(schema.schema_ref).ref == current_ref


def test_empty_query_view_is_a_valid_current_snapshot() -> None:
    schema = declaration()
    schemas = InMemorySchemaRegistry()
    experiences = InMemoryExperienceStore()
    views = InMemoryQueryViewStore()
    schemas.register_schema(schema)

    ref = QueryViewMaintainer(schemas, experiences, views).rebuild(schema.schema_ref)

    assert ref.local_page_sequence == 0
    assert views.current_view(schema.schema_ref).visible_experience_ids == ()


def test_view_changes_when_an_immutable_experience_changes() -> None:
    schema = declaration()
    first = experience(schema, run_id="run-1", seq=0)
    changed = replace(first, reflection={"text": "A different immutable record."})

    original = QueryViewBuilder().build(schema, stored(first))
    updated = QueryViewBuilder().build(schema, stored(changed))

    assert original.ref.view_id != updated.ref.view_id
