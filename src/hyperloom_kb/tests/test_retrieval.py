from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from hyperloom_kb import (
    METADATA_FIELDS,
    Experience,
    ExperienceDeclaration,
    ExperienceStatus,
    FieldDeclaration,
    FieldKind,
    FieldRole,
    FileRef,
    InMemoryExperienceStore,
    LocalRetrievalService,
    ObjectiveDeclaration,
    Provenance,
    QueryViewBuilder,
    derive_experience_id,
)
from hyperloom_kb.retrieval import render_complete_experience
from hyperloom_kb.tests.logs import LOG
from hyperloom_kb.tests.view_fixtures import build_view, submit_complete

NOW = datetime(2026, 9, 17, tzinfo=timezone.utc)
TRACE = FileRef("traces/decode.json", "b" * 64, 4096)


def declaration() -> ExperienceDeclaration:
    return ExperienceDeclaration(
        objectives=(ObjectiveDeclaration("throughput@v1", "Maximize throughput."),),
        identity=(
            FieldDeclaration("model", "Model."),
            FieldDeclaration("gpu", "GPU."),
        ),
        baseline=(FieldDeclaration("config", "Baseline configuration.", group=True),),
        change=(
            FieldDeclaration("knob", "Changed knob.", group=True),
            FieldDeclaration("summary", "What changed.", role=FieldRole.SUMMARY),
            FieldDeclaration("content", "The change.", kind=FieldKind.TEXT),
        ),
        outcome=(
            FieldDeclaration("decision", "Decision.", role=FieldRole.DECISION, values=("keep", "revert")),
            FieldDeclaration("value", "Throughput.", kind=FieldKind.NUMBER, role=FieldRole.MEASUREMENT),
            FieldDeclaration("traces", "Profiles taken.", kind=FieldKind.FILE, many=True),
        ),
        reflection=(FieldDeclaration("text", "Reflection.", kind=FieldKind.TEXT),),
    )


def experience(schema: ExperienceDeclaration, run_id: str, knob: str) -> Experience:
    return Experience(
        id=derive_experience_id("test", run_id, 0),
        run_id=run_id,
        seq=0,
        created_at=NOW,
        completed_at=NOW,
        identity={"model": "qwen3", "gpu": "mi355x"},
        objective="throughput@v1",
        baseline={"config": "default"},
        provenance=Provenance("test", "1"),
        schema_ref=schema.schema_ref,
        status=ExperienceStatus.COMPLETE,
        rationale={"reasoning": f"Test {knob} against the measured bottleneck."},
        change={"knob": knob, "summary": f"Change {knob}."},
        outcome={"decision": "keep", "value": 110.0},
        reflection={"text": "Throughput improved."},
    )


def setup():
    schema = declaration()
    experiences = InMemoryExperienceStore()
    first = experience(schema, "run-1", "page_size")
    repeated = experience(schema, "run-2", "page_size")
    other = experience(schema, "run-3", "chunk")
    submit_complete(schema, experiences, first, repeated, other)
    view = build_view(schema, experiences)
    return experiences, view, first, repeated, other


def test_exact_organize_and_explicit_render() -> None:
    experiences, view, first, repeated, other = setup()
    read = LocalRetrievalService(experiences, view)

    exact = read.exact({"identity.model": "qwen3", "baseline.config": "default", "change.knob": "page_size"})
    filtered = read.exact({"outcome.decision": "keep"})
    groups = read.organize((exact, filtered))
    rendered = read.render((first.id, other.id))

    assert {hit.experience_id for hit in exact} == {first.id, repeated.id}
    assert {hit.experience_id for hit in filtered} == {
        first.id,
        repeated.id,
        other.id,
    }
    assert len(groups) == 2
    assert groups[0].annotations.member_count in {1, 2}
    assert "Annotations:" in rendered.text
    assert rendered.rendered_refs[0].id == first.id


def test_a_view_stays_as_built_while_its_store_takes_new_experiences() -> None:
    experiences, view, first, _, _ = setup()
    read = LocalRetrievalService(experiences, view)
    new = experience(declaration(), "run-4", "new-knob")
    experiences.insert_complete(new)

    hits = {hit.experience_id for hit in read.exact({})}

    assert new.id not in hits and first.id in hits


def test_a_render_budget_keeps_whole_records_and_names_only_those_it_shows() -> None:
    experiences, view, first, _, other = setup()
    read = LocalRetrievalService(experiences, view)
    both = (first.id, other.id)
    full = read.render(both, budget_chars=None)
    first_only = read.render((first.id,), budget_chars=None)
    other_only = read.render((other.id,), budget_chars=None)
    smallest = min(len(first_only.text), len(other_only.text))

    fits_one = read.render(both, budget_chars=len(full.text) - 1)
    fits_none = read.render(both, budget_chars=smallest - 1)
    fits_all = read.render(both, budget_chars=len(full.text))

    assert (fits_one.text, [item.id for item in fits_one.rendered_refs], fits_one.truncated) == (
        first_only.text,
        [first.id],
        True,
    )
    assert (fits_none.text, fits_none.rendered_refs, fits_none.truncated) == ("", (), True)
    assert (fits_all.text, fits_all.truncated) == (full.text, False)


def test_a_record_too_large_for_the_budget_does_not_keep_out_the_records_after_it() -> None:
    experiences, view, first, _, other = setup()
    read = LocalRetrievalService(experiences, view)
    alone = {experience.id: read.render((experience.id,), budget_chars=None).text for experience in (first, other)}
    large, small = sorted(alone, key=lambda experience_id: len(alone[experience_id]), reverse=True)
    assert len(alone[large]) > len(alone[small])

    shown = read.render((large, small), budget_chars=len(alone[small]))

    assert (shown.text, [item.id for item in shown.rendered_refs], shown.truncated) == (alone[small], [small], True)


def test_complete_renderer_without_budget_keeps_every_knowledge_field_and_no_metadata() -> None:
    schema = declaration()
    experiences = InMemoryExperienceStore()
    long_reasoning = "Measured decode stalls point at page size. " * 400
    record = replace(
        experience(schema, "run-long", "page_size"),
        rationale={
            "preconditions": ("measured_baseline_tput=100.0",),
            "reasoning": long_reasoning,
            "alternatives": ("Chunking was already exhausted.",),
        },
        change={"knob": "page_size", "summary": "Change page_size.", "content": "--page-size 32\n--max-num-seqs 256"},
    )
    submit_complete(schema, experiences, record)
    read = LocalRetrievalService(experiences, build_view(schema, experiences), renderer=render_complete_experience)

    rendered = read.render((record.id,), budget_chars=None)

    assert rendered.truncated is False
    assert rendered.text.startswith(f"Experience {record.id}\n")
    record_json = json.loads(rendered.text.split("\nRecord:\n", 1)[1])
    assert record_json == record.knowledge()
    assert not METADATA_FIELDS & set(record_json)
    annotations_json = rendered.text.split("Repeat Group Annotations:\n", 1)[1].split("\nRecord:\n", 1)[0]
    assert json.loads(annotations_json)["decision_counts"] == {"keep": 1}


def test_a_file_renders_as_its_name_size_and_where_it_is_read_never_its_content() -> None:
    schema = declaration()
    record = replace(
        experience(schema, "run-file", "page_size"), outcome={"decision": "keep", "value": 110.0, "traces": (TRACE,)}
    )
    store = InMemoryExperienceStore()
    store.insert_complete(record)
    view = QueryViewBuilder().build(schema, store.list_experiences(schema.schema_ref))

    located = render_complete_experience(record, view, file_path=lambda ref: Path("/kb/files") / ref.sha256)
    unlocated = render_complete_experience(record, view)

    assert json.loads(located.split("\nRecord:\n", 1)[1])["outcome"]["traces"] == [
        {"file": "traces/decode.json", "bytes": 4096, "path": f"/kb/files/{TRACE.sha256}"}
    ]
    assert json.loads(unlocated.split("\nRecord:\n", 1)[1])["outcome"]["traces"] == [
        {"file": "traces/decode.json", "bytes": 4096, "sha256": TRACE.sha256}
    ]


def test_raw_logs_render_like_files_and_a_record_without_them_renders_none() -> None:
    schema = declaration()
    logged = replace(experience(schema, "run-logged", "page_size"), raw_logs=(LOG,))
    unlogged = experience(schema, "run-unlogged", "page_size")
    store = InMemoryExperienceStore()
    for record in (logged, unlogged):
        store.insert_complete(record)
    view = QueryViewBuilder().build(schema, store.list_experiences(schema.schema_ref))

    def rendered(record: Experience) -> dict[str, Any]:
        text = render_complete_experience(record, view, file_path=lambda ref: Path("/kb/files") / ref.sha256)
        return json.loads(text.split("\nRecord:\n", 1)[1])

    assert rendered(logged)["raw_logs"] == [{"file": LOG.name, "bytes": LOG.bytes, "path": f"/kb/files/{LOG.sha256}"}]
    assert "raw_logs" not in rendered(unlogged)


def test_restricted_view_limits_recall_but_keeps_full_group_annotations() -> None:
    schema = declaration()
    experiences = InMemoryExperienceStore()
    kept = experience(schema, "run-1", "page_size")
    reverted = replace(
        experience(schema, "run-2", "page_size"),
        outcome={"decision": "revert", "value": 95.0},
    )
    submit_complete(schema, experiences, kept, reverted)
    full = QueryViewBuilder().build(schema, experiences.list_experiences(schema.schema_ref))

    restricted = QueryViewBuilder().restrict(full, (kept.id,))
    hits = LocalRetrievalService(experiences, restricted).exact({"identity.model": "qwen3"})

    assert [hit.experience_id for hit in hits] == [kept.id]
    assert restricted.ref.view_id != full.ref.view_id
    group = restricted.groups[restricted.experience_groups[kept.id]]
    assert set(group.member_ids) == {kept.id, reverted.id}
    assert group.annotations.decision_counts == {"keep": 1, "revert": 1}
