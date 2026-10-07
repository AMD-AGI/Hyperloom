from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hyperloom_kb import (
    METADATA_FIELDS,
    CandidateHit,
    CapabilityState,
    CapabilityUnavailable,
    Experience,
    ExperienceDeclaration,
    ExperienceService,
    ExperienceStatus,
    FieldDeclaration,
    FieldKind,
    FieldRole,
    FileRef,
    InMemoryExperienceStore,
    InMemoryQueryViewStore,
    InMemorySchemaRegistry,
    LeaseExpired,
    LocalRetrievalService,
    ObjectiveDeclaration,
    Provenance,
    QueryView,
    QueryViewBuilder,
    QueryViewMaintainer,
    ReadLeaseManager,
    RetrievalCapability,
    derive_experience_id,
)
from hyperloom_kb.retrieval import render_complete_experience

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


class StaticProvider:
    def __init__(
        self,
        capability: RetrievalCapability,
        hits: tuple[CandidateHit, ...],
    ) -> None:
        self.capability = capability
        self.hits = hits

    def recall(self, request, view, *, limit):
        return self.hits[:limit]


def setup(*, fuzzy_ready: bool = False, semantic_ready: bool = False):
    schema = declaration()
    schemas = InMemorySchemaRegistry()
    experiences = InMemoryExperienceStore()
    views = InMemoryQueryViewStore()
    service = ExperienceService(schemas, experiences)
    service.register_schema(schema)
    first = experience(schema, "run-1", "page_size")
    repeated = experience(schema, "run-2", "page_size")
    other = experience(schema, "run-3", "chunk")
    for item in (first, repeated, other):
        service.submit_complete(item)
    ref = QueryViewMaintainer(schemas, experiences, views).rebuild(
        schema.schema_ref,
        fuzzy_ready=fuzzy_ready,
        semantic_ready=semantic_ready,
    )
    return schema, experiences, views, ref, first, repeated, other


def test_exact_filter_organize_and_explicit_render() -> None:
    schema, experiences, views, ref, first, repeated, other = setup()
    read = LocalRetrievalService(experiences, views)
    lease = read.acquire_view(schema.schema_ref)

    exact = read.exact(
        {
            "identity.model": "qwen3",
            "baseline.config": "default",
            "change.knob": "page_size",
        },
        view=ref,
        lease_id=lease.lease_id,
    )
    filtered = read.query(
        {"outcome.decision": "keep"},
        view=ref,
        lease_id=lease.lease_id,
    )
    groups = read.organize(
        (exact, filtered),
        view=ref,
        lease_id=lease.lease_id,
    )
    rendered = read.render(
        (first.id, other.id),
        view=ref,
        lease_id=lease.lease_id,
    )
    result = read.result(
        groups,
        rendered,
        view=ref,
        lease_id=lease.lease_id,
        executed_capabilities=(
            RetrievalCapability.EXACT,
            RetrievalCapability.FILTER,
        ),
    )

    assert {hit.experience_id for hit in exact} == {first.id, repeated.id}
    assert {hit.experience_id for hit in filtered} == {
        first.id,
        repeated.id,
        other.id,
    }
    assert len(groups) == 2
    assert groups[0].annotations.member_count in {1, 2}
    assert "Annotations:" in result.rendered.text
    assert result.rendered.rendered_refs[0].id == first.id
    assert result.executed_capabilities == (
        RetrievalCapability.EXACT,
        RetrievalCapability.FILTER,
    )


def test_fuzzy_and_semantic_are_independent_optional_capabilities() -> None:
    schema, experiences, views, ref, first, _, _ = setup(
        fuzzy_ready=True,
        semantic_ready=False,
    )
    provider = StaticProvider(
        RetrievalCapability.FUZZY,
        (
            CandidateHit(first.id, RetrievalCapability.FUZZY, 0.8),
            CandidateHit("exp-00000000000000000000000000000000", RetrievalCapability.FUZZY, 1.0),
        ),
    )
    read = LocalRetrievalService(experiences, views, providers=(provider,))
    lease = read.acquire_view(schema.schema_ref)

    fuzzy = read.recall(
        RetrievalCapability.FUZZY,
        {"identity": {"model": "qwen"}},
        view=ref,
        lease_id=lease.lease_id,
    )

    assert fuzzy == (CandidateHit(first.id, RetrievalCapability.FUZZY, 0.8),)
    with pytest.raises(CapabilityUnavailable):
        read.recall(
            RetrievalCapability.SEMANTIC,
            {"text": "scheduler"},
            view=ref,
            lease_id=lease.lease_id,
        )
    assert (
        views.current_view(schema.schema_ref).capabilities[RetrievalCapability.SEMANTIC] is CapabilityState.UNAVAILABLE
    )


def test_read_lease_pins_old_view_while_current_advances() -> None:
    schema, experiences, views, old_ref, first, _, _ = setup()
    read = LocalRetrievalService(experiences, views)
    lease = read.acquire_view(schema.schema_ref)
    new = experience(schema, "run-4", "new-knob")
    experiences.insert_complete(new)
    current_ref = QueryViewMaintainer(
        InMemorySchemaRegistryProxy(schema),
        experiences,
        views,
    ).rebuild(schema.schema_ref)

    old_hits = read.query({}, view=old_ref, lease_id=lease.lease_id)

    assert current_ref != old_ref
    assert new.id not in {hit.experience_id for hit in old_hits}
    assert first.id in {hit.experience_id for hit in old_hits}


class InMemorySchemaRegistryProxy:
    def __init__(self, schema: ExperienceDeclaration) -> None:
        self.schema = schema

    def register_schema(self, declaration: ExperienceDeclaration) -> str:
        raise AssertionError("not used")

    def get_schema(self, schema_ref: str) -> ExperienceDeclaration | None:
        return self.schema if schema_ref == self.schema.schema_ref else None


def test_expired_or_released_lease_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    schema, experiences, views, ref, _, _, _ = setup()
    clock = [100.0]
    monkeypatch.setattr("hyperloom_kb.retrieval.time.monotonic", lambda: clock[0])
    leases = ReadLeaseManager(max_ttl_seconds=5)
    read = LocalRetrievalService(experiences, views, leases=leases)
    lease = read.acquire_view(schema.schema_ref, ttl_seconds=2)
    clock[0] = 103.0

    with pytest.raises(LeaseExpired):
        read.query({}, view=ref, lease_id=lease.lease_id)

    current = read.acquire_view(schema.schema_ref)
    leases.release(current.lease_id)
    with pytest.raises(LeaseExpired):
        read.query({}, view=ref, lease_id=current.lease_id)


def test_a_render_budget_keeps_whole_records_and_names_only_those_it_shows() -> None:
    schema, experiences, views, ref, first, _, other = setup()
    read = LocalRetrievalService(experiences, views)
    lease = read.acquire_view(schema.schema_ref)
    both = (first.id, other.id)
    full = read.render(both, view=ref, lease_id=lease.lease_id, budget_chars=None)
    first_only = read.render((first.id,), view=ref, lease_id=lease.lease_id, budget_chars=None)

    fits_one = read.render(both, view=ref, lease_id=lease.lease_id, budget_chars=len(full.text) - 1)
    fits_none = read.render(both, view=ref, lease_id=lease.lease_id, budget_chars=len(first_only.text) - 1)
    fits_all = read.render(both, view=ref, lease_id=lease.lease_id, budget_chars=len(full.text))

    assert (fits_one.text, [item.id for item in fits_one.rendered_refs], fits_one.truncated) == (
        first_only.text,
        [first.id],
        True,
    )
    assert (fits_none.text, fits_none.rendered_refs, fits_none.truncated) == ("", (), True)
    assert (fits_all.text, fits_all.truncated) == (full.text, False)


def test_complete_renderer_without_budget_keeps_every_knowledge_field_and_no_metadata() -> None:
    schema = declaration()
    experiences = InMemoryExperienceStore()
    views = InMemoryQueryViewStore()
    schemas = InMemorySchemaRegistry()
    service = ExperienceService(schemas, experiences)
    service.register_schema(schema)
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
    service.submit_complete(record)
    ref = QueryViewMaintainer(schemas, experiences, views).rebuild(schema.schema_ref)
    read = LocalRetrievalService(experiences, views, renderer=render_complete_experience)
    lease = read.acquire_view(schema.schema_ref)

    rendered = read.render((record.id,), view=ref, lease_id=lease.lease_id, budget_chars=None)

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


def test_restricted_view_limits_recall_but_keeps_full_group_annotations() -> None:
    schema = declaration()
    experiences = InMemoryExperienceStore()
    schemas = InMemorySchemaRegistry()
    service = ExperienceService(schemas, experiences)
    service.register_schema(schema)
    kept = experience(schema, "run-1", "page_size")
    reverted = replace(
        experience(schema, "run-2", "page_size"),
        outcome={"decision": "revert", "value": 95.0},
    )
    for item in (kept, reverted):
        service.submit_complete(item)
    full = QueryViewBuilder().build(schema, experiences.list_experiences(schema.schema_ref))

    restricted = QueryViewBuilder().restrict(full, (kept.id,))
    views = InMemoryQueryViewStore()
    views.publish_view(restricted)
    read = LocalRetrievalService(experiences, views)
    lease = read.acquire_view(schema.schema_ref)
    hits = read.query({"identity.model": "qwen3"}, view=restricted.ref, lease_id=lease.lease_id)

    assert [hit.experience_id for hit in hits] == [kept.id]
    assert restricted.ref.view_id != full.ref.view_id
    assert QueryView.from_dict(restricted.to_dict()) == restricted
    group = restricted.groups[restricted.experience_groups[kept.id]]
    assert set(group.member_ids) == {kept.id, reverted.id}
    assert group.annotations.decision_counts == {"keep": 1, "revert": 1}
