# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from hyperloom_kb import (
    LEXICAL_FUZZY_PROVIDER_REF,
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
    LexicalFuzzyProvider,
    LocalRetrievalService,
    ObjectiveDeclaration,
    Provenance,
    QueryRequest,
    QueryViewMaintainer,
    RepresentativePolicy,
    RetrievalCapability,
    RetrievalConfiguration,
    RetrievalPolicyError,
    RetrievalRunner,
    SemanticCandidateProvider,
    SemanticHit,
    derive_experience_id,
    provider_refs,
)

NOW = datetime(2026, 9, 21, tzinfo=timezone.utc)


def _declaration(*, content_search: float | None = None) -> ExperienceDeclaration:
    return ExperienceDeclaration(
        objectives=(ObjectiveDeclaration("throughput@v1", "Maximize throughput."),),
        identity=(
            FieldDeclaration("model", "Model."),
            FieldDeclaration("gpu", "GPU."),
        ),
        baseline=(FieldDeclaration("config", "Baseline.", group=True),),
        change=(
            FieldDeclaration("knob", "Knob.", group=True),
            FieldDeclaration("summary", "What changed.", role=FieldRole.SUMMARY, search=4),
            FieldDeclaration("content", "The change.", kind=FieldKind.TEXT, search=content_search),
        ),
        outcome=(FieldDeclaration("decision", "Decision.", role=FieldRole.DECISION, values=("keep", "revert")),),
        reflection=(FieldDeclaration("text", "Reflection.", kind=FieldKind.TEXT),),
    )


def _experience(
    schema: ExperienceDeclaration,
    run_id: str,
    knob: str,
    reasoning: str,
    *,
    model: str = "qwen3",
    content: str = "",
) -> Experience:
    return Experience(
        id=derive_experience_id("policy-test", run_id, 0),
        run_id=run_id,
        seq=0,
        created_at=NOW,
        completed_at=NOW,
        identity={"model": model, "gpu": "mi355x"},
        objective="throughput@v1",
        baseline={"config": "default"},
        provenance=Provenance("policy-test", "1"),
        schema_ref=schema.schema_ref,
        status=ExperienceStatus.COMPLETE,
        rationale={"reasoning": reasoning},
        change={"knob": knob, "summary": f"Change {knob}.", **({"content": content} if content else {})},
        outcome={"decision": "keep"},
        reflection={"text": "The measured result improved."},
    )


def _views(schema: ExperienceDeclaration, *records: Experience):
    schemas = InMemorySchemaRegistry()
    experiences = InMemoryExperienceStore()
    views = InMemoryQueryViewStore()
    service = ExperienceService(schemas, experiences)
    service.register_schema(schema)
    for item in records:
        service.submit_complete(item)
    QueryViewMaintainer(schemas, experiences, views).rebuild(schema.schema_ref, fuzzy_ready=True)
    return experiences, views.current_view(schema.schema_ref)


def _setup():
    schema = _declaration()
    schemas = InMemorySchemaRegistry()
    experiences = InMemoryExperienceStore()
    views = InMemoryQueryViewStore()
    service = ExperienceService(schemas, experiences)
    service.register_schema(schema)
    page = _experience(
        schema,
        "run-page",
        "page_size",
        "Page size may reduce allocator fragmentation.",
    )
    chunk = _experience(
        schema,
        "run-chunk",
        "chunk",
        "Larger chunks may reduce scheduler dispatch overhead.",
    )
    for item in (page, chunk):
        service.submit_complete(item)
    fuzzy = LexicalFuzzyProvider(experiences, schema)
    view = QueryViewMaintainer(schemas, experiences, views).rebuild(
        schema.schema_ref,
        fuzzy_ready=True,
        semantic_ready=False,
    )
    retrieval = LocalRetrievalService(experiences, views, providers=(fuzzy,))
    return schema, experiences, view, retrieval, page, chunk, fuzzy


def test_configuration_is_content_addressed_and_round_trips() -> None:
    schema = _declaration()
    config = RetrievalConfiguration.create(
        schema.schema_ref,
        "test-policy@v1",
        provider_refs={RetrievalCapability.FUZZY: LEXICAL_FUZZY_PROVIDER_REF},
        representative_policy=RepresentativePolicy.FIRST_MEMBER,
    )

    assert RetrievalConfiguration.from_dict(config.to_dict()) == config
    assert set(config.capabilities) == set(RetrievalCapability)
    assert config.configuration_id.startswith("retrieval-config:sha256:")


def test_unbounded_render_budget_is_explicit_and_round_trips() -> None:
    schema = _declaration()
    unbounded = RetrievalConfiguration.create(
        schema.schema_ref,
        "test-policy@v1",
        render_budget_chars=None,
    )

    assert unbounded.render_budget_chars is None
    assert RetrievalConfiguration.from_dict(unbounded.to_dict()) == unbounded
    with pytest.raises(RetrievalPolicyError, match="render_budget_chars"):
        RetrievalConfiguration.create(schema.schema_ref, "test-policy@v1", render_budget_chars=0)


def test_configuration_cannot_silently_omit_fuzzy() -> None:
    schema = _declaration()

    with pytest.raises(RetrievalPolicyError, match="must enumerate"):
        RetrievalConfiguration.create(
            schema.schema_ref,
            "invalid@v1",
            capabilities=(
                RetrievalCapability.EXACT,
                RetrievalCapability.FILTER,
                RetrievalCapability.SEMANTIC,
            ),
        )


def test_runner_executes_available_capabilities_and_records_unavailable() -> None:
    schema, _, view, retrieval, page, chunk, fuzzy = _setup()
    config = RetrievalConfiguration.create(
        schema.schema_ref,
        "parallel-tools@v1",
        provider_refs={RetrievalCapability.FUZZY: fuzzy.provider_ref},
        ranking_policy_ref="capability-priority@v1",
    )
    request = QueryRequest(
        schema.schema_ref,
        exact_where={"change.knob": "page_size"},
        filter_where={"identity.model": "qwen3"},
        intent="scheduler dispatch chunks",
    )

    result = RetrievalRunner(
        retrieval,
        provider_refs=provider_refs((fuzzy,)),
    ).execute(request, config, view=view)

    assert RetrievalCapability.EXACT in result.retrieval.executed_capabilities
    assert RetrievalCapability.FILTER in result.retrieval.executed_capabilities
    assert RetrievalCapability.FUZZY in result.retrieval.executed_capabilities
    assert result.retrieval.unavailable_capabilities == (RetrievalCapability.SEMANTIC,)
    assert result.representative_ids[0] == page.id
    assert {page.id, chunk.id} >= set(result.representative_ids)
    assert result.ranked_group_keys
    assert result.retrieval.rendered.rendered_refs
    assert result.candidate_count == 2


class _SemanticBackend:
    provider_ref = "test-semantic@v1"

    def __init__(self, experience_id: str) -> None:
        self.experience_id = experience_id

    def search(self, text: str, *, limit: int) -> tuple[SemanticHit, ...]:
        assert text
        return (SemanticHit(self.experience_id, 0.95, {"backend": "test"}),)[:limit]


def test_semantic_adapter_is_version_pinned() -> None:
    schema, experiences, _, _, page, _, fuzzy = _setup()
    semantic = SemanticCandidateProvider(_SemanticBackend(page.id))
    schemas = InMemorySchemaRegistry()
    schemas.register_schema(schema)
    views = InMemoryQueryViewStore()
    view = QueryViewMaintainer(schemas, experiences, views).rebuild(
        schema.schema_ref,
        fuzzy_ready=True,
        semantic_ready=True,
    )
    service = LocalRetrievalService(experiences, views, providers=(fuzzy, semantic))
    config = RetrievalConfiguration.create(
        schema.schema_ref,
        "semantic@v1",
        provider_refs={
            RetrievalCapability.FUZZY: fuzzy.provider_ref,
            RetrievalCapability.SEMANTIC: semantic.provider_ref,
        },
    )

    result = RetrievalRunner(
        service,
        provider_refs=provider_refs((fuzzy, semantic)),
    ).execute(
        QueryRequest(schema.schema_ref, intent="allocator fragmentation"),
        config,
        view=view,
    )

    assert RetrievalCapability.SEMANTIC in result.retrieval.executed_capabilities
    assert result.retrieval.unavailable_capabilities == ()
    assert page.id in result.representative_ids


def test_runner_rejects_provider_version_drift() -> None:
    schema, _, view, retrieval, _, _, fuzzy = _setup()
    config = RetrievalConfiguration.create(
        schema.schema_ref,
        "pinned@v1",
        provider_refs={RetrievalCapability.FUZZY: "different-provider@v1"},
    )

    with pytest.raises(RetrievalPolicyError, match="provider does not match"):
        RetrievalRunner(
            retrieval,
            provider_refs=provider_refs((fuzzy,)),
        ).execute(QueryRequest(schema.schema_ref, intent="scheduler"), config, view=view)


def test_fuzzy_finds_an_experience_by_its_notes() -> None:
    schema = _declaration()
    plain = _experience(schema, "run-plain", "page_size", "The measured bottleneck suggests this knob.")
    noted = replace(
        _experience(schema, "run-noted", "chunk", "The measured bottleneck suggests this knob."),
        notes={"interconnect": "XGMI links saturate during the all-reduce."},
    )
    experiences, view = _views(schema, plain, noted)

    hits = LexicalFuzzyProvider(experiences, schema).recall({"text": "xgmi links saturate"}, view, limit=10)

    assert [(hit.experience_id, hit.details["matched_tokens"]) for hit in hits] == [
        (noted.id, ["link", "saturate", "xgmi"]),
        (plain.id, []),
    ]


@pytest.mark.parametrize(("content_search", "matched"), [(None, []), (1.0, ["fragmentation"])])
def test_fuzzy_reads_only_the_fields_the_declaration_weights(content_search, matched) -> None:
    schema = _declaration(content_search=content_search)
    record = _experience(
        schema, "run-content", "page_size", "The measured bottleneck suggests this knob.", content="--fix fragmentation"
    )
    experiences, view = _views(schema, record)

    [hit] = LexicalFuzzyProvider(experiences, schema).recall({"text": "fragmentation"}, view, limit=10)

    assert hit.details["matched_tokens"] == matched


def test_fuzzy_penalizes_explicit_model_size_mismatch() -> None:
    schema = _declaration()
    schemas = InMemorySchemaRegistry()
    experiences = InMemoryExperienceStore()
    views = InMemoryQueryViewStore()
    service = ExperienceService(schemas, experiences)
    service.register_schema(schema)
    qwen14 = _experience(
        schema,
        "run-qwen14",
        "batch",
        "Wider batching may reduce scheduler overhead.",
        model="Qwen3-14B",
    )
    qwen8 = _experience(
        schema,
        "run-qwen8",
        "batch",
        "Wider batching may reduce scheduler overhead.",
        model="Qwen3-8B",
    )
    for item in (qwen14, qwen8):
        service.submit_complete(item)
    fuzzy = LexicalFuzzyProvider(experiences, schema)
    view = QueryViewMaintainer(schemas, experiences, views).rebuild(
        schema.schema_ref,
        fuzzy_ready=True,
    )
    retrieval = LocalRetrievalService(experiences, views, providers=(fuzzy,))
    config = RetrievalConfiguration.create(
        schema.schema_ref,
        "model-boundary@v1",
        provider_refs={RetrievalCapability.FUZZY: fuzzy.provider_ref},
        ranking_policy_ref="max-hit-score@v1",
    )

    result = RetrievalRunner(
        retrieval,
        provider_refs=provider_refs((fuzzy,)),
    ).execute(
        QueryRequest(schema.schema_ref, intent="Qwen 14B scheduler batching"),
        config,
        view=view,
    )

    assert result.representative_ids[:2] == (qwen14.id, qwen8.id)
