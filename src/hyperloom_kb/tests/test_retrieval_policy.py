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
    ExperienceStatus,
    FieldDeclaration,
    FieldKind,
    FieldRole,
    InMemoryExperienceStore,
    InMemoryQueryViewStore,
    LexicalFuzzyProvider,
    ObjectiveDeclaration,
    Provenance,
    RepresentativePolicy,
    ResultShape,
    RetrievalCapability,
    RetrievalConfiguration,
    RetrievalPolicyError,
    derive_experience_id,
)
from hyperloom_kb.tests.view_fixtures import publish_view, submit_complete

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
    experiences = InMemoryExperienceStore()
    views = InMemoryQueryViewStore()
    submit_complete(schema, experiences, *records)
    publish_view(schema, experiences, views, fuzzy_ready=True)
    return experiences, views.current_view(schema.schema_ref)


def test_configuration_is_content_addressed_and_round_trips() -> None:
    schema = _declaration()
    config = RetrievalConfiguration.create(
        schema.schema_ref,
        "test-policy@v1",
        provider_refs={RetrievalCapability.FUZZY: LEXICAL_FUZZY_PROVIDER_REF},
        shape=ResultShape(representative_policy=RepresentativePolicy.FIRST_MEMBER),
    )

    assert RetrievalConfiguration.from_dict(config.to_dict()) == config
    assert set(config.capabilities) == set(RetrievalCapability)
    assert config.configuration_id.startswith("retrieval-config:sha256:")


def test_a_configuration_id_is_fixed_by_its_content() -> None:
    schema_ref = "schema:sha256:" + "0" * 64
    shape = ResultShape("weighted-signal-sum@v1", max_groups=7, render_budget_chars=None)

    default = RetrievalConfiguration.create(schema_ref, "pin@v1")
    shaped = RetrievalConfiguration.create(schema_ref, "pin@v1", shape=shape)

    assert default.configuration_id == (
        "retrieval-config:sha256:0c950abb6dd28a35d40ee6998fd44b2f3f367336ff088f03ac4fec7b17d82aa0"
    )
    assert shaped.configuration_id == (
        "retrieval-config:sha256:591d40ec7aca5a311aba96fc234ff46fbec2d7b8df618fd5e1ed21388fc292cf"
    )


def test_unbounded_render_budget_is_explicit_and_round_trips() -> None:
    schema = _declaration()
    unbounded = RetrievalConfiguration.create(
        schema.schema_ref, "test-policy@v1", shape=ResultShape(render_budget_chars=None)
    )

    assert unbounded.render_budget_chars is None
    assert RetrievalConfiguration.from_dict(unbounded.to_dict()) == unbounded
    with pytest.raises(RetrievalPolicyError, match="render_budget_chars"):
        RetrievalConfiguration.create(schema.schema_ref, "test-policy@v1", shape=ResultShape(render_budget_chars=0))


def test_configuration_cannot_silently_omit_fuzzy() -> None:
    schema = _declaration()

    stored = RetrievalConfiguration.create(schema.schema_ref, "invalid@v1").to_dict()

    with pytest.raises(RetrievalPolicyError, match="must enumerate"):
        RetrievalConfiguration.from_dict({**stored, "capabilities": ["exact", "filter", "semantic"]})


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

    hits = LexicalFuzzyProvider(experiences, schema).recall({"text": "fragmentation"}, view, limit=10)

    assert [hit.details["matched_tokens"] for hit in hits] == [matched]


def test_fuzzy_penalizes_explicit_model_size_mismatch() -> None:
    schema = _declaration()
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
    experiences, view = _views(schema, qwen14, qwen8)

    hits = LexicalFuzzyProvider(experiences, schema).recall({"text": "Qwen 14B scheduler batching"}, view, limit=10)

    assert [hit.experience_id for hit in hits[:2]] == [qwen14.id, qwen8.id]
