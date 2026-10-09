# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from datetime import datetime, timezone

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
    PlannerProvenance,
    Provenance,
    QueryExecutor,
    QuerySignal,
    QueryViewMaintainer,
    RepresentativePolicy,
    RetrievalCapability,
    RetrievalConfiguration,
    WeightedQueryPlan,
    derive_experience_id,
)

NOW = datetime(2026, 9, 23, tzinfo=timezone.utc)
FROZEN = PlannerProvenance("frozen-case", "none", "none", "frozen-input")


def _declaration() -> ExperienceDeclaration:
    return ExperienceDeclaration(
        objectives=(ObjectiveDeclaration("throughput@v1", "Maximize throughput."),),
        identity=(
            FieldDeclaration("model", "Model."),
            FieldDeclaration("gpu", "GPU."),
        ),
        baseline=(
            FieldDeclaration("config", "Baseline.", group=True),
            FieldDeclaration("value", "Baseline throughput.", kind=FieldKind.NUMBER, role=FieldRole.MEASUREMENT),
        ),
        change=(
            FieldDeclaration("knob", "Knob.", group=True),
            FieldDeclaration("summary", "What changed.", role=FieldRole.SUMMARY, search=4),
        ),
        outcome=(
            FieldDeclaration("decision", "Decision.", role=FieldRole.DECISION, values=("keep", "revert")),
            FieldDeclaration("value", "Throughput.", kind=FieldKind.NUMBER, role=FieldRole.MEASUREMENT),
        ),
        reflection=(FieldDeclaration("text", "Reflection.", kind=FieldKind.TEXT),),
    )


def _experience(
    schema: ExperienceDeclaration,
    run_id: str,
    *,
    model: str,
    reasoning: str,
    knob: str,
) -> Experience:
    return Experience(
        id=derive_experience_id("executor-test", run_id, 0),
        run_id=run_id,
        seq=0,
        created_at=NOW,
        completed_at=NOW,
        identity={"model": model, "gpu": "mi300x"},
        objective="throughput@v1",
        baseline={"config": "default", "value": 100.0},
        provenance=Provenance("executor-test", "1"),
        schema_ref=schema.schema_ref,
        status=ExperienceStatus.COMPLETE,
        rationale={"reasoning": reasoning},
        change={"knob": knob, "summary": f"Change {knob}."},
        outcome={"decision": "keep", "value": 110.0},
        reflection={"text": "Measured result."},
    )


def _stack():
    schema = _declaration()
    schemas = InMemorySchemaRegistry()
    experiences = InMemoryExperienceStore()
    views = InMemoryQueryViewStore()
    service = ExperienceService(schemas, experiences)
    service.register_schema(schema)
    target = _experience(
        schema,
        "target",
        model="Qwen3-8B",
        reasoning="FP8 KV cache reduces decode KV bandwidth.",
        knob="fp8_kv",
    )
    distractor = _experience(
        schema,
        "distractor",
        model="Qwen3-14B",
        reasoning="FP8 KV cache reduces decode KV bandwidth.",
        knob="fp8_kv_other",
    )
    for item in (target, distractor):
        service.submit_complete(item)
    fuzzy = LexicalFuzzyProvider(experiences, schema)
    view = QueryViewMaintainer(schemas, experiences, views).rebuild(
        schema.schema_ref,
        fuzzy_ready=True,
    )
    executor = QueryExecutor(
        LocalRetrievalService(experiences, views, providers=(fuzzy,)),
        provider_refs={
            RetrievalCapability.FUZZY: LEXICAL_FUZZY_PROVIDER_REF,
        },
    )
    return schema, view, executor, target, distractor


def _configuration(
    schema_ref: str,
    *,
    ranking: str = "weighted-signal-sum@v1",
    representative: RepresentativePolicy = RepresentativePolicy.MATCHED_FIRST,
) -> RetrievalConfiguration:
    return RetrievalConfiguration.create(
        schema_ref,
        "executor-test-v1",
        limits={capability: 20 for capability in RetrievalCapability},
        provider_refs={
            RetrievalCapability.FUZZY: LEXICAL_FUZZY_PROVIDER_REF,
        },
        ranking_policy_ref=ranking,
        representative_policy=representative,
    )


def _plan(
    schema_ref: str,
    *,
    model_weight: float,
    text_weight: float = 0.9,
) -> WeightedQueryPlan:
    return WeightedQueryPlan.create(
        schema_ref,
        (
            QuerySignal.create(
                source_path="context.model",
                field="identity.model",
                value="Qwen3-8B",
                weight=model_weight,
            ),
            QuerySignal.create(
                source_path="context.observation",
                text="KV bandwidth decode",
                weight=text_weight,
            ),
        ),
        FROZEN,
    )


def test_weight_increase_changes_only_matching_group_contribution() -> None:
    schema, view, executor, target, distractor = _stack()
    configuration = _configuration(schema.schema_ref)

    low = executor.execute(
        _plan(schema.schema_ref, model_weight=0.1),
        configuration,
        view=view,
    )
    high = executor.execute(
        _plan(schema.schema_ref, model_weight=1.0),
        configuration,
        view=view,
    )
    low_scores = {group.representative_ids[0]: group.score for group in low.groups}
    high_scores = {group.representative_ids[0]: group.score for group in high.groups}

    assert high_scores[target.id] - low_scores[target.id] == 0.9
    assert high_scores[distractor.id] == low_scores[distractor.id]


def test_provider_scores_are_normalized_before_weighting() -> None:
    assert QueryExecutor._normalized_score(RetrievalCapability.EXACT, 0.2) == 1.0
    assert QueryExecutor._normalized_score(RetrievalCapability.FUZZY, -0.5) == 0.0
    assert QueryExecutor._normalized_score(RetrievalCapability.SEMANTIC, 3.0) == 1.0


def test_group_key_policy_controls_executor_order() -> None:
    schema, view, executor, _, _ = _stack()
    result = executor.execute(
        _plan(schema.schema_ref, model_weight=1.0),
        _configuration(schema.schema_ref, ranking="group-key@v1"),
        view=view,
    )

    assert tuple(group.group_key for group in result.groups) == tuple(
        sorted(group.group_key for group in result.groups)
    )


def test_renderer_exposes_conditions_baseline_outcome_and_repeat_support() -> None:
    schema, view, executor, target, _ = _stack()
    result = executor.execute(
        _plan(schema.schema_ref, model_weight=1.0),
        _configuration(schema.schema_ref),
        view=view,
    )

    section = result.rendered.text.split(f"Experience {target.id}\n", 1)[1]
    annotations, record = section.split("Repeat Group Annotations:\n", 1)[1].split("\nRecord:\n", 1)
    rendered, _ = json.JSONDecoder().raw_decode(record)
    assert json.loads(annotations)["member_count"] == 1
    assert rendered["identity"] == target.identity
    assert rendered["baseline"] == {"config": "default", "value": 100.0}
    assert rendered["outcome"] == {"decision": "keep", "value": 110.0}
