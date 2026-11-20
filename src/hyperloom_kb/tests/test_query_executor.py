# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from datetime import datetime, timezone


from hyperloom_kb import (
    Experience,
    ExperienceDeclaration,
    ExperienceStatus,
    FieldDeclaration,
    FieldKind,
    FieldRole,
    InMemoryExperienceStore,
    LexicalFuzzyProvider,
    LocalRetrievalService,
    ObjectiveDeclaration,
    Provenance,
    QueryExecutor,
    QuerySignal,
    ResultShape,
    derive_experience_id,
)
from hyperloom_kb.tests.view_fixtures import build_view, submit_complete

NOW = datetime(2026, 9, 23, tzinfo=timezone.utc)


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
    experiences = InMemoryExperienceStore()
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
    submit_complete(schema, experiences, target, distractor)
    executor = QueryExecutor(
        LocalRetrievalService(experiences, build_view(schema, experiences)), LexicalFuzzyProvider(experiences, schema)
    )
    return schema, executor, target, distractor


def _signals(*, model_weight: float, text_weight: float = 0.9) -> tuple[QuerySignal, ...]:
    return (
        QuerySignal(field="identity.model", value="Qwen3-8B", weight=model_weight),
        QuerySignal(text="KV bandwidth decode", weight=text_weight),
    )


def test_weight_increase_changes_only_matching_group_contribution() -> None:
    schema, executor, target, distractor = _stack()
    configuration = ResultShape()

    low = executor.execute(_signals(model_weight=0.1), configuration)
    high = executor.execute(_signals(model_weight=1.0), configuration)
    low_scores = {group.representative_ids[0]: group.score for group in low.groups}
    high_scores = {group.representative_ids[0]: group.score for group in high.groups}

    assert high_scores[target.id] - low_scores[target.id] == 0.9
    assert high_scores[distractor.id] == low_scores[distractor.id]


def test_renderer_exposes_conditions_baseline_outcome_and_repeat_support() -> None:
    schema, executor, target, _ = _stack()
    result = executor.execute(
        _signals(model_weight=1.0),
        ResultShape(),
    )

    section = result.rendered.text.split(f"Experience {target.id}\n", 1)[1]
    annotations, record = section.split("Repeat Group Annotations:\n", 1)[1].split("\nRecord:\n", 1)
    rendered, _ = json.JSONDecoder().raw_decode(record)
    assert json.loads(annotations)["member_count"] == 1
    assert rendered["identity"] == target.identity
    assert rendered["baseline"] == {"config": "default", "value": 100.0}
    assert rendered["outcome"] == {"decision": "keep", "value": 110.0}
