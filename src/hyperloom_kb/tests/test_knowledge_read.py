# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from hyperloom_kb import (
    LEXICAL_FUZZY_PROVIDER_REF,
    AnthropicPlannerBackend,
    Change,
    Experience,
    ExperienceDeclaration,
    ExperienceService,
    ExperienceStatus,
    FieldDeclaration,
    InMemoryExperienceStore,
    InMemoryQueryViewStore,
    InMemorySchemaRegistry,
    KnowledgeReadService,
    LexicalFuzzyProvider,
    LLMQueryPlanner,
    LocalRetrievalService,
    ObjectiveDeclaration,
    ObjectiveDirection,
    Outcome,
    PlannerConfiguration,
    PlannerExecutionError,
    PlannerGatewayConfig,
    Provenance,
    QueryExecutor,
    QueryViewMaintainer,
    ReadRequest,
    ReadStatus,
    ReadTrace,
    RetrievalCapability,
    RetrievalConfiguration,
    WeightedQueryPlan,
    derive_experience_id,
)

NOW = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _declaration() -> ExperienceDeclaration:
    return ExperienceDeclaration(
        identity=(
            FieldDeclaration("model", "Model."),
            FieldDeclaration("gpu", "GPU."),
        ),
        baseline_identity=(FieldDeclaration("config", "Baseline."),),
        change_identity=(FieldDeclaration("knob", "Knob."),),
        objectives=(
            ObjectiveDeclaration(
                "throughput@v1",
                ObjectiveDirection.HIGHER_IS_BETTER,
                "Throughput.",
            ),
        ),
        decisions=("keep", "revert"),
    )


def _experience(
    schema: ExperienceDeclaration,
    run_id: str,
    *,
    model: str,
    knob: str,
    reasoning: str,
) -> Experience:
    return Experience(
        id=derive_experience_id("read-test", run_id, 0),
        run_id=run_id,
        seq=0,
        created_at=NOW,
        completed_at=NOW,
        identity={"model": model, "gpu": "mi300x"},
        objective="throughput@v1",
        baseline_identity={"config": "default"},
        baseline_value=100.0,
        provenance=Provenance("read-test", "1"),
        schema_ref=schema.schema_ref,
        status=ExperienceStatus.COMPLETE,
        reasoning=reasoning,
        change=Change({"knob": knob}, f"Change {knob}.", kind="config"),
        outcome=Outcome("keep", 110.0),
        reflection="The measured result improved.",
    )


class FakePlannerBackend:
    model = "fake-planner"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def complete(self, *, system_prompt, user_payload, temperature):
        assert system_prompt
        assert user_payload
        assert temperature == 0
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _planner_response() -> str:
    return json.dumps(
        {
            "signals": [
                {
                    "source_path": "context.model",
                    "field": "identity.model",
                    "value": "Qwen3-8B",
                    "weight": 1.0,
                },
                {
                    "source_path": "context.observation",
                    "text": "KV bandwidth dominates decode.",
                    "weight": 0.9,
                },
            ],
        }
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
        "run-target",
        model="Qwen3-8B",
        knob="fp8_kv",
        reasoning="FP8 KV cache reduces decode KV bandwidth.",
    )
    other = _experience(
        schema,
        "run-other",
        model="Qwen3-14B",
        knob="fp8_kv",
        reasoning="FP8 KV cache reduces decode KV bandwidth.",
    )
    for item in (target, other):
        service.submit_complete(item)
    fuzzy = LexicalFuzzyProvider(experiences)
    view = QueryViewMaintainer(schemas, experiences, views).rebuild(
        schema.schema_ref,
        fuzzy_ready=True,
    )
    retrieval = LocalRetrievalService(experiences, views, providers=(fuzzy,))
    configuration = RetrievalConfiguration.create(
        schema.schema_ref,
        "weighted-read@v1",
        limits={capability: 20 for capability in RetrievalCapability},
        provider_refs={
            RetrievalCapability.FUZZY: LEXICAL_FUZZY_PROVIDER_REF,
        },
        ranking_policy_ref="max-hit-score@v1",
    )
    return (
        schema,
        schemas,
        experiences,
        views,
        service,
        view,
        retrieval,
        configuration,
        target,
        other,
    )


def _request(schema: ExperienceDeclaration) -> ReadRequest:
    return ReadRequest(
        "Select the next framework optimization to benchmark.",
        {
            "model": "Qwen3-8B",
            "observation": "KV bandwidth dominates decode.",
        },
    )


def test_llm_planner_retries_invalid_output_and_returns_validated_plan() -> None:
    schema = _declaration()
    backend = FakePlannerBackend(["not-json", _planner_response()])
    planner = LLMQueryPlanner(
        backend,
        PlannerConfiguration.create(backend.model),
    )

    plan = planner.plan(_request(schema), schema)

    assert backend.calls == 2
    assert {item.field for item in plan.signals if item.field} == {"identity.model"}
    assert WeightedQueryPlan.from_dict(plan.to_dict()) == plan


def test_anthropic_planner_backend_reads_text_response() -> None:
    seen = {}

    def opener(request, **kwargs):
        seen["url"] = request.full_url
        seen["timeout"] = kwargs["timeout"]
        return FakeHTTPResponse(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "submit_query_plan",
                        "input": json.loads(_planner_response()),
                    }
                ],
                "stop_reason": "tool_use",
            }
        )

    backend = AnthropicPlannerBackend(
        PlannerGatewayConfig(
            "https://planner.example/api",
            "secret",
            "test-model",
            timeout_seconds=5,
        ),
        opener=opener,
    )

    text = backend.complete(
        system_prompt="system",
        user_payload="payload",
        temperature=0,
    )

    assert json.loads(text)["signals"]
    assert seen == {
        "url": "https://planner.example/api/v1/messages",
        "timeout": 5,
    }


def test_planner_rejects_unknown_structured_field() -> None:
    schema = _declaration()
    value = json.loads(_planner_response())
    value["signals"][0]["field"] = "identity.not_declared"
    backend = FakePlannerBackend([json.dumps(value)])
    planner = LLMQueryPlanner(
        backend,
        PlannerConfiguration.create(backend.model, max_attempts=1),
    )

    with pytest.raises(PlannerExecutionError, match="not declared"):
        planner.plan(_request(schema), schema)


def test_planner_rejects_structured_value_not_copied_from_context() -> None:
    schema = _declaration()
    value = json.loads(_planner_response())
    value["signals"][0]["value"] = "Qwen3-14B"
    backend = FakePlannerBackend([json.dumps(value)])
    planner = LLMQueryPlanner(
        backend,
        PlannerConfiguration.create(backend.model, max_attempts=1),
    )

    with pytest.raises(PlannerExecutionError, match=r"does not copy context\.model"):
        planner.plan(_request(schema), schema)


def test_planner_rejects_more_than_eight_signals() -> None:
    schema = _declaration()
    signal = json.loads(_planner_response())["signals"][1]
    backend = FakePlannerBackend([json.dumps({"signals": [signal] * 9})])
    planner = LLMQueryPlanner(
        backend,
        PlannerConfiguration.create(backend.model, max_attempts=1),
    )

    with pytest.raises(PlannerExecutionError, match="at most 8 signals"):
        planner.plan(_request(schema), schema)


def test_a_planner_call_gives_up_after_twenty_seconds_unless_configured_otherwise() -> None:
    env = {"ANTHROPIC_BASE_URL": "https://planner.example", "ANTHROPIC_API_KEY": "secret", "CLAUDE_MODEL": "m"}

    assert PlannerGatewayConfig.from_env(env).timeout_seconds == 20.0
    assert PlannerGatewayConfig.from_env({**env, "LOCAL_KB_PLANNER_TIMEOUT_SECONDS": "45"}).timeout_seconds == 45.0


def test_anthropic_backend_reports_truncated_tool_output() -> None:
    backend = AnthropicPlannerBackend(
        PlannerGatewayConfig(
            "https://planner.example/api",
            "secret",
            "test-model",
        ),
        opener=lambda *_args, **_kwargs: FakeHTTPResponse(
            {
                "content": [],
                "stop_reason": "max_tokens",
            }
        ),
    )

    with pytest.raises(PlannerExecutionError, match="truncated"):
        backend.complete(
            system_prompt="system",
            user_payload="payload",
            temperature=0,
        )


def test_read_service_plans_queries_and_returns_prompt_ready_evidence() -> None:
    (
        schema,
        _,
        _,
        _,
        _,
        _,
        retrieval,
        configuration,
        target,
        other,
    ) = _stack()
    backend = FakePlannerBackend([_planner_response()])
    planner = LLMQueryPlanner(
        backend,
        PlannerConfiguration.create(backend.model),
    )
    traces: list[ReadTrace] = []
    service = KnowledgeReadService(
        schema,
        QueryExecutor(
            retrieval,
            provider_refs={
                RetrievalCapability.FUZZY: LEXICAL_FUZZY_PROVIDER_REF,
            },
        ),
        configuration,
        planner=planner,
        trace_sink=traces.append,
    )

    request = _request(schema)
    result = service.read(request.decision, request.context)

    assert result.status is ReadStatus.COMPLETED
    assert len(traces) == 1
    execution = traces[0].execution
    assert execution.groups[0].representative_ids == (target.id,)
    assert any(group.representative_ids == (other.id,) for group in execution.groups)
    assert result.prompt_block.startswith("=== Relevant Experience KB ===")
    assert "untrusted evidence" in result.prompt_block
    assert result.rendered_refs


def test_read_keeps_same_corpus_snapshot_while_planner_runs() -> None:
    (
        schema,
        schemas,
        experiences,
        views,
        experience_service,
        view,
        retrieval,
        configuration,
        _,
        _,
    ) = _stack()
    late = _experience(
        schema,
        "run-late",
        model="Qwen3-8B",
        knob="late_change",
        reasoning="KV bandwidth late result.",
    )

    class PublishingBackend(FakePlannerBackend):
        def complete(self, **kwargs):
            experience_service.submit_complete(late)
            QueryViewMaintainer(schemas, experiences, views).rebuild(
                schema.schema_ref,
                fuzzy_ready=True,
            )
            return super().complete(**kwargs)

    backend = PublishingBackend([_planner_response()])
    traces: list[ReadTrace] = []
    service = KnowledgeReadService(
        schema,
        QueryExecutor(
            retrieval,
            provider_refs={
                RetrievalCapability.FUZZY: LEXICAL_FUZZY_PROVIDER_REF,
            },
        ),
        configuration,
        planner=LLMQueryPlanner(
            backend,
            PlannerConfiguration.create(backend.model),
        ),
        trace_sink=traces.append,
    )
    request = _request(schema)

    result = service.read(request.decision, request.context)

    assert result.status is ReadStatus.COMPLETED
    assert traces[0].execution.view == view
    assert views.current_view(schema.schema_ref).ref != view
    assert late.id not in {member_id for group in traces[0].execution.groups for member_id in group.member_ids}


def test_direct_query_is_replayable_without_planner() -> None:
    (
        schema,
        _,
        _,
        _,
        _,
        view,
        retrieval,
        configuration,
        target,
        _,
    ) = _stack()
    request = _request(schema)
    planner = LLMQueryPlanner(
        FakePlannerBackend([_planner_response()]),
        PlannerConfiguration.create("fake-planner"),
    )
    plan = planner.plan(request, schema)
    runner = QueryExecutor(
        retrieval,
        provider_refs={
            RetrievalCapability.FUZZY: LEXICAL_FUZZY_PROVIDER_REF,
        },
    )

    first = runner.execute(plan, configuration, view=view)
    second = runner.execute(plan, configuration, view=view)

    assert first.groups == second.groups
    assert first.groups[0].representative_ids == (target.id,)


def test_disabled_read_is_a_no_op() -> None:
    schema = _declaration()
    service = KnowledgeReadService(
        schema,
        None,
        None,
        enabled=False,
    )

    request = _request(schema)
    result = service.read(request.decision, request.context)

    assert result.status is ReadStatus.DISABLED
    assert result.prompt_block == ""
    assert result.rendered_refs == ()
