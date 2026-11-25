# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import functools
import os
import json
import types
from datetime import datetime, timezone

import httpx
import openai
import pytest

from hyperloom_kb import (
    LEXICAL_FUZZY_PROVIDER_REF,
    Experience,
    ExperienceDeclaration,
    ExperienceStatus,
    FieldDeclaration,
    FieldKind,
    FieldRole,
    GatewayPlannerBackend,
    InMemoryExperienceStore,
    InMemoryQueryViewStore,
    KnowledgeReadService,
    LexicalFuzzyProvider,
    LLMQueryPlanner,
    LocalRetrievalService,
    ObjectiveDeclaration,
    PlannerConfiguration,
    PlannerExecutionError,
    PlannerGatewayConfig,
    Provenance,
    QueryExecutor,
    QueryPlanValidationError,
    ReadRequest,
    ReadStatus,
    ReadTrace,
    ResultShape,
    RetrievalCapability,
    RetrievalConfiguration,
    WeightedQueryPlan,
    derive_experience_id,
)
from hyperloom_common.llm import llm_config
from hyperloom_kb.tests.view_fixtures import publish_view, submit_complete

NOW = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _declaration() -> ExperienceDeclaration:
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
            FieldDeclaration("content", "The change.", kind=FieldKind.TEXT),
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
        baseline={"config": "default"},
        provenance=Provenance("read-test", "1"),
        schema_ref=schema.schema_ref,
        status=ExperienceStatus.COMPLETE,
        rationale={"reasoning": reasoning},
        change={"knob": knob, "summary": f"Change {knob}."},
        outcome={"decision": "keep", "value": 110.0},
        reflection={"text": "The measured result improved."},
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
    experiences = InMemoryExperienceStore()
    views = InMemoryQueryViewStore()
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
    submit_complete(schema, experiences, target, other)
    fuzzy = LexicalFuzzyProvider(experiences, schema)
    view = publish_view(schema, experiences, views, fuzzy_ready=True)
    retrieval = LocalRetrievalService(experiences, views, providers=(fuzzy,))
    configuration = RetrievalConfiguration.create(
        schema.schema_ref,
        "weighted-read@v1",
        limits=dict.fromkeys(RetrievalCapability, 20),
        provider_refs={
            RetrievalCapability.FUZZY: LEXICAL_FUZZY_PROVIDER_REF,
        },
        shape=ResultShape(ranking_policy_ref="max-hit-score@v1"),
    )
    return (
        schema,
        experiences,
        views,
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


GATEWAY = {
    "ANTHROPIC_BASE_URL": "https://gateway.test/Unified",
    "ANTHROPIC_API_KEY": "secret",
    "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: ${ANTHROPIC_API_KEY}",
}
OPENAI_GATEWAY = {
    "OPENAI_BASE_URL": "https://gateway.test/v1",
    "OPENAI_API_KEY": "sk",
    "OPENAI_CUSTOM_HEADERS": "X-Team: kb",
}


def _messages_reply(text: str, stop_reason: str) -> bytes:
    events = (
        {"type": "message_start", "message": {"content": [], "usage": {"input_tokens": 1, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    )
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


def _chat_reply(text: str, finish_reason: str) -> bytes:
    choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}
    return json.dumps(
        {"id": "c", "object": "chat.completion", "created": 0, "model": "m", "choices": [choice]}
    ).encode()


def _gateway(monkeypatch: pytest.MonkeyPatch, reply: bytes, status: int = 200) -> list[httpx.Request]:
    """Answers both protocols' clients with ``reply`` through the main flow's own client construction."""

    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, content=reply)

    transport = httpx.MockTransport(handle)
    stub = types.SimpleNamespace(
        Client=functools.partial(httpx.Client, transport=transport),
        ConnectError=httpx.ConnectError,
        ConnectTimeout=httpx.ConnectTimeout,
    )
    monkeypatch.setattr(llm_config, "_httpx_module", lambda: stub)
    monkeypatch.setattr(
        openai, "OpenAI", functools.partial(openai.OpenAI, http_client=httpx.Client(transport=transport))
    )
    return seen


def _complete(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> str:
    """One plan through the gateway ``env`` names, as the service's own environment."""

    for name in list(os.environ):
        if name.startswith(("ANTHROPIC_", "OPENAI_", "CLAUDE_", "CODEX_", "LOCAL_KB_PLANNER_")):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    backend = GatewayPlannerBackend(PlannerGatewayConfig.from_env())
    return backend.complete(system_prompt="system", user_payload="payload", temperature=0)


def test_the_planner_reads_through_the_anthropic_gateway_with_its_custom_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _gateway(monkeypatch, _messages_reply(_planner_response(), "end_turn"))

    text = _complete(monkeypatch, GATEWAY)

    [request] = seen
    body = json.loads(request.content)
    assert json.loads(text)["signals"] and str(request.url) == "https://gateway.test/Unified/v1/messages"
    assert request.headers["ocp-apim-subscription-key"] == request.headers["x-api-key"] == "secret"
    assert (body["model"], body["system"], body["temperature"]) == (llm_config.DEFAULT_CLAUDE_MODEL, "system", 0)
    assert "tool_choice" not in body and "tools" not in body


def test_an_openai_only_environment_plans_through_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _gateway(monkeypatch, _chat_reply(_planner_response(), "stop"))

    text = _complete(monkeypatch, OPENAI_GATEWAY)

    [request] = seen
    body = json.loads(request.content)
    assert json.loads(text)["signals"] and str(request.url) == "https://gateway.test/v1/chat/completions"
    assert request.headers["x-team"] == "kb" and request.headers["authorization"] == "Bearer sk"
    assert (body["model"], body["temperature"]) == (llm_config.DEFAULT_CODEX_MODEL, 0)
    assert [message["role"] for message in body["messages"]] == ["system", "user"]


@pytest.mark.parametrize(
    ("env", "reply"),
    [(GATEWAY, _messages_reply("{", "max_tokens")), (OPENAI_GATEWAY, _chat_reply("{", "length"))],
    ids=["anthropic", "openai"],
)
def test_a_reply_cut_at_the_token_limit_fails_the_plan(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], reply: bytes
) -> None:
    _gateway(monkeypatch, reply)

    with pytest.raises(PlannerExecutionError, match="truncated output at max_tokens"):
        _complete(monkeypatch, env)


def test_a_gateway_that_refuses_the_call_fails_the_plan_not_the_service(monkeypatch: pytest.MonkeyPatch) -> None:
    _gateway(monkeypatch, b'{"error": "missing subscription key"}', status=401)

    with pytest.raises(PlannerExecutionError, match="planner gateway failed with RuntimeError.*status=401"):
        _complete(monkeypatch, GATEWAY)


def test_the_planner_gateway_is_the_anthropic_side_with_a_key_else_the_openai_side() -> None:
    both = PlannerGatewayConfig.from_env({**OPENAI_GATEWAY, **GATEWAY, "LOCAL_KB_PLANNER_MODEL": "planner"})

    assert (both.protocol, both.model) == ("anthropic", "planner")
    assert both.headers == (("Ocp-Apim-Subscription-Key", "secret"),)
    assert PlannerGatewayConfig.from_env(OPENAI_GATEWAY).protocol == "openai"
    with pytest.raises(QueryPlanValidationError, match="no Anthropic or OpenAI key"):
        PlannerGatewayConfig.from_env({"ANTHROPIC_BASE_URL": "https://gateway.test"})


@pytest.mark.parametrize("field", ["identity.not_declared", "change.content", "baseline_identity.config"])
def test_planner_rejects_a_structured_field_exact_lookup_does_not_index(field: str) -> None:
    schema = _declaration()
    value = json.loads(_planner_response())
    value["signals"][0]["field"] = field
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


def test_read_service_plans_queries_and_returns_prompt_ready_evidence() -> None:
    (
        schema,
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
        experiences,
        views,
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
            submit_complete(schema, experiences, late)
            publish_view(schema, experiences, views, fuzzy_ready=True)
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
