# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

import pytest

from hyperloom_kb import (
    ExperienceDeclaration,
    FieldDeclaration,
    FieldRole,
    LLMQueryPlanner,
    ObjectiveDeclaration,
    PlannerConfiguration,
    PlannerExecutionError,
    ReadRequest,
)


def _declaration() -> ExperienceDeclaration:
    return ExperienceDeclaration(
        objectives=(ObjectiveDeclaration("throughput@v1", "Maximize throughput."),),
        identity=(
            FieldDeclaration("model", "Model."),
            FieldDeclaration("precision", "Precision."),
        ),
        baseline=(FieldDeclaration("fingerprint", "Baseline.", group=True),),
        change=(FieldDeclaration("family", "Change family.", group=True),),
        outcome=(FieldDeclaration("decision", "Decision.", role=FieldRole.DECISION, values=("keep", "revert")),),
    )


class ContextPlannerBackend:
    model = "planner-component-test"

    def complete(self, *, user_payload: str, **_kwargs) -> str:
        request = json.loads(user_payload.splitlines()[0])["request"]
        context = request["context"]
        signals = []
        if context.get("model"):
            signals.append(
                {
                    "source_path": "context.model",
                    "field": "identity.model",
                    "value": context["model"],
                    "weight": 0.9,
                }
            )
        if context.get("precision"):
            signals.append(
                {
                    "source_path": "context.precision",
                    "field": "identity.precision",
                    "value": context["precision"],
                    "weight": 0.7,
                }
            )
        if context.get("observation"):
            signals.append(
                {
                    "source_path": "context.observation",
                    "text": context["observation"],
                    "weight": 1.0,
                }
            )
        return json.dumps({"signals": signals})


def _planner() -> LLMQueryPlanner:
    return LLMQueryPlanner(
        ContextPlannerBackend(),
        PlannerConfiguration.create("planner-component-test"),
    )


def test_planner_tracks_structured_value_replacement() -> None:
    planner = _planner()
    schema = _declaration()

    first = planner.plan(
        ReadRequest("Select the next action.", {"model": "Qwen3-8B"}),
        schema,
    )
    second = planner.plan(
        ReadRequest("Select the next action.", {"model": "Qwen3-14B"}),
        schema,
    )

    assert first.signals[0].value == "Qwen3-8B"
    assert second.signals[0].value == "Qwen3-14B"
    assert first.plan_id != second.plan_id


def test_planner_cannot_reference_a_deleted_context_field() -> None:
    schema = _declaration()

    with pytest.raises(PlannerExecutionError, match="requires signals"):
        _planner().plan(ReadRequest("Select the next action.", {}), schema)


def test_planner_ignores_unrelated_metadata() -> None:
    plan = _planner().plan(
        ReadRequest(
            "Select the next action.",
            {
                "model": "Qwen3-8B",
                "observation": "Decode is KV bandwidth bound.",
                "ui_color": "blue",
                "slack_message_timestamp": "123.456",
            },
        ),
        _declaration(),
    )

    assert {signal.source_path for signal in plan.signals} == {
        "context.model",
        "context.observation",
    }


def test_planner_preserves_conflicting_observations_as_grounded_text() -> None:
    observation = "Profiler says decode is bandwidth bound; a later note says host launch overhead may dominate."
    plan = _planner().plan(
        ReadRequest(
            "Select the next action.",
            {
                "model": "Qwen3-8B",
                "observation": observation,
            },
        ),
        _declaration(),
    )

    text_signal = next(signal for signal in plan.signals if not signal.is_structured)
    assert text_signal.text == observation
    assert text_signal.source_path == "context.observation"


def test_primary_preproposal_request_contains_no_future_candidate_state() -> None:
    request = ReadRequest(
        "Select the next framework optimization to benchmark.",
        {
            "model": "Qwen3-8B",
            "precision": "bf16",
            "observation": "Decode is KV bandwidth bound.",
            "benchmark_baseline": {"throughput": 100.0},
            "current_best": {"throughput": 112.0},
            "recent_results": [],
            "already_tried": [],
        },
    )

    assert "candidate_change" not in request.context
    assert "question" not in request.context


def test_planner_accepts_request_prefixed_grounding_path() -> None:
    class RequestPrefixedBackend:
        model = "request-prefixed"

        def complete(self, **_kwargs):
            return json.dumps(
                {
                    "signals": [
                        {
                            "source_path": "request.context.model",
                            "field": "identity.model",
                            "value": "Qwen3-8B",
                            "weight": 1.0,
                        }
                    ]
                }
            )

    planner = LLMQueryPlanner(
        RequestPrefixedBackend(),
        PlannerConfiguration.create("request-prefixed"),
    )

    plan = planner.plan(
        ReadRequest("Select the next action.", {"model": "Qwen3-8B"}),
        _declaration(),
    )

    assert plan.signals[0].source_path == "request.context.model"
