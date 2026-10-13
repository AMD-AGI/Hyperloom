"""Tests for the order in which the Implementer's history is assembled."""

from __future__ import annotations

from dataclasses import replace

from kernelforge.loop.prompt_view import ImplementerHistoryParts, compose_implementer_history
from kernelforge.loop.iteration_result import IterationResult
from kernelforge.loop.search_policy import OBJECTIVE_IMMEDIATE_CANONICAL_GAIN, SEARCH_MODE_EXPLOIT, SearchPolicyDecision

_EMPTY = ImplementerHistoryParts(
    long_horizon_header="",
    digest="",
    lessons="",
    ledger="",
    recent_results=(),
    analysis_evidence="",
    case_config_coverage="",
    ceiling_advisory="",
    uncommittable_new_paths="",
    search_policy=None,
    supervisor_ruling="",
    optimization_plan_path="",
)

_POLICY = SearchPolicyDecision(
    mode=SEARCH_MODE_EXPLOIT,
    reason_codes=("baseline",),
    objective_kind=OBJECTIVE_IMMEDIATE_CANONICAL_GAIN,
)


def _positions(history: str, markers: list[str]) -> list[int]:
    return [history.index(marker) for marker in markers]


def test_every_block_is_placed_above_the_ones_assembled_before_it():
    parts = replace(
        _EMPTY,
        long_horizon_header="HEADER",
        lessons="LESSONS",
        ledger="LEDGER",
        analysis_evidence="ANALYSIS",
        case_config_coverage="COVERAGE",
        ceiling_advisory="CEILING",
        uncommittable_new_paths="NEW_PATHS",
        search_policy=_POLICY,
        supervisor_ruling="RULING",
        optimization_plan_path="/plan.md",
    )

    history = compose_implementer_history(parts)

    top_to_bottom = [
        "## Required optimization plan",
        "HEADER",
        "## Latest Supervisor Ruling",
        "RULING",
        "## Search Policy",
        "NEW_PATHS",
        "CEILING",
        "COVERAGE",
        "ANALYSIS",
        "LESSONS",
        "LEDGER",
    ]
    positions = _positions(history, top_to_bottom)
    assert positions == sorted(positions)
    assert history.startswith("## Required optimization plan\nRead /plan.md ")
    assert "follow the ruling" in history


def test_the_digest_is_left_out_when_the_long_horizon_header_replaces_it():
    parts = replace(_EMPTY, digest="DIGEST", lessons="LESSONS", ledger="LEDGER")

    assert compose_implementer_history(parts) == "DIGEST\n\nLESSONS\n\nLEDGER"
    assert compose_implementer_history(replace(parts, long_horizon_header="HEADER")) == ("HEADER\n\nLESSONS\n\nLEDGER")


def test_recent_results_stand_in_when_there_is_no_digest_lesson_or_ledger():
    result = IterationResult(
        iteration=3,
        duration_sec=0.1,
        validation_passed=True,
        validation_summary="passed",
        wall_ms=0.8,
        mean_case_speedup=1.25,
        kept=True,
    )

    history = compose_implementer_history(replace(_EMPTY, recent_results=(result,)))

    assert history == "iter 3 KEEP mean_case_speedup=1.2500x wall=0.800ms"


def test_the_plan_instruction_mentions_the_ruling_only_when_there_is_one():
    history = compose_implementer_history(replace(_EMPTY, ledger="LEDGER", optimization_plan_path="/plan.md"))

    assert "The plan was synthesized from current evidence. " in history
    assert "Supervisor Ruling" not in history
