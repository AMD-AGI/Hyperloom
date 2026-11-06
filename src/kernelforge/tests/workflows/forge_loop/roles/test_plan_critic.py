"""Tests for the fail-open orchestration plan critic and the answer it is held to."""

from __future__ import annotations

import json

import pytest

from kernelforge.agent_backends import AgentRunResult
from kernelforge.workflows.forge_loop.roles.contracts import (
    CaseEvidence,
    DispatchPlan,
    LaneDrop,
    OrchestrationContext,
    PlanCriticOutcome,
    SpecialistRound,
    SynthesizedPlan,
)
from kernelforge.workflows.forge_loop.roles.plan_critic import (
    PlanCriticAgent,
    PlanCriticAnswer,
    build_plan_critic_prompts,
)


class _Backend:
    def __init__(self, result: AgentRunResult | Exception):
        self.result = result
        self.specs = []

    async def run(self, spec, usage=None):
        self.specs.append(spec)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _answer(verdict: str = "ACCEPT", review: str = "The plan is evidence-grounded.", *drops: dict) -> dict:
    return {"verdict": verdict, "review": review, "lane_narrowing": list(drops)}


def _answered(verdict: str = "ACCEPT", review: str = "The plan is evidence-grounded.", *drops: dict) -> AgentRunResult:
    """A session whose provider constrained the answer itself."""
    return AgentRunResult(structured_output=_answer(verdict, review, *drops))


def _two_lane_drafts() -> list[SynthesizedPlan]:
    return [
        SynthesizedPlan(text="# Lane 1", ground="a.py"),
        SynthesizedPlan(text="# Lane 2", ground="b.py"),
    ]


def _context(tmp_path) -> OrchestrationContext:
    workspace = tmp_path.resolve()
    source = workspace / "kernel.py"
    source.write_text("def kernel():\n    return 1\n")
    return OrchestrationContext(
        analysis_commit="abc123",
        workspace=str(workspace),
        gpu_target="gfx942",
        objective="mean case speedup",
        program_context="Optimize the kernel.",
        source_map_path=str(source),
        cases=(CaseEvidence(case_id="case-a", latency_ms=1.0),),
    )


def _specialists() -> SpecialistRound:
    return SpecialistRound(DispatchPlan(analysis_commit="abc123", assignments=()), (), {})


async def _review(tmp_path, result, drafts=None) -> tuple[PlanCriticOutcome, _Backend]:
    backend = _Backend(result)
    critic = PlanCriticAgent(backend=backend, timeout_sec=2, max_turns=4)
    outcome = await critic.review(
        context=_context(tmp_path),
        drafts=drafts or [SynthesizedPlan(text="# Plan\nUse vector loads.")],
        specialists=_specialists(),
    )
    return outcome, backend


def test_critic_prompt_uses_checklist_and_workspace_paths(tmp_path):
    context = _context(tmp_path)
    system, user = build_plan_critic_prompts(
        context=context,
        drafts=[SynthesizedPlan(text="# Plan\nUse vector loads.")],
        specialists=SpecialistRound(
            DispatchPlan(
                analysis_commit="abc123",
                assignments=(),
            ),
            (),
            {
                "successful_roles": [],
                "covered_cases": [],
                "missing_cases": ["case-a"],
                "failed_roles": [],
            },
        ),
    )
    payload = json.loads(user)

    assert payload["context"]["workspace"] == str(tmp_path.resolve())
    assert payload["context"]["source_map_path"] == str((tmp_path / "kernel.py").resolve())
    assert payload["draft_plan"].startswith("# Plan")
    assert "should continue to exist" in system
    assert "existing GEMM" in system
    assert "opportunity cost" in system
    assert "do not mechanically repeat every item" in system


@pytest.mark.asyncio
async def test_critic_runs_one_independent_read_only_session_held_to_its_answer(tmp_path):
    outcome, backend = await _review(tmp_path, _answered("ACCEPT"))

    assert outcome.verdict == "ACCEPT"
    assert outcome.review == "The plan is evidence-grounded."
    assert outcome.fail_open is False
    assert outcome.duration_sec >= 0
    assert "review" not in outcome.to_dict()
    assert len(backend.specs) == 1
    spec = backend.specs[0]
    assert spec.output_schema is PlanCriticAnswer
    assert spec.writable is False
    assert spec.tool_policy.read is True
    assert spec.tool_policy.search is True
    assert spec.tool_policy.write is False
    assert spec.tool_policy.shell is False
    assert spec.tool_policy.max_turns == 4


@pytest.mark.asyncio
async def test_an_answer_a_provider_returned_as_text_is_read_the_same(tmp_path):
    """A provider that cannot constrain its answer is told the schema and answers in prose around it."""
    text = "Here is my review.\n\n```json\n" + json.dumps(_answer("REPLACE", "Use the library GEMM.")) + "\n```\n"

    outcome, _ = await _review(tmp_path, AgentRunResult(text=text))

    assert outcome.verdict == "REPLACE"
    assert outcome.review == "Use the library GEMM."
    assert outcome.requires_revision is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        AgentRunResult(text=""),
        AgentRunResult(
            text="provider failure",
            end_reason="api_error",
            stderr_tail="gateway unavailable",
        ),
        AgentRunResult(
            text=("I will inspect the evidence.\n[session ended with SDK error: Reached maximum number of turns]"),
            end_reason="turn_cap",
        ),
        AgentRunResult(text="The plan needs a canonical comparison."),
        AgentRunResult(structured_output={"verdict": "MAYBE", "review": "x", "lane_narrowing": []}),
        AgentRunResult(structured_output={"verdict": "ACCEPT", "review": "   ", "lane_narrowing": []}),
        RuntimeError("provider crashed"),
    ],
    ids=["empty", "api-error", "turn-cap", "prose-only", "unknown-verdict", "empty-review", "provider-crash"],
)
async def test_critic_failure_records_no_verdict_and_still_publishes(
    tmp_path,
    result,
    caplog,
):
    with caplog.at_level(
        "WARNING",
        logger="kernelforge.workflows.forge_loop.roles.plan_critic",
    ):
        outcome, _ = await _review(tmp_path, result, drafts=[SynthesizedPlan(text="# Draft")])

    # An outage or an answer in no usable shape is not a judgement: reading this round as an ACCEPT is what told the
    # next round, and anyone reading the artifact afterwards, that a review the critic never delivered had passed it.
    assert outcome.verdict == "NOT_REVIEWED"
    assert outcome.fail_open is True
    assert outcome.requires_revision is False
    assert outcome.error
    assert outcome.lane_drops == ()
    assert outcome.duration_sec >= 0
    assert outcome.to_dict()["status"] == "CRITIC_ERROR"
    assert outcome.to_dict()["verdict"] == "NOT_REVIEWED"
    artifact = outcome.render_artifact()
    assert artifact.startswith("STATUS: CRITIC_ERROR")
    assert "VERDICT: NOT_REVIEWED" in artifact
    assert "VERDICT: ACCEPT" not in artifact
    # The draft still runs: this round loses its review, not its plan.
    assert "plan critic failed open to the draft" in caplog.text


@pytest.mark.asyncio
async def test_the_artifact_states_the_verdict_above_the_review(tmp_path):
    outcome, _ = await _review(tmp_path, _answered("REVISE", "Add the missing canonical comparison."))

    assert outcome.render_artifact() == "VERDICT: REVISE\n\nAdd the missing canonical comparison."


def _round_prompts(tmp_path, drafts):
    return build_plan_critic_prompts(
        context=_context(tmp_path),
        drafts=drafts,
        specialists=SpecialistRound(
            DispatchPlan(analysis_commit="abc123", assignments=()),
            (),
            {
                "successful_roles": [],
                "covered_cases": [],
                "missing_cases": [],
                "failed_roles": [],
            },
        ),
    )


def test_a_single_plan_is_reviewed_as_one_plan(tmp_path):
    """There is no division to review and no sibling to compare against."""
    system, user = _round_prompts(tmp_path, [SynthesizedPlan(text="# Plan\nUse vector loads.")])
    payload = json.loads(user)

    assert payload["draft_plan"] == "# Plan\nUse vector loads."
    assert "draft_lane_plans" not in payload
    assert "This round was divided into several lanes" not in system


def test_a_round_is_reviewed_with_its_division_in_view(tmp_path):
    """What a round raises cannot be asked of any lane on its own."""
    system, user = _round_prompts(
        tmp_path,
        [
            SynthesizedPlan(text="# Lane 1", ground="chunk_intra.py: the epilogue"),
            SynthesizedPlan(text="# Lane 2", ground="chunk.py: the dispatch gate"),
        ],
    )
    payload = json.loads(user)

    assert "draft_plan" not in payload
    assert payload["draft_lane_plans"] == [
        {
            "lane_id": 1,
            "ground": "chunk_intra.py: the epilogue",
            "joint": False,
            "fallback": "",
            "draft_plan": "# Lane 1",
        },
        {
            "lane_id": 2,
            "ground": "chunk.py: the dispatch gate",
            "joint": False,
            "fallback": "",
            "draft_plan": "# Lane 2",
        },
    ]
    assert "One verdict covers the round" in system
    assert "`lane_narrowing`" in system
    assert "At least one lane always runs" in system


def test_a_review_needs_something_to_review(tmp_path):
    """An empty round is a programming error, not a plan worth a critic call."""
    with pytest.raises(ValueError):
        _round_prompts(tmp_path, [])


@pytest.mark.asyncio
async def test_critic_error_detail_is_single_line_and_bounded(tmp_path):
    outcome, _ = await _review(tmp_path, RuntimeError("first line\n" + ("x" * 3000)))

    assert "\n" not in outcome.error
    assert len(outcome.error) <= 2000
    assert outcome.error.endswith("...")


@pytest.mark.asyncio
async def test_the_reviews_narrowing_reaches_the_round(tmp_path):
    outcome, backend = await _review(
        tmp_path,
        _answered(
            "REVISE",
            "The division buys one answer twice.",
            {"lane_id": 2, "reason": "it is lane 1's   change in different words"},
        ),
        drafts=_two_lane_drafts(),
    )

    assert outcome.verdict == "REVISE"
    assert outcome.lane_drops == (LaneDrop(lane_id=2, reason="it is lane 1's change in different words"),)
    assert len(backend.specs) == 1
    assert outcome.to_dict()["lane_drops"] == [{"lane_id": 2, "reason": "it is lane 1's change in different words"}]


@pytest.mark.asyncio
async def test_a_lane_named_twice_is_dropped_once_for_the_first_reason(tmp_path):
    outcome, _ = await _review(
        tmp_path,
        _answered(
            "REVISE",
            "Lane 2 repeats lane 1.",
            {"lane_id": 2, "reason": "it duplicates lane 1"},
            {"lane_id": 2, "reason": "its ground is unsupported"},
        ),
        drafts=_two_lane_drafts(),
    )

    assert outcome.lane_drops == (LaneDrop(lane_id=2, reason="it duplicates lane 1"),)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "drop",
    [{"lane_id": 0, "reason": "it duplicates lane 1"}, {"lane_id": 2, "reason": "   "}, {"lane_id": 2}],
    ids=["no-lane", "no-reason", "missing-reason"],
)
async def test_a_drop_that_names_no_lane_or_no_reason_is_no_review(tmp_path, drop):
    """A reason is what makes a narrowed round auditable afterwards, so a drop without one is an unusable answer."""
    outcome, _ = await _review(tmp_path, _answered("REVISE", "Narrow the round.", drop), drafts=_two_lane_drafts())

    assert outcome.fail_open is True
    assert outcome.lane_drops == ()


def test_a_lane_cannot_be_dropped_for_nothing():
    """The reason is what makes a narrowed round auditable afterwards."""
    with pytest.raises(ValueError, match="lane drop.reason"):
        LaneDrop(lane_id=2, reason="   ")
    with pytest.raises(ValueError, match="must be positive"):
        LaneDrop(lane_id=0, reason="it duplicates lane 1")


def test_one_lane_is_dropped_for_one_reason():
    with pytest.raises(ValueError, match="name each lane once"):
        PlanCriticOutcome(
            verdict="REVISE",
            lane_drops=(
                LaneDrop(lane_id=2, reason="it duplicates lane 1"),
                LaneDrop(lane_id=2, reason="its ground is unsupported"),
            ),
        )


def test_an_outage_and_a_judgement_are_never_recorded_together():
    """``NOT_REVIEWED`` is the whole of what a failed review leaves behind, and the only thing it leaves behind."""
    with pytest.raises(ValueError, match="must be recorded together"):
        PlanCriticOutcome(verdict="ACCEPT", error="backend timed out")
    with pytest.raises(ValueError, match="must be recorded together"):
        PlanCriticOutcome(verdict="NOT_REVIEWED")


def test_a_one_plan_review_keeps_the_budget_it_always_had():
    """Guards the scaling from moving the single-lane path it was not for."""
    critic = PlanCriticAgent(backend=_Backend(AgentRunResult(text="")), timeout_sec=600)

    assert critic._budget_for(1) == 600


def test_a_round_of_several_plans_is_several_times_the_reading():
    """A budget sized for one plan fails a round open, losing its verdict."""
    critic = PlanCriticAgent(
        backend=_Backend(AgentRunResult(text="")),
        timeout_sec=600,
        ceiling_sec=1800,
    )

    assert critic._budget_for(2) == 1200
    assert critic._budget_for(3) == 1800
    # Never past what the provider allows one call.
    assert critic._budget_for(9) == 1800
